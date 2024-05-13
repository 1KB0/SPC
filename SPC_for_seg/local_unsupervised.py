from torch.utils.data import DataLoader, Dataset
import copy
import torch
import torch.optim
import torch.nn.functional as F
from monai.transforms import RandAffined
from networks.Unet3d import BaselineUNet
from options import args_parser
from networks.models import DenseNet121
from utils import losses, ramps
from loss.centerloss import CenterLoss, step_center, co_center
from loss.softmatch import SoftMatch
from loss.pgcloss import PseudoGroupContrast
from FedAvg import cal_dist

args = args_parser()
center_loss_unlabeled = CenterLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]


# 一致性损失系数
def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, 40)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class UnsupervisedLocalUpdate(object):
    def __init__(self, args, dataset=None, length=None):
        self.ldr_train = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
        self.length = length

        net = BaselineUNet(in_channels=1, n_cls=3, n_filters=24)

        if torch.cuda.device_count() > 1:
            self.ema_model = torch.nn.DataParallel(net, device_ids=device_ids).to(device)
        else:
            self.ema_model = net.to(device)

        for param in self.ema_model.parameters():
            param.detach_()

        self.flag = True

        self.epoch = 0
        self.iter_num = 0
        self.base_lr = args.base_lr

        self.init_center = True
        self.softmatch = SoftMatch()
        self.pgcloss = PseudoGroupContrast()

    def train(self, args, net, op_dict, epoch, center_avg, c_t_avg, mean_avg, var_avg):
        net.train()
        self.optimizer = torch.optim.AdamW(net.parameters(), lr=args.base_lr, weight_decay=1e-5)
        self.optimizer.load_state_dict(op_dict)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr

        if self.epoch == 0:
            self.center = center_avg
            self.c_t = c_t_avg
            self.softmatch.mean = mean_avg
            self.softmatch.var = var_avg

        self.epoch = epoch

        if self.flag:
            self.ema_model.load_state_dict(net.state_dict())
            self.flag = False

        epoch_loss = []

        print('begin unsup_training')

        for epoch in range(args.local_ep):
            batch_loss = []

            for i_batch, sampled_batch in enumerate(self.ldr_train):
                image_batch, label_batch = sampled_batch['img'], sampled_batch['mask']
                image_batch, label_batch = image_batch.to(device), label_batch.to(device)

                inputs = image_batch
                noise = torch.clamp(torch.randn_like(image_batch) * 0.05, -0.0001, 0.0001)
                ema_inputs = image_batch + noise

                features, outputs = net(inputs)
                with torch.no_grad():
                    ema_features, ema_outputs = self.ema_model(ema_inputs)

                feature = features
                pseudo_label_batch = F.softmax(ema_outputs, dim=1)
                pseudo_label_batch = torch.argmax(pseudo_label_batch, dim=1)
                pseudo_label_batch = F.one_hot(pseudo_label_batch, num_classes=3).transpose(1, 4)
                label = pseudo_label_batch
                prediction = F.softmax(ema_outputs, dim=1)

                if self.init_center is True:
                    feature_list = feature
                    label_list = label
                    prediction_list = prediction
                else:
                    feature_list = torch.cat([feature, self.feature_list], dim=0)
                    label_list = torch.cat([label, self.label_list], dim=0)
                    prediction_list = torch.cat([prediction, self.prediction_list], dim=0)
                    length_max = self.length
                    if len(feature_list) > length_max:
                        feature_list = feature_list[0:length_max, :]
                        label_list = label_list[0:length_max, :]
                        prediction_list = prediction_list[0:length_max, :]
                self.init_center = False
                self.feature_list = feature_list.clone().detach()
                self.label_list = label_list.clone().detach()
                self.prediction_list = prediction_list.clone().detach()

                self.softmatch.update(self.prediction_list)
                mask_batch = self.softmatch.masking(prediction).to(device)
                mask_list = self.softmatch.masking(prediction_list).to(device)

                consistency_weight = get_current_consistency_weight(self.epoch)
                consistency_loss = torch.mean(mask_batch * losses.softmax_mse_loss(outputs, ema_outputs))

                center_batch, _ = center_loss_unlabeled.init_center_unlabeled(feature_list, label_list, mask_list)
                center_batch, _ = center_loss_unlabeled.reweight_center_unlabeled(feature_list, label_list, center_batch, mask_list)
                co_center_loss = co_center(center_batch, center_avg)

                loss = consistency_weight * (co_center_loss + consistency_loss)

                pgc_loss = self.pgcloss.forward(features, ema_features, label, mask_batch)

                if self.epoch > 600:  #PEAD_600
                    loss = loss + 0.01 * consistency_weight * pgc_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                update_ema_variables(net, self.ema_model, args.ema_decay, self.iter_num)
                batch_loss.append(loss.item())

                self.iter_num = self.iter_num + 1

            with torch.no_grad():
                if self.epoch % 20 == 0:
                    mask = self.softmatch.masking(self.prediction_list).to(device)
                    self.center, self.c_t = center_loss_unlabeled.init_center_unlabeled(self.feature_list, self.label_list, mask)
                    self.center, self.c_t = center_loss_unlabeled.reweight_center_unlabeled(self.feature_list, self.label_list, self.center, mask)

            with torch.no_grad():
                self.center = step_center(self.center, self.c_t)

            with torch.no_grad():
                self.dist = cal_dist(self.center, center_avg)

            self.epoch = self.epoch + 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            print("unsup:", epoch_loss)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), copy.deepcopy(self.optimizer.state_dict())
