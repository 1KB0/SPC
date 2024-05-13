from torch.utils.data import DataLoader, Dataset
import copy
import torch
import torch.optim
import torch.nn.functional as F
from options import args_parser
from networks.models import DenseNet121
from utils import losses, ramps
from loss.centerloss import CenterLoss, step_center, co_center
from loss.softmatch import SoftMatch
from loss.pgcloss import PseudoGroupContrast
from FedAvg import cal_dist

args = args_parser()
center_loss_unlabeled = CenterLoss()


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, 30)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


class UnsupervisedLocalUpdate(object):
    def __init__(self, args, dataset=None, length=None):
        self.ldr_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        self.length = length

        net = DenseNet121(out_size=3, mode=args.label_uncertainty, drop_rate=args.drop_rate)

        if len(args.gpu.split(',')) > 1:
            net = torch.nn.DataParallel(net, device_ids=[0, 1])

        self.ema_model = net.cuda()
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
        self.optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=5e-4)
        self.optimizer.load_state_dict(op_dict)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr

        if self.epoch == 0:
            self.center = center_avg
            self.c_t = c_t_avg
            self.softmatch.mean = mean_avg
            self.softmatch.var = var_avg
            print("init mean:", mean_avg)
            print("init var:", var_avg)

        self.epoch = epoch

        if self.flag:
            self.ema_model.load_state_dict(net.state_dict())
            self.flag = False
            print('done')

        epoch_loss = []
        print('begin unsup_training')

        for epoch in range(args.local_ep):
            batch_loss = []
            iter_max = len(self.ldr_train)
            print(iter_max)

            for i, (_, _, (image_batch, ema_image_batch), label_batch) in enumerate(self.ldr_train):
                image_batch, ema_image_batch, label_batch = image_batch.cuda(), ema_image_batch.cuda(), label_batch.cuda()
                inputs = image_batch
                ema_inputs = ema_image_batch
                _, projections, features, outputs = net(inputs)
                with torch.no_grad():
                    _, ema_projections, _, ema_outputs = self.ema_model(ema_inputs)

                feature = features
                pseudo_label_batch = F.softmax(ema_outputs, dim=1)
                pseudo_label_batch = torch.argmax(pseudo_label_batch, dim=1)
                pseudo_label_batch = F.one_hot(pseudo_label_batch, num_classes=3)
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
                mask_batch = self.softmatch.masking(prediction).cuda()
                mask_list = self.softmatch.masking(prediction_list).cuda()

                consistency_weight = get_current_consistency_weight(self.epoch)
                consistency_loss = torch.sum(mask_batch * losses.softmax_mse_loss(outputs, ema_outputs)) / args.batch_size

                loss = 15 * consistency_weight * consistency_loss

                center_batch, _ = center_loss_unlabeled.init_center_unlabeled(feature_list, label_list, mask_list)
                center_batch, _ = center_loss_unlabeled.reweight_center_unlabeled(feature_list, label_list, center_batch, mask_list)
                co_center_loss = co_center(center_batch, center_avg)

                pgc_loss = self.pgcloss.forward(projections, ema_projections, label, mask_batch)

                if self.epoch > 20:
                    loss = loss + 15 * consistency_weight * co_center_loss + 0.01 * consistency_weight * pgc_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                update_ema_variables(net, self.ema_model, args.ema_decay, self.iter_num)
                batch_loss.append(loss.item())

                self.iter_num = self.iter_num + 1

            #print("current mean:", self.softmatch.mean)
            #print("current var:", self.softmatch.var)

            with torch.no_grad():
                if self.epoch % 5 == 0:
                    mask = self.softmatch.masking(self.prediction_list).cuda()
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
