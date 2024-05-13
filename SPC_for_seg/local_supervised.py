from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim
import torch.nn.functional as F
from options import args_parser
import copy
from utils import losses, ramps
from loss.centerloss import CenterLoss, step_center, co_center
from loss.softmatch import SoftMatch
from FedAvg import cal_dist
from monai.losses import DiceCELoss

args = args_parser()
center_loss_labeled = CenterLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]
loss_function = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)

# 一致性损失系数
def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, 40)


# 标记客户端本地训练
class SupervisedLocalUpdate(object):
    def __init__(self, args, dataset, length):
        self.ldr_train = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
        self.length = length
        self.epoch = 0
        self.iter_num = 0
        self.base_lr = args.base_lr

        self.init_center = True
        self.softmatch = SoftMatch()

    def train(self, args, net, op_dict, epoch, center_avg, c_t_avg):
        net.train()
        self.optimizer = torch.optim.AdamW(net.parameters(), lr=args.base_lr, weight_decay=1e-5)
        self.optimizer.load_state_dict(op_dict)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr

        if self.epoch == 0:
            self.center = center_avg
            self.c_t = c_t_avg

        self.epoch = epoch

        # train and update
        epoch_loss = []
        print('begin sup_training')

        for epoch in range(args.local_ep):
            batch_loss = []
            iter_max = len(self.ldr_train)
            # print(iter_max)

            # 计算loss
            for i_batch, sampled_batch in enumerate(self.ldr_train):
                image_batch, label_batch = sampled_batch['img'], sampled_batch['mask']
                image_batch, label_batch = image_batch.to(device), label_batch.to(device)
                inputs = image_batch
                features, outputs = net(inputs)

                feature = features
                label = label_batch
                prediction = F.softmax(outputs, dim=1)

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

                loss_classification = loss_function(outputs, label)

                loss = loss_classification

                if self.epoch > 100:  #PEAD_100
                    consistency_weight = get_current_consistency_weight(self.epoch)
                    center_batch, _ = center_loss_labeled.init_center(feature_list, label_list)
                    co_center_loss = co_center(center_batch, center_avg)
                    # print("co-center-loss:", co_center_loss)
                    loss = loss + consistency_weight * co_center_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())

                self.iter_num = self.iter_num + 1

            with torch.no_grad():
                if self.epoch % 20 == 0:
                    self.center, self.c_t = center_loss_labeled.init_center(self.feature_list, self.label_list)

            with torch.no_grad():
                self.center = step_center(self.center, self.c_t)

            with torch.no_grad():
                self.dist = cal_dist(self.center, center_avg)

            self.epoch = self.epoch + 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            print("sup:", epoch_loss)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), copy.deepcopy(self.optimizer.state_dict())
