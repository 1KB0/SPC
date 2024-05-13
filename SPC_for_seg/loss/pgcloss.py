import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.centerloss import CenterLoss

center_loss_unlabeled = CenterLoss()

# 伪标组对比
class PseudoGroupContrast(nn.Module):
    def __init__(self):
        super(PseudoGroupContrast, self).__init__()
        # 特征大小
        self.projector_dim = 192
        # 分类数量
        self.class_num = 3
        # 列表长度
        self.queue_size = 50
        # 申请长期缓存，随机初始值正则化
        self.register_buffer("queue_list", torch.randn(self.queue_size*self.class_num, self.projector_dim))
        self.queue_list = F.normalize(self.queue_list, dim=1).cuda()
        # 温度参数
        self.temperature = 0.5
        # self.projectors = self.Projector(192)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, ema_feature, label):
        # 当前类列表
        temp_list = self.queue_list[label*self.queue_size:(label+1)*self.queue_size, :]
        # 把当前特征拼接到队列中，再从头取相同大小的队列
        temp_list = torch.cat([ema_feature, temp_list], dim=0)
        temp_list = temp_list[0:self.queue_size, :]
        # 替换原队列
        self.queue_list[label*self.queue_size:(label+1)*self.queue_size, :] = temp_list

    # 计算pgc-loss
    def forward(self, feature, ema_feature, pseudo_label, mask=None):
        # 正则化
        # feature = self.projectors(feature)
        # ema_feature = self.projectors(ema_feature)
        # one-hot伪标转换为数字型
        # label = torch.argmax(pseudo_label, dim=1)
        # 输入图像数量
        batch_size = feature.size(0)
        # 对比损失
        contrast_loss = 0.0
        # 获取当前列表
        current_queue_list = self.queue_list.clone().detach()

        # 获取所有样本的特征的3个中心点
        total_prototype, _ = center_loss_unlabeled.init_center_unlabeled(feature, pseudo_label, mask)
        total_ema_prototype, _ = center_loss_unlabeled.init_center_unlabeled(ema_feature, pseudo_label, mask)
        total_proto = F.normalize(total_prototype, dim=1)
        total_ema_proto = F.normalize(total_ema_prototype, dim=1)

        # feature和ema_feature对应位置计算点乘
        # 使用einsum代替dot提高计算效率
        l_pos = torch.einsum('nl,nl->n', [total_proto, total_ema_proto])
        l_pos = torch.exp(l_pos/self.temperature)

        # 计算所有样本
        for i in range(batch_size):
            # 当前样本
            current_f = feature[i:i+1]
            current_ema_f = ema_feature[i:i+1]
            # 获取一个样本的特征的3个中心点
            prototype, _ = center_loss_unlabeled.init_center_unlabeled(current_f, pseudo_label, mask)
            ema_prototype, _ = center_loss_unlabeled.init_center_unlabeled(current_ema_f, pseudo_label, mask)
            proto = F.normalize(prototype, dim=1)
            ema_proto = F.normalize(ema_prototype, dim=1)

            for j in range(3):
                current_c = j
                ith_ema = l_pos[i:i+1]

                # 构造正样本和负样本列表
                pos_sample = current_queue_list[current_c*self.queue_size:(current_c+1)*self.queue_size, :]
                neg_sample = torch.cat([current_queue_list[0:current_c*self.queue_size, :],
                                        current_queue_list[(current_c+1)*self.queue_size:, :]], dim=0)

                # 计算正样本
                ith_pos = torch.einsum('nl,nl->n', [proto[j].view(1, -1), pos_sample])
                ith_pos = torch.exp(ith_pos/self.temperature)  # 为式（11）log分子上的exp（..）
                pos = torch.sum(ith_pos)  # 分母Pos的后半段
                # 计算负样本
                ith_neg = torch.einsum('nl,nl->n', [proto[j].view(1, -1), neg_sample])
                ith_neg = torch.exp(ith_neg/self.temperature)
                neg = torch.sum(ith_neg)  # 分母Neg

                # 计算当前样本的对比损失
                # 正样本列表D+1
                contrast_D = ith_pos/(ith_ema + pos + neg)
                contrast_1 = ith_ema/(ith_ema + pos + neg)

                contrast = torch.cat([contrast_D, contrast_1], dim=0)
                contrast = -torch.log(contrast + 1e-6)
                contrast = torch.sum(contrast)/(self.queue_size + 1)

                # 损失累加
                contrast_loss = contrast_loss + contrast

                # 更新队列
                self._dequeue_and_enqueue(ema_proto, current_c)
                j += 1

        # 返回pgc-loss
        # xx = torch.sum(contrast_loss)
        return contrast_loss/batch_size

