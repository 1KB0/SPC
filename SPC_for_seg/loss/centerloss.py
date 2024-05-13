import torch
import numpy as np
import torch.nn.functional as F

CLASS_NUM = 3


# center-loss，每个batch操作
class CenterLoss:
    def __init__(self):
        super(CenterLoss, self).__init__()

    # 计算分类点
    def init_center(self, feature, label):
        global sum_feature, sum_index
        b, c, h, w, d = feature.shape
        # 计算每类样本到各自分类点的距离
        for i in range(CLASS_NUM):
            if i == 0:
                index = torch.eq(label, i).float()
                index = F.adaptive_avg_pool3d(index, (h, w, d)) > 0.5
                index = index.repeat(1, c, 1, 1, 1)
                feature = feature.permute(0, 2, 3, 4, 1)
                index = index.permute(0, 2, 3, 4, 1)

                feature_i = torch.masked_select(feature, index).view(-1, c).mean(0).unsqueeze(0)
                center_c = feature_i
            else:
                index = torch.eq(label, i).float()
                index = F.adaptive_avg_pool3d(index, (h, w, d)) > 0.5
                index = index.repeat(1, c, 1, 1, 1)
                index = index.permute(0, 2, 3, 4, 1)
                feature_i = torch.masked_select(feature, index).view(-1, c).mean(0).unsqueeze(0)
                center_c = torch.cat([center_c, feature_i], dim=0)
        # 计算中心点
        center = center_c
        c_t = torch.sum(center, dim=0) / CLASS_NUM
        return center, c_t  # 所有类别的初始中心以及所有类别的平均中心。

    # 根据中心点距离重新计算分类点
    def reweight_center(self, feature, label, center):
        # one-hot型标签转换为数字型标签
        label = torch.argmax(label, dim=1).unsqueeze(1)
        # tensor([7, 128])
        global sum_feature, sum_index
        # 计算每类样本到各自分类点的距离
        for i in range(CLASS_NUM):
            index = torch.eq(label, i).int()
            feature_i = feature.mul(index)
            center_i = center[i:i+1]
            dist = F.cosine_similarity(feature, center_i, dim=1).unsqueeze(1)
            dist = 0.5 + 0.5 * dist  # 归一化
            weight_i = dist.mul(index)
            if i == 0:
                sum_feature = torch.sum(weight_i * feature_i, dim=0).view(1, -1)
                sum_index = torch.sum(weight_i).view(1, 1)
            else:
                sum_feature = torch.cat([sum_feature, torch.sum(weight_i * feature_i, dim=0).view(1, -1)], dim=0)
                sum_index = torch.cat([sum_index, torch.sum(weight_i).view(1, 1)], dim=0)
        # 计算中心点
        reweight_center = sum_feature / (sum_index + 1)
        reweight_c_t = torch.sum(center, dim=0) / CLASS_NUM
        return reweight_center, reweight_c_t

    # 计算分类点
    def init_center_unlabeled(self, feature, label, mask):
        # one-hot型标签转换为数字型标签
        label = torch.argmax(label, dim=1).unsqueeze(1)
        global sum_feature, sum_index
        b, c, h, w, d = feature.shape
        # 计算每类样本到各自分类点的距离
        for i in range(CLASS_NUM):
            if i == 0:
                index = torch.eq(label, i).float()
                weight_i = mask.mul(index)  # (2,3,144,144,144)
                index = F.adaptive_avg_pool3d(index, (h, w, d)) > 0.5
                weight_i = F.avg_pool3d(weight_i, kernel_size=(8, 8, 8))  # (2,3,18,18,18)
                index = index.repeat(1, c, 1, 1, 1)
                weight_i = weight_i.repeat(1, c//3, 1, 1, 1)  # （2,192,18,18,18）
                feature = feature.permute(0, 2, 3, 4, 1)
                index = index.permute(0, 2, 3, 4, 1)
                weight_i = weight_i.permute(0, 2, 3, 4, 1)  # (2,18,18,18,192)

                feature_i = feature.mul(weight_i)
                feature_i = torch.masked_select(feature_i, index).view(-1, c).mean(0).unsqueeze(0)
                center_c = feature_i
            else:
                index = torch.eq(label, i).float()
                weight_i = mask.mul(index)  # (2,3,144,144,144)
                index = F.adaptive_avg_pool3d(index, (h, w, d)) > 0.5
                weight_i = F.avg_pool3d(weight_i, kernel_size=(8, 8, 8))  # (2,3,18,18,18)
                index = index.repeat(1, c, 1, 1, 1)
                weight_i = weight_i.repeat(1, c//3, 1, 1, 1)  # （2,192,18,18,18）
                index = index.permute(0, 2, 3, 4, 1)
                weight_i = weight_i.permute(0, 2, 3, 4, 1)  # (2,18,18,18,192)
                feature_i = feature.mul(weight_i)
                feature_i = torch.masked_select(feature_i, index).view(-1, c).mean(0).unsqueeze(0)
                center_c = torch.cat([center_c, feature_i], dim=0)
        # 计算中心点
        center = center_c
        # has_nan = torch.isnan(center).any()
        # ss = torch.sum(center, dim=0)
        c_t = torch.sum(center, dim=0) / CLASS_NUM
        return center, c_t  # 所有类别的初始中心以及所有类别的平均中心。

    # 根据中心点距离重新计算分类点
    def reweight_center_unlabeled(self, feature, label, center, mask):
        # one-hot型标签转换为数字型标签
        label = torch.argmax(label, dim=1).unsqueeze(1)
        global sum_feature, sum_index
        b, c, h, w, d = feature.shape
        # 计算每类样本到各自分类点的距离
        for i in range(CLASS_NUM):
            if i == 0:
                index = torch.eq(label, i).float()
                weight_i = mask.mul(index)  # (2,3,144,144,144)
                center_i = center[i:i+1]  # (1,192)
                index = F.adaptive_avg_pool3d(index, (h, w, d)) > 0.5
                weight_i = F.avg_pool3d(weight_i, kernel_size=(8, 8, 8))  # (2,3,18,18,18)
                index = index.repeat(1, c, 1, 1, 1)
                weight_i = weight_i.repeat(1, c//3, 1, 1, 1)  # （2,192,18,18,18）
                feature = feature.permute(0, 2, 3, 4, 1)
                index = index.permute(0, 2, 3, 4, 1)
                weight_i = weight_i.permute(0, 2, 3, 4, 1)  # (2,18,18,18,192)

                feature_i = feature.mul(weight_i)
                feature_i = torch.masked_select(feature_i, index).view(-1, c).mean(0).unsqueeze(0)

                feature_dist = torch.masked_select(feature, index).view(-1, c).mean(0).unsqueeze(0)
                dist = F.cosine_similarity(feature_dist, center_i, dim=1).unsqueeze(1)
                dist = 0.5 + 0.5 * dist  # 归一化

                feature_i = feature_i * dist

                center_c = feature_i
            else:
                index = torch.eq(label, i).float()
                weight_i = mask.mul(index)  # (2,3,144,144,144)
                center_i = center[i:i + 1]  # (1,192)
                index = F.adaptive_avg_pool3d(index, (h, w, d)) > 0.5
                weight_i = F.avg_pool3d(weight_i, kernel_size=(8, 8, 8))  # (2,3,18,18,18)
                index = index.repeat(1, c, 1, 1, 1)
                weight_i = weight_i.repeat(1, c//3, 1, 1, 1)  # （2,192,18,18,18）
                index = index.permute(0, 2, 3, 4, 1)
                weight_i = weight_i.permute(0, 2, 3, 4, 1)  # (2,18,18,18,192)
                feature_i = feature.mul(weight_i)
                feature_i = torch.masked_select(feature_i, index).view(-1, c).mean(0).unsqueeze(0)

                feature_dist = torch.masked_select(feature, index).view(-1, c).mean(0).unsqueeze(0)
                dist = F.cosine_similarity(feature_dist, center_i, dim=1).unsqueeze(1)
                dist = 0.5 + 0.5 * dist  # 归一化

                feature_i = feature_i * dist

                center_c = torch.cat([center_c, feature_i], dim=0)
        # 计算中心点
        reweight_center = center_c
        reweight_c_t = torch.sum(center, dim=0) / CLASS_NUM
        return reweight_center, reweight_c_t


# 移动分类点，每个epoch操作
def step_center(center, c_t):
    alpha = 0.002
    step_center = center + alpha * ((center-c_t) / torch.norm(center-c_t, p=2, dim=1).view((CLASS_NUM, -1)))
    # 返回分类点（移动）
    return step_center


# 每个batch操作
def co_center(center_unlabeled, center_labeled):
    # 正则化
    center_unlabeled = F.normalize(center_unlabeled, dim=1)
    center_labeled = F.normalize(center_labeled, dim=1)
    # 计算mse损失
    center_consistency_loss = F.mse_loss(center_unlabeled, center_labeled, reduction='sum') / CLASS_NUM
    # 返回分类点一致性loss
    return center_consistency_loss

