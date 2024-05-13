import torch
import numpy as np
import torch.nn.functional as F

CLASS_NUM = 3


class CenterLoss:
    def __init__(self):
        super(CenterLoss, self).__init__()

    def init_center(self, feature, label):
        label = torch.argmax(label, dim=1).unsqueeze(1)
        global sum_feature, sum_index

        for i in range(CLASS_NUM):
            index = torch.eq(label, i).int()
            feature_i = feature.mul(index)
            if i == 0:
                sum_feature = torch.sum(feature_i, dim=0).view(1, -1)
                sum_index = torch.sum(index).view(1, 1)
            else:
                sum_feature = torch.cat([sum_feature, torch.sum(feature_i, dim=0).view(1, -1)], dim=0)
                sum_index = torch.cat([sum_index, torch.sum(index).view(1, 1)], dim=0)

        center = sum_feature / (sum_index + 1)
        c_t = torch.sum(center, dim=0) / CLASS_NUM
        return center, c_t

    def reweight_center(self, feature, label, center):
        label = torch.argmax(label, dim=1).unsqueeze(1)
        global sum_feature, sum_index

        for i in range(CLASS_NUM):
            index = torch.eq(label, i).int()
            feature_i = feature.mul(index)
            center_i = center[i:i+1]
            dist = F.cosine_similarity(feature, center_i, dim=1).unsqueeze(1)
            dist = 0.5 + 0.5 * dist
            weight_i = dist.mul(index)
            if i == 0:
                sum_feature = torch.sum(weight_i * feature_i, dim=0).view(1, -1)
                sum_index = torch.sum(weight_i).view(1, 1)
            else:
                sum_feature = torch.cat([sum_feature, torch.sum(weight_i * feature_i, dim=0).view(1, -1)], dim=0)
                sum_index = torch.cat([sum_index, torch.sum(weight_i).view(1, 1)], dim=0)

        reweight_center = sum_feature / (sum_index + 1)
        reweight_c_t = torch.sum(center, dim=0) / CLASS_NUM
        return reweight_center, reweight_c_t

    def init_center_unlabeled(self, feature, label, mask):
        label = torch.argmax(label, dim=1).unsqueeze(1)
        global sum_feature, sum_index

        for i in range(CLASS_NUM):
            index = torch.eq(label, i).int()
            feature_i = feature.mul(index)
            weight_i = mask.mul(index)
            if i == 0:
                sum_feature = torch.sum(weight_i * feature_i, dim=0).view(1, -1)
                sum_index = torch.sum(weight_i).view(1, 1)
            else:
                sum_feature = torch.cat([sum_feature, torch.sum(weight_i * feature_i, dim=0).view(1, -1)], dim=0)
                sum_index = torch.cat([sum_index, torch.sum(weight_i).view(1, 1)], dim=0)

        center = sum_feature / (sum_index + 1)
        c_t = torch.sum(center, dim=0) / CLASS_NUM
        return center, c_t

    def reweight_center_unlabeled(self, feature, label, center, mask):
        label = torch.argmax(label, dim=1).unsqueeze(1)
        global sum_feature, sum_index

        for i in range(CLASS_NUM):
            index = torch.eq(label, i).int()
            feature_i = feature.mul(index)
            center_i = center[i:i+1]
            dist = F.cosine_similarity(feature, center_i, dim=1).unsqueeze(1)
            dist = 0.5 + 0.5 * dist
            dist = mask * dist
            weight_i = dist.mul(index)
            if i == 0:
                sum_feature = torch.sum(weight_i * feature_i, dim=0).view(1, -1)
                sum_index = torch.sum(weight_i).view(1, 1)
            else:
                sum_feature = torch.cat([sum_feature, torch.sum(weight_i * feature_i, dim=0).view(1, -1)], dim=0)
                sum_index = torch.cat([sum_index, torch.sum(weight_i).view(1, 1)], dim=0)

        reweight_center = sum_feature / (sum_index + 1)
        reweight_c_t = torch.sum(reweight_center, dim=0) / CLASS_NUM
        return reweight_center, reweight_c_t


def step_center(center, c_t):
    alpha = 0.002
    step_center = center + alpha * ((center-c_t) / torch.norm(center-c_t, p=2, dim=1).view((CLASS_NUM, -1)))

    return step_center


def co_center(center_unlabeled, center_labeled):
    center_unlabeled = F.normalize(center_unlabeled, dim=1)
    center_labeled = F.normalize(center_labeled, dim=1)

    center_consistency_loss = F.mse_loss(center_unlabeled, center_labeled, reduction='sum') / CLASS_NUM

    return center_consistency_loss

