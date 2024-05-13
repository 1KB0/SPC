import torch
import numpy as np
import torch.nn.functional as F

CLASS_NUM = 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]  # 使用 cuda:0 和 cuda:1

# 样本可靠性权重，每个epoch操作
class SoftMatch:
    def __init__(self, n_sigma=2.0, momentum=0.999):
        super(SoftMatch, self).__init__()

        # 为每个类计算均值和方差（默认均值为1/C，方差为1）
        self.mean = (1.0 / CLASS_NUM) * torch.ones((CLASS_NUM, 144, 144, 144)).to(device)
        self.var = torch.ones((CLASS_NUM, 144, 144, 144)).to(device)
        # 指数移动更新
        self.momentum = momentum
        self.n_sigma = n_sigma

    @torch.no_grad()
    def update(self, prediction_list):
        prediction, index = torch.max(prediction_list, dim=1)
        # 为每个类更新均值和方差（默认均值为0，方差为1）
        pred_mean = (1.0 / CLASS_NUM) * torch.ones((CLASS_NUM, 144, 144, 144)).to(device)
        pred_var = torch.ones((CLASS_NUM, 144, 144, 144)).to(device)
        for i in range(CLASS_NUM):
            pred = prediction[index == i]
            if pred.shape[0] > 1:
                pred_mean[i] = torch.mean(pred)  # mu
                pred_var[i] = torch.var(pred, unbiased=True)  # sigma**2
        # 指数移动更新
        self.mean = self.momentum * self.mean + (1 - self.momentum) * pred_mean
        self.var = self.momentum * self.var + (1 - self.momentum) * pred_var

    @torch.no_grad()
    def masking(self, prediction_list):  # 计算权重
        prediction, index = torch.max(prediction_list, dim=1)
        # 为每个样本计算权重
        prediction = prediction.to(device)
        index = index.to(device)
        total = prediction.shape[0]
        mask = torch.zeros((total, CLASS_NUM, 144, 144, 144)).to(device)
        # mask = torch.zeros((total, 144, 144, 144)).to(device)
        # mask = torch.unsqueeze(mask, dim=1)
        for i in range(total):
            prediction_i = prediction[i]
            index_i = index[i]
            for j in range(CLASS_NUM):
                cls = index_i == j
                cls = cls.float()

                pred = prediction_i * cls  # 拿到这一批次中属于这一类的预测
                mask[i][j] = 1.0 * torch.exp(-(torch.clamp(pred - self.mean[j], max=0.0) ** 2) / (self.n_sigma * self.var[j]))
        # 返回权重
        return mask


