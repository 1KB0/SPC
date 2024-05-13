import torch
import numpy as np
import torch.nn.functional as F

CLASS_NUM = 3


class SoftMatch:
    def __init__(self, n_sigma=2.0, momentum=0.999):
        super(SoftMatch, self).__init__()

        self.mean = torch.ones((CLASS_NUM)) / CLASS_NUM
        self.var = torch.ones((CLASS_NUM))

        self.momentum = momentum
        self.n_sigma = n_sigma

    @torch.no_grad()
    def update(self, prediction_list):
        prediction, index = torch.max(prediction_list, dim=1)

        pred_mean = torch.ones_like(self.mean) / CLASS_NUM
        pred_var = torch.ones_like(self.var)
        for i in range(CLASS_NUM):
            pred = prediction[index == i]
            if pred.shape[0] > 1:
                pred_mean[i] = torch.mean(pred)  # mu
                pred_var[i] = torch.var(pred, unbiased=True)  # sigma**2

        self.mean = self.momentum * self.mean + (1 - self.momentum) * pred_mean
        self.var = self.momentum * self.var + (1 - self.momentum) * pred_var

    @torch.no_grad()
    def masking(self, prediction_list):
        prediction, index = torch.max(prediction_list, dim=1)

        total = prediction.shape[0]
        mask = torch.zeros((total)).unsqueeze(1)
        for j in range(total):
            pred = prediction[j]
            idx = index[j]
            mask[j] = 1.0 * torch.exp(-(torch.clamp(pred - self.mean[idx], max=0.0) ** 2) / (self.n_sigma * self.var[idx]))

        return mask

