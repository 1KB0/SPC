import copy
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


def FedAvg(w, dict_length):
    dict_list = dict_length[0:len(w)]
    client_weight = [(dict_list[i] / sum(dict_list)) for i in range(len(w))]
    print('client-weight:', client_weight)

    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        temp = torch.zeros_like(w_avg[key], dtype=torch.float32).cuda()
        for i in range(len(w)):
            temp += client_weight[i] * w[i][key]
        w_avg[key].data.copy_(temp)

    return w_avg


def cal_dist(center, center_avg):
    source = F.softmax(center, dim=1)
    target = F.softmax(center_avg, dim=1)
    dist = (F.kl_div(source.log(), target, reduction='batchmean') + F.kl_div(target.log(), source, reduction='batchmean')) / 2.0
    dist = torch.exp(-1.0 * dist)
    print('center-dist:', dist.item())
    return dist.item()


def aggregation(w, dict_length, trainer_locals):
    dict_list = dict_length[0:len(w)]
    client_dist = [(dict_list[i] / sum(dict_list)) * trainer_locals[i].dist for i in range(len(w))]
    client_weight = [(client_dist[i] / sum(client_dist)) for i in range(len(w))]
    print('client-weight:', client_weight)

    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        temp = torch.zeros_like(w_avg[key], dtype=torch.float32).cuda()
        for i in range(len(w)):
            temp += client_weight[i] * w[i][key]
        w_avg[key].data.copy_(temp)

    return w_avg

