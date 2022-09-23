import torch
import copy
from torch import nn


def average_weights(w, s_num):
    total_sample_num = sum(s_num)
    temp_sample_num = s_num[0]
    w_avg = copy.deepcopy(w[0])  # copy the first client's weights
    for k in w_avg.keys():  # the nn layer loop
        # for i in range(1, len(w)):  # the client loop
        #     w_avg[k] += (torch.mul(w[i][k], s_num[i] / temp_sample_num))  # 使用权重进行聚合
        # w_avg[k] = torch.mul(w_avg[k], temp_sample_num / total_sample_num)
        # # print(type(w_avg[k]))  # dict(tensor)
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
