"""
1. Client initialization, dataloaders, model(include optimizer)
2. Client model update
3. Client send updates to edge server
4. Client receives updates from edge server
5. Client modify local model based on the feedback from the edge server
"""

import torch
from torch import nn
from torch.autograd import Variable
from models.model_initialization import initialize_model
import copy
from torch.utils.data import DataLoader, Dataset
import time


class Client(object):
    def __init__(self, id, train_loader, test_loader, args, device, global_model):
        self.args = args
        self.id = id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = initialize_model(args, device)
        # for name, param in self.model.shared_layers.state_dict().items():
        #     print(name, param.size(), type(param))
        self.receive_buffer = {}
        self.batch_size = args.batch_size
        self.epoch = 0
        self.clock = []
        self.mask = {}  # 模型参数稀疏化掩码矩阵
        # TODO: adaptive prop
        self.prop = args.prop  # 稀疏化程度
        self.global_model = global_model

    def local_update(self, local_epoch, device):
        start_time = time.time()
        epoch_count = 0
        loss = 0.0
        end = False
        lr_temp = 0
        # the upper bound of the local_update is 1000 (never reached)
        for i in range(1000):
            for data in self.train_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                loss += self.model.optimize_model(input_batch=inputs, label_batch=labels)
                epoch_count += 1
                if epoch_count >= local_epoch:
                    end = True
                    self.epoch += 1
                    self.model.scheduler.step()
                    train_time = time.time() - start_time
                    break
            if end:
                break
            self.epoch += 1
        loss /= local_epoch
        # return loss, train_time, [param['lr'] for param in self.model.optimizer.param_groups]
        return loss, train_time

    def test_model(self, device):
        acc = 0.0
        total = 0.0
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model.test_model(input_batch=inputs)
                _, predict = torch.max(outputs, 1)
                total += labels.size(0)
                acc += (predict == labels).sum().item()
        return acc, total

    # TODO: model compressed random-k, top-k and quantify
    def randomk(self):
        # for name, param in self.model.shared_layers.state_dict().items():
        #     p = torch.ones_like(param) * self.prop
        #     if torch.is_floating_point(param):
        #         self.mask[name] = torch.bernoulli(p)  # bernoulli分布按p概率抽取，随机生成mask
        #     else:
        #         self.mask[name] = torch.bernoulli(p).long()
        #     # 只传输randomk选中的模型参数，其余参数使用上一轮的
        #     self.model.shared_layers.state_dict()[name][:] = param * self.mask[name]
        #     # print(self.model.shared_layers.state_dict()[name], type(self.model.shared_layers.state_dict()[name]))

        diff = dict()
        for name, param in self.model.shared_layers.state_dict().items():
            # print(param.size())
            p = torch.ones_like(param) * self.prop
            if torch.is_floating_point(param):
                self.mask[name] = torch.bernoulli(p)  # bernoulli分布按p概率抽取，随机生成mask
            else:
                self.mask[name] = torch.bernoulli(p).long()
            # 只传输randomk选中的模型参数与全局模型的差值，其余参数使用上一轮的
            diff[name] = (param - self.global_model.state_dict()[name]) * self.mask[name]
        return diff

    # def topk(self):
    #     diff = dict()
    #     for name, param in self.model.shared_layers.state_dict().items():
    #         self.mask[name] = torch.zeros_like(param)
    #         try:
    #             # 四维tensor
    #             num_k = int(max(param[0][0][0].numel() * self.prop, 1))
    #             values, indices = torch.topk(torch.abs(param.data), k=num_k)
    #             for m in range(len(self.mask[name])):
    #                 for n in range(len(self.mask[name][0])):
    #                     for i in range(len(self.mask[name][0][0])):
    #                         for j in range(len(self.mask[name][0][0][0])):
    #                             if j in indices[m][n][i]:
    #                                 self.mask[name][m][n][i][j] = 1
    #         except:
    #             try:
    #                 # 二维tensor
    #                 num_k = int(max(param[0].numel() * self.prop, 1))
    #                 values, indices = torch.topk(torch.abs(param.data), k=num_k)
    #                 for i in range(len(self.mask[name])):
    #                     for j in range(len(self.mask[name][0])):
    #                         if j in indices[i]:
    #                             self.mask[name][i][j] = 1
    #             except:
    #                 # 一维tensor
    #                 num_k = int(max(param.numel() * self.prop, 1))
    #                 values, indices = torch.topk(torch.abs(param.data), k=num_k)
    #                 for i in range(len(self.mask[name])):
    #                     if i in indices:
    #                         self.mask[name][i] = 1
    #         diff[name] = (param - self.global_model.state_dict()[name]) * self.mask[name]
    #         # print(diff[name])
    #     return diff

    def topk(self):
        diff = dict()
        # param_num = 0.0
        # sparse_num = 0.0
        for name, param in self.model.shared_layers.state_dict().items():
            # param_num += param.numel()
            if param.numel() > 100:  # 小参数层不压缩
                temp = (param - self.global_model.state_dict()[name])  # 选择改变量大的参数进行更新
                select_size = int(temp.numel() * self.prop)
                select_size = select_size if select_size > 0 else 1
                values, indices = torch.topk(torch.abs(temp.reshape(1, -1)), k=select_size)
                k = float(values[:, -1])
                self.mask[name] = torch.ge(torch.abs(temp), k)
                diff[name] = temp * self.mask[name]
                # sparse_num += float(torch.count_nonzero(diff[name]))
            else:
                # sparse_num += param.numel()
                diff[name] = param - self.global_model.state_dict()[name]
            # print(diff[name])
        # print(param_num / sparse_num)
        return diff

    def quantify(self):
        diff = dict()
        for name, param in self.model.shared_layers.state_dict().items():
            diff[name] = torch.quantize_per_tensor(param - self.global_model.state_dict()[name],
                                                   scale=0.005, zero_point=0, dtype=torch.qint8)
        return diff

    def send_to_edge(self, edge):
        if self.args.compressed == 0:
            edge.receive_from_client(client_id=self.id,
                                     client_shared_state_dict=copy.deepcopy(self.model.shared_layers.state_dict()))
        else:
            if self.args.compressed == 1:
                diff = self.randomk()
            elif self.args.compressed == 2:
                diff = self.topk()
            else:
                diff = self.quantify()
            edge.receive_from_client(client_id=self.id,
                                     client_shared_state_dict=diff)
        return None

    def receive_from_edge(self, shared_state_dict):
        self.receive_buffer = shared_state_dict
        return None

    def sync_with_edge(self):
        # self.model.shared_layers.load_state_dict(self.receive_buffer)
        self.model.update_model(self.receive_buffer)
        return None
