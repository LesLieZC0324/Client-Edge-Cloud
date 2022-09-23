"""
1. Edge Server initialization
2. Edge Server receives updates from the client
3. Edge Server sends the aggregated information back to clients
4. Edge Server sends the updates to the Cloud Server
5. Edge Server receives the aggregated information from the Cloud Server
"""

import torch
import copy
from utils import average_weights


class Edge(object):
    def __init__(self, id, client_ids, shared_layers, global_model, args):
        self.args = args
        self.id = id
        self.client_ids = client_ids
        self.receive_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.all_trainsample_num = 0
        self.shared_state_dict = shared_layers.state_dict()
        self.clock = []
        self.mask = {}  # 模型参数稀疏化掩码矩阵
        # TODO: adaptive prop
        self.prop = args.prop  # 稀疏化程度
        self.global_model = global_model

    def refresh_edge(self):
        self.receive_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def client_register(self, client):
        self.id_registration.append(client.id)
        self.sample_registration[client.id] = len(client.train_loader.dataset)
        return None

    # TODO: model compressed top-k and quantify
    def randomk(self):
        # diff = dict()
        # for name, param in self.shared_state_dict:
        #     p = torch.ones_like(param) * self.prop
        #     if torch.is_floating_point(param):
        #         self.mask[name] = torch.bernoulli(p)
        #     else:
        #         self.mask[name] = torch.bernoulli(p).long()
        #     diff[name] = (param - self.global_model.state_dict()[name]) * self.mask[name]
        # return diff
        for name, param in self.shared_state_dict.items():
            p = torch.ones_like(param) * self.prop
            self.mask[name] = torch.bernoulli(p)
            self.shared_state_dict[name] = param * self.mask[name]
        return None

    def topk(self):
        if self.args.model_set == 2:
            # for name, param in self.shared_state_dict.items():
            #     # print(name, param.numel())
            #     temp = param.reshape(1, -1)
            #     select_size = int(temp.numel() * self.prop)
            #     select_size = select_size if select_size > 0 else 1
            #     values, indices = torch.topk(torch.abs(temp), k=select_size)
            #     k = float(values[:, -1])
            #     self.mask[name] = torch.ge(torch.abs(param), k)
            #     self.shared_state_dict[name] = param * self.mask[name]
            #     # print(self.shared_state_dict[name])
            param_num = 0.0
            sparse_num = 0.0
            for name, param in self.shared_state_dict.items():
                param_num += float(torch.count_nonzero(param))
                if param.numel() > 100:
                    select_size = int(param.numel() * self.prop)
                    select_size = select_size if select_size > 0 else 1
                    values, indices = torch.topk(torch.abs(param.reshape(1, -1)), k=select_size)
                    k = float(values[:, -1])
                    self.mask[name] = torch.ge(torch.abs(param), k)
                    self.shared_state_dict[name] = param * self.mask[name]
                    sparse_num += float(torch.count_nonzero(self.shared_state_dict[name]))  # non_zero after sparasification
                else:
                    sparse_num += float(torch.count_nonzero(param))
                # print(self.shared_state_dict[name])
            compress_ratio = param_num / sparse_num
            return compress_ratio
        else:
            pass

    # TODO: error
    def quantify(self):
        for name, param in self.shared_state_dict.items():
            self.shared_state_dict[name] = torch.quantize_per_tensor(param, scale=0.005, zero_point=0, dtype=torch.qint8)
            # print(self.shared_state_dict[name])
        return None

    def dequantify(self, quantized_model):
        dequantized_model = torch.dequantize(quantized_model)
        return dequantized_model

    def receive_from_client(self, client_id, client_shared_state_dict):
        if self.args.compressed == 3:
            for name, param in client_shared_state_dict.items():
                client_shared_state_dict[name] = self.dequantify(param)
        self.receive_buffer[client_id] = client_shared_state_dict
        return None

    def aggregate(self, args):
        receive_dict = [client_dict for client_dict in self.receive_buffer.values()]
        sample_num = [client_num for client_num in self.sample_registration.values()]
        self.shared_state_dict = average_weights(w=receive_dict, s_num=sample_num)
        # if self.args.compressed == 0:
        #     self.shared_state_dict = average_weights(w=receive_dict, s_num=sample_num)
        # else:
        #     w_avg = average_weights(w=receive_dict, s_num=sample_num)
        #     for name, param in self.global_model.state_dict().items():
        #         self.global_model.state_dict()[name].add_(w_avg[name])
        #     self.shared_state_dict = dict(self.global_model.state_dict())

    def send_to_client(self, client):
        client.receive_from_edge(copy.deepcopy(self.shared_state_dict))
        return None

    def send_to_cloud(self, cloud):
        if self.args.compressed == 0:
            cloud.receive_from_edge(edge_id=self.id, edge_shared_state_dict=copy.deepcopy(self.shared_state_dict))
            return None
        elif self.args.compressed == 3 and self.args.model_set == 2:
            self.quantify()
            cloud.receive_from_edge(edge_id=self.id, edge_shared_state_dict=self.shared_state_dict)
            return None
        elif self.args.compressed == 1:
            self.randomk()
            cloud.receive_from_edge(edge_id=self.id, edge_shared_state_dict=copy.deepcopy(self.shared_state_dict))
            return None
        elif self.args.compressed == 2:
            compress_ratio = self.topk()
            cloud.receive_from_edge(edge_id=self.id, edge_shared_state_dict=copy.deepcopy(self.shared_state_dict))
            return compress_ratio
        else:
            raise ValueError("`{}` is not right".format(self.args.compressed))

    def receive_from_cloud(self, shared_state_dict):
        self.shared_state_dict = shared_state_dict
        return None
