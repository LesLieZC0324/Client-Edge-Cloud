"""
1. Server initialization
2. Server reveives updates from the user
3. Server send the aggregated information back to clients
"""

import torch
import copy
from utils import average_weights


class Cloud(object):
    def __init__(self, shared_layers, global_model, args):
        self.args = args
        self.global_model = global_model
        self.receive_buffer = {}
        self.shared_state_dict = {}
        # self.shared_state_dict = shared_layers.state_dict()
        self.id_registration = []
        self.sample_registration = {}
        self.clock = []

    def refresh_cloud(self):
        self.receive_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def edge_registration(self, edge):
        self.id_registration.append(edge.id)
        self.sample_registration[edge.id] = edge.all_trainsample_num
        return None

    def dequantify(self, quantized_model):
        dequantized_model = torch.dequantize(quantized_model)
        return dequantized_model

    def receive_from_edge(self, edge_id, edge_shared_state_dict):
        if self.args.model_set == 2 and self.args.compressed == 3:
            for name, param in edge_shared_state_dict.items():
                edge_shared_state_dict[name] = self.dequantify(param)
        self.receive_buffer[edge_id] = edge_shared_state_dict
        return None

    def aggregate(self, args):
        receive_dict = [edge_dict for edge_dict in self.receive_buffer.values()]
        sample_num = [edge_num for edge_num in self.sample_registration.values()]
        if self.args.compressed == 0:
            self.shared_state_dict = average_weights(w=receive_dict, s_num=sample_num)
        else:
            w_avg = average_weights(w=receive_dict, s_num=sample_num)
            for name, param in self.global_model.state_dict().items():
                self.global_model.state_dict()[name].add_(w_avg[name])
            self.shared_state_dict = dict(self.global_model.state_dict())
        # self.shared_state_dict = average_weights(w=receive_dict, s_num=sample_num)
        return None

    def send_to_edge(self, edge):
        edge.receive_from_cloud(copy.deepcopy(self.shared_state_dict))
        return None
