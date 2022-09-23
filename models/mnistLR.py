import torch
from torch import nn


class mnistLR(nn.Module):
    def __init__(self, input_dim=1, output_dim=10):
        super(mnistLR, self).__init__()
        self.linear = nn.Linear(input_dim * 28 * 28, output_dim)

    def forward(self, x):
        x = x.view(-1, 784)
        output = self.linear(x)
        return output
