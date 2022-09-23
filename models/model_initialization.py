"""
Used to initialize the model in the client
Include initialization, training for one iteration and test function
"""

from models.mnistLR import mnistLR
from models.mnistCNN import mnistCNN
from models.cifarCNN import cifarCNN
from models.cifarResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
import torch
from torch import nn
import torch.optim as optim


class MyModel(nn.Module):
    def __init__(self, shared_layers, specific_layers, lr, momentum):
        super(MyModel, self).__init__()
        self.shared_layers = shared_layers
        self.specific_layers = specific_layers
        self.lr = lr
        self.lr_decay = 0.99
        self.momentum = momentum
        param_dict = [{"params": self.shared_layers.parameters()}]
        if self.specific_layers:
            param_dict += [{"params": self.specific_layers.parameters()}]
        self.optimizer = optim.SGD(params=param_dict, lr=lr, momentum=momentum, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99, last_epoch=-1)
        self.optimizer_state_dict = self.optimizer.state_dict()
        self.criterion = nn.CrossEntropyLoss()

    def lr_sheduler(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.lr_decay
            return None

    def optimize_model(self, input_batch, label_batch):
        self.shared_layers.train(True)
        if self.specific_layers:
            self.specific_layers.train(True)
        if self.specific_layers:
            output_batch = self.specific_layers(self.shared_layers(input_batch))
        else:
            output_batch = self.shared_layers(input_batch)
        self.shared_layers.zero_grad()
        batch_loss = self.criterion(output_batch, label_batch)
        batch_loss.backward()
        self.optimizer.step()
        return batch_loss.item()

    def test_model(self, input_batch):
        self.shared_layers.eval()
        with torch.no_grad():
            if self.specific_layers:
                output_batch = self.specific_layers(self.shared_layers(input_batch))
            else:
                output_batch = self.shared_layers(input_batch)
        return output_batch

    def update_model(self, new_shared_layers):
        self.shared_layers.load_state_dict(new_shared_layers)


def initialize_model(args, device):
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        if args.model == 'mnistLR' or args.model == 'fmnistLR':
            shared_layers = mnistLR(input_dim=1, output_dim=10)
            specific_layers = None
        elif args.model == 'mnistCNN' or args.model == 'fmnistCNN':
            shared_layers = mnistCNN(input_channels=1, output_channels=10)
            specific_layers = None
        else:
            raise ValueError('Model not implemented for MNIST/FMNIST')
    elif args.dataset == 'cifar10' or args.dataset == 'cifar':
        if args.model == 'cifarCNN':
            shared_layers = cifarCNN(input_channels=3, output_channels=10)
            specific_layers = None
        elif args.model == 'cifarResNet':
            shared_layers = ResNet18()
            specific_layers = None
        else:
            raise ValueError('Model not implemented for CIFAR-10')
    else:
        raise ValueError('Wrong input for dataset, only mnist, fmnist and cifar10 are valid')
    if args.cuda:
        shared_layers = shared_layers.cuda(device)
    model = MyModel(shared_layers=shared_layers, specific_layers=specific_layers, lr=args.lr, momentum=args.momentum)
    return model
