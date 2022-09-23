import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser()

    # dataset and models
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='name of the dataset: mnist, fmnist or cifar10')
    parser.add_argument('--model', type=str, default='mnistCNN',
                        help='name of model. mnist/fmnist: mnistLR, mnistCNN; cifar10: cifarCNN, cifarResNet')
    parser.add_argument('--input_channels', type=int, default=1,
                        help='input channels. mnist:1, fmnist:1, cifar10 :3')
    parser.add_argument('--output_channels', type=int, default=10,
                        help='output channels')

    # training parameter
    parser.add_argument('--prop', type=float, default=1.0,
                        help='sparsity')
    parser.add_argument('--compressed', type=int, default=0,
                        help='whether to use the model compressed, 0=not use, 1=random-k, 2=top-k, 3=quantify')
    parser.add_argument('--percentage', type=float, default=0.0,
                        help='the percentage of heterogenous clients')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size when training on the clients')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of rounds of training')
    # parser.add_argument('--num_communication', type=int, default=1,
    #                     help='number of communication rounds with the cloud server')
    parser.add_argument('--num_local_update', type=int, default=10,
                        help='number of local update (client)')
    parser.add_argument('--num_edge_aggregation', type=int, default=1,
                        help='number of edge aggregation (edge server)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate of the SGD when trained on client')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum')
    parser.add_argument('--verbose', type=int, default=1,
                        help='verbose for print progress bar')
    parser.add_argument('--model_set', type=int, default=1,
                        help='1 or 2')
    parser.add_argument('--acc_target', type=float, default=1.0,
                        help='the target of the test accuracy')

    # setting for federated learning
    parser.add_argument('--iid', type=int, default=1,
                        help='distribution of the data')
    parser.add_argument('--unequal', type=int, default=0,
                        help='distribution equal or unequal of the data')
    parser.add_argument('--num_clients', type=int, default=100,
                        help='number of all available clients')
    parser.add_argument('--frac', type=float, default=0.1,
                        help='fraction of participated clients')
    parser.add_argument('--num_edges', type=int, default=2,
                        help='number of edge servers')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (defaul: 1)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to be selected, 0, 1, 2, 3')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args
