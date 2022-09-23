"""
Download the required dataset, split the data among the clients, and generate DataLoader for training
"""

import os
from tqdm import tqdm
from sklearn import metrics
import numpy as np

import torch
import torch.backends.cudnn as cudnn
cudnn.banchmark = True
cudnn.enabled = True

import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from options import args_parser


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        super(DatasetSplit, self).__init__()
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, target = self.dataset[self.idxs[item]]
        return image, target


def gen_ran_sum(_sum, num_users):
    base = 100 * np.ones(num_users, dtype=np.int32)
    _sum = _sum - 100 * num_users
    p = np.random.dirichlet(np.ones(num_users), size=1)
    print(p.sum())
    p = p[0]
    size_users = np.random.multinomial(_sum, p, size=1)[0]
    size_users = size_users + base
    print(size_users.sum())
    return size_users


def iid_esize_split(dataset, args, kwargs, is_shuffle=True):
    """
    Equally split the dataset to clients
    """
    sum_samples = len(dataset)
    num_samples_per_client = int(sum_samples / args.num_clients)
    data_loaders = [0] * args.num_clients
    dict_users, all_idxs = {}, [i for i in range(sum_samples)]
    for i in range(args.num_clients):
        dict_users[i] = np.random.choice(all_idxs, num_samples_per_client, replace=False)
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]), batch_size=args.batch_size,
                                     shuffle=is_shuffle, **kwargs)
    return data_loaders


def iid_nesize_split(dataset, args, kwargs, is_shuffle=True):
    """
    Unequally split the dataset to clients
    """
    sum_samples = len(dataset)
    num_samples_per_client = gen_ran_sum(sum_samples, args.num_clients)
    data_loaders = [0] * args.num_clients
    dict_users, all_idxs = {}, [i for i in range(sum_samples)]
    for (i, num_samples_client) in enumerate(num_samples_per_client):
        dict_users[i] = np.random.choice(all_idxs, num_samples_client, replace=False)
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]), batch_size=args.batch_size,
                                     shuffle=is_shuffle, **kwargs)
    return data_loaders


def niid_esize_split(dataset, args, kwargs, is_shuffle=True):
    """
    Equally split the dataset to clients(Non-IID)
    """
    data_loaders = [0] * args.num_clients
    num_shards = 2 * args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    # is_shuffle is used to differentiate between train and test
    if is_shuffle:
        labels = dataset.train_labels
    else:
        labels = dataset.test_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # sort the data according to their label
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)

    for i in range(args.num_clients):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]), batch_size=args.batch_size,
                                     shuffle=is_shuffle, **kwargs)
    return data_loaders


def split_data(dataset, args, kwargs, is_shuffle=True):
    """
    Return data_loaders
    """
    if args.iid == 1:
        data_loaders = iid_esize_split(dataset, args, kwargs, is_shuffle)
    elif args.iid == 0:
        data_loaders = niid_esize_split(dataset, args, kwargs, is_shuffle)
    elif args.iid == -1:
        data_loaders = iid_nesize_split(dataset, args, kwargs, is_shuffle)
    else:
        raise ValueError('Data Distribution pattern `{}` not implemented '.format(args.iid))
    return data_loaders


def get_mnist(args):
    data_dir = '../datasets/mnist/'
    is_cuda = args.cuda
    kwargs = {'num_workers': 0, 'pin_memory': True} if is_cuda else {}
    apply_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)
    train_loaders = split_data(train, args, kwargs, is_shuffle=True)
    test_loaders = split_data(test, args, kwargs, is_shuffle=False)
    v_train_loader = DataLoader(train, batch_size=args.batch_size * args.num_clients, shuffle=True, **kwargs)
    v_test_loader = DataLoader(test, batch_size=args.batch_size * args.num_clients, shuffle=False, **kwargs)
    return train_loaders, test_loaders, v_train_loader, v_test_loader


def get_fmnist(args):
    data_dir = '../datasets/fmnist/'
    is_cuda = args.cuda
    kwargs = {'num_workers': 0, 'pin_memory': True} if is_cuda else {}
    apply_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.FashionMNIST(data_dir, train=True, download=True, transform=apply_transform)
    test = datasets.FashionMNIST(data_dir, train=False, download=True, transform=apply_transform)
    train_loaders = split_data(train, args, kwargs, is_shuffle=True)
    test_loaders = split_data(test, args, kwargs, is_shuffle=False)
    v_train_loader = DataLoader(train, batch_size=args.batch_size * args.num_clients, shuffle=True, **kwargs)
    v_test_loader = DataLoader(test, batch_size=args.batch_size * args.num_clients, shuffle=False, **kwargs)
    return train_loaders, test_loaders, v_train_loader, v_test_loader


def get_cifar(args):
    data_dir = '../datasets/cifar/'
    is_cuda = args.cuda
    kwargs = {'num_workers': 0, 'pin_memory': True} if is_cuda else {}
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
    train_loaders = split_data(train, args, kwargs, is_shuffle=True)
    test_loaders = split_data(test, args, kwargs, is_shuffle=False)
    v_train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)
    v_test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, **kwargs)
    return train_loaders, test_loaders, v_train_loader, v_test_loader


def get_dataset(dataset, args):
    train_loaders, v_train_loader, test_loaders, v_test_loader = {}, {}, {}, {}
    if args.dataset == 'mnist':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_mnist(args)
    elif args.dataset == 'fmnist':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_fmnist(args)
    elif args.dataset == 'cifar':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_cifar(args)
    else:
        raise ValueError('Dataset `{}` not found'.format(dataset))
    return train_loaders, test_loaders, v_train_loader, v_test_loader


def show_distribution(dataloader, args):
    """
    show the distribution of the data on certain client with dataloader
    return:
        percentage of each class of the label
    """
    if args.dataset == 'mnist':
        try:
            labels = dataloader.dataset.dataset.train_labels.numpy()
        except:
            print(f"Using test_labels")
            labels = dataloader.dataset.dataset.test_labels.numpy()
        # labels = dataloader.dataset.dataset.train_labels.numpy()
    elif args.dataset == 'fmnist':
        try:
            labels = dataloader.dataset.dataset.train_labels.numpy()
        except:
            print(f"Using test_labels")
            labels = dataloader.dataset.dataset.test_labels.numpy()
        # labels = dataloader.dataset.dataset.train_labels.numpy()
    elif args.dataset == 'cifar':
        try:
            labels = dataloader.dataset.dataset.train_labels
        except:
            print(f"Using test_labels")
            labels = dataloader.dataset.dataset.test_labels
        # labels = dataloader.dataset.dataset.train_labels
    else:
        raise ValueError("`{}` dataset not included".format(args.dataset))
    num_samples = len(dataloader.dataset)
    # print(num_samples)
    idxs = [i for i in range(num_samples)]
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    distribution = [0] * len(unique_labels)
    for idx in idxs:
        img, label = dataloader.dataset[idx]
        distribution[label] += 1
    distribution = np.array(distribution)
    distribution = distribution / num_samples
    return distribution
