"""
Client <--> Edge Server <--> Cloud Server
"""

from options import args_parser
from tensorboardX import SummaryWriter
from client import Client
from edge import Edge
from cloud import Cloud
from datasets.get_data import get_dataloaders, show_distribution
from models.mnistLR import mnistLR
from models.mnistCNN import mnistCNN
from models.cifarCNN import cifarCNN
from models.cifarResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
import torch
from torch import nn
import copy
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import torch.backends.cudnn as cudnn
cudnn.banchmark = True
cudnn.enabled = True


def initialize_global_model(args):
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        if args.model == 'mnistLR' or args.model == 'fmnistLR':
            global_model = mnistLR(input_dim=1, output_dim=10)
        elif args.model == 'mnistCNN' or args.model == 'fmnistCNN':
            global_model = mnistCNN(input_channels=1, output_channels=10)
        else:
            raise ValueError(f"Model{args.model} not implemented for mnist/fmnist")
    elif args.dataset == 'cifar10' or args.dataset == 'cifar':
        if args.model == 'cifarCNN':
            global_model = cifarCNN(input_channels=3, output_channels=10)
        elif args.model == 'cifarResNet':
            global_model = ResNet18()
        else:
            raise ValueError(f"Model{args.model} not implemented for cifar")
    else:
        raise ValueError('Wrong input for dataset, only mnist, fmnist and cifar10 are valid')
    return global_model


def all_client_test(server, clients, client_ids, device):
    for client_id in client_ids:
        server.send_to_client(clients[client_id])
        clients[client_id].sync_with_edge()

    correct_client = 0.0
    total_client = 0.0
    for client_id in client_ids:
        correct, total = clients[client_id].test_model(device)
        correct_client += correct
        total_client += total
    return correct_client, total_client


def global_model_test(v_test_loader, global_model, device):
    correct_all = 0.0
    total_all = 0.0
    with torch.no_grad():
        for data in v_test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = global_model(inputs)
            _, preds = torch.max(outputs, 1)
            correct_all += (preds == labels).sum().item()
            total_all += labels.size(0)
    return correct_all, total_all


def GenerateLocalEpochs(percentage, size, max_epochs):  # 生成本地训练回合
    """
    Generates list of epochs for selected clients to replicate system heterogeneity
        :param percentage: percentage of clients to have fewer than E epochs
        :param size: total size of the list
        :param max_epochs: maximum value for local epochs
        :return: List of size epochs for each Client Update
    """
    # if percentage is 0 then each client runs for E epochs
    if percentage == 0.0:
        return np.array([max_epochs] * size)
    else:
        heterogenous_size = int(percentage * size)
        # generate random uniform epochs of heterogenous size between 1 and E
        epoch_list = np.random.randint(1, max_epochs, heterogenous_size)
        remaining_size = size - heterogenous_size
        remaining_list = [max_epochs] * remaining_size
        epoch_list = np.append(epoch_list, remaining_list, axis=0)
        np.random.shuffle(epoch_list)  # shuffle the list and return
        print("The epoch_list is ", epoch_list)
        return epoch_list


def myFedAvg(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        cuda_to_use = torch.device(f'cuda:{args.gpu}')
    device = cuda_to_use if torch.cuda.is_available() else "cpu"
    print(f'Using device {device}')

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # GPU_id

    # FILEOUT = f"{args.dataset}_clients{args.num_clients}_edges{args.num_edges}_" \
    #           f"t1-{args.num_local_update}_t2-{args.num_edge_aggregation}" \
    #           f"_model_{args.model}iid{args.iid}epoch{args.epochs}" \
    #           f"bs{args.batch_size}lr{args.lr}momentum{args.momentum}"
    # writer = SummaryWriter(comment=FILEOUT)

    # get train_dataset(need to split) and test_dataset
    train_loaders, test_loaders, v_train_loader, v_test_loader = get_dataloaders(args)

    # global model initialization
    global_model = initialize_global_model(args)
    if args.cuda:
        global_model = global_model.cuda(device)

    # client model initialization
    clients = []
    for i in range(args.num_clients):
        clients.append(Client(id=i, train_loader=train_loaders[i], test_loader=test_loaders[i],
                              args=args, device=device, global_model=global_model))
    initialization_parameters = list(clients[0].model.shared_layers.parameters())
    parameters_length = len(initialization_parameters)
    for client in clients:
        client_parameters = list(client.model.shared_layers.parameters())
        for i in range(parameters_length):
            client_parameters[i].data[:] = initialization_parameters[i].data[:]  # initialization model parameters

    # edge initialization
    edges = []
    cids = np.arange(args.num_clients)
    client_per_edge = int(args.num_clients / args.num_edges)
    p_clients = [0.0] * args.num_edges
    for i in range(args.num_edges):
        selected_client_id = np.random.choice(cids, client_per_edge, replace=False)
        cids = list(set(cids) - set(selected_client_id))
        edges.append(Edge(id=i, client_ids=selected_client_id, args=args,
                          shared_layers=copy.deepcopy(clients[0].model.shared_layers), global_model=global_model))

        for sid in selected_client_id:
            edges[i].client_register(clients[sid])
        edges[i].all_trainsample_num = sum(edges[i].sample_registration.values())
        p_clients[i] = [sample / float(edges[i].all_trainsample_num) for sample in
                        list(edges[i].sample_registration.values())]
        edges[i].refresh_edge()

    # cloud initialization
    cloud = Cloud(shared_layers=copy.deepcopy(clients[0].model.shared_layers), global_model=global_model, args=args)
    for edge in edges:
        cloud.edge_registration(edge=edge)
    p_edges = [sample / sum(cloud.sample_registration.values()) for sample in
              list(cloud.sample_registration.values())]
    cloud.refresh_cloud()

    train_loss, train_acc, test_acc, train_time = [], [], [], [0.0] * args.epochs
    edge_compress_ratio, complete_epoch = [], [0] * args.epochs

    # begin training
    for epoch in tqdm(range(args.epochs)):
        cloud.refresh_cloud()
        for edge in edges:
            cloud.edge_registration(edge=edge)

        # client <--> edge
        for num_edge_agg in range(args.num_edge_aggregation):
            edge_loss = [0.0] * args.num_edges
            edge_sample = [0] * args.num_edges
            correct_client = 0.0
            total_client = 0.0
            for i, edge in enumerate(edges):
                edge.refresh_edge()
                client_loss = 0.0
                m = max(int(args.frac * client_per_edge), 1)
                selected_client = np.random.choice(edge.client_ids, m, replace=False, p=p_clients[i])
                heterogenous_list = GenerateLocalEpochs(args.percentage, m, args.num_local_update)  # local epochs

                temp_time = []
                for idx in selected_client:
                    edge.client_register(clients[idx])
                for j, idx in enumerate(selected_client):
                    edge.send_to_client(clients[idx])  # send model
                    clients[idx].sync_with_edge()  # update local model
                    loss, client_train_time = clients[idx].local_update(local_epoch=heterogenous_list[j],
                                                                        device=device)  # cal
                    # loss, client_train_time, temp = clients[idx].local_update(local_epoch=heterogenous_list[j], device=device)  # cal
                    # print(temp)
                    client_loss += loss
                    temp_time.append(client_train_time)
                    clients[idx].send_to_edge(edge)  # push to edge server
                print(temp_time)
                train_time[epoch] = max(train_time[epoch], max(temp_time))  # time depends on the slowest client
                edge_loss[i] = client_loss
                edge_sample[i] = sum(edge.sample_registration.values())

                edge.aggregate(args)  # aggregation
                # correct, total = all_client_test(edge, clients, edge.client_ids, device)
                # correct_client += correct  # calculate the training acc
                # total_client += total

            all_client_loss = sum([e_loss * e_sample for e_loss, e_sample in zip(edge_loss, edge_sample)]) / sum(edge_sample)
            # avg_client_acc = correct_client / total_client
            train_loss.append(all_client_loss)
            # train_acc.append(avg_client_acc)
            print(f"The Training Loss is {all_client_loss}")
            # print("The Average Training Test Acc is {:.2f}%".format(100* avg_client_acc))
            print("The max client training time is {:.4f}s".format(train_time[epoch]))

        # edge <--> cloud
        temp_compress_ratio = 0.0
        for edge in edges:
            if args.compressed == 2 and args.model_set == 2:
                compress_ratio = edge.send_to_cloud(cloud)
                temp_compress_ratio += compress_ratio
            else:
                edge.send_to_cloud(cloud)
        edge_compress_ratio.append(temp_compress_ratio / len(edges))
        print("The Edge Compress Ratio is {:.6f}".format(edge_compress_ratio[-1]))
        cloud.aggregate(args)
        for edge in edges:
            cloud.send_to_edge(edge)

        global_model.load_state_dict(state_dict=copy.deepcopy(cloud.shared_state_dict))
        global_model.eval()
        correct_all, total_all = global_model_test(v_test_loader=v_test_loader, global_model=global_model, device=device)
        avg_acc = correct_all / total_all
        test_acc.append(avg_acc)
        print("The Global Model Test Acc is {:.2f}%".format(100 * avg_acc))
        # writer.add_scalar(f'Test_Acc', avg_acc, epoch + 1)

        if epoch >= 4 and len([i for i in test_acc[-5:] if i > args.acc_target]) == 5:  # 连续五轮大于acc_target
            complete_epoch[epoch] = epoch + 1

    # writer.close()
    print("The final Acc is {:.2f}%".format(100 * (max(test_acc))))

    file_dir = "/root/results_new"
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    xlsx_name = "{}_{}_{}_iid[{}]_frac[{}]_lr[{}]_E[{}]_P[{}]_C[{}]_prop[{}]_M[{}].xlsx".\
        format(args.dataset, args.model, args.epochs, args.iid, args.frac, args.lr, args.num_local_update, args.percentage,
               args.compressed, args.prop, args.model_set)
    res = [train_loss, test_acc, train_time, edge_compress_ratio, complete_epoch]
    columns = ['train_loss', 'test_acc', 'train_time', 'edge_compress_ratio', 'complete_epoch']
    dt = pd.DataFrame(res, index=columns)
    dt.to_excel(os.path.join(file_dir, xlsx_name))


def main():
    args = args_parser()
    myFedAvg(args)


if __name__ == '__main__':
    main()
