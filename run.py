from asyncio import proactor_events
import copy
import random
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd
import json
import csv
import sys
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops, degree
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from torch.nn.parameter import Parameter
import csrgraph as cg
from ast import literal_eval
import warnings
import time
import gc
from dataset import PlanetoidData  # Code in the outermost folder
import tqdm

import pickle
import gzip
warnings.filterwarnings('ignore')

# Parameters
# batch_size = 16
lr = 0.005
weight_decay = 0.0005
epochs = 1000
rounds = 10
hidden_size = [128, 128, 128, 128, 128,
               128, 128, 128, 128, 128]
num_of_walks = [40, 40, 40, 40, 40,
                40, 40, 40, 40]
walk_length = [4, 4, 4, 4, 4,
               4, 4, 4, 4, 4]
data_name = ['cora', 'pubmed', 'citeseer', 'cornell', 'Dblp', 'bgp', 'Electronics'] 
start, end = 0, 1
mode = 'pagg'
save_file_name = "result"
splits_file_path = "geom-gcn/splits/"

# Seed
# random_seed = 1
# random.seed(random_seed)
# torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

# Compute Homophily

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_data_ranked(name):
    datasets = json.load(
        open("dataset.json"))
    dataset_run = datasets[name]["dataset"]
    dataset_path = datasets[name]["dataset_path"][0]
    dataset_path = "dataset" / Path(dataset_path)
    val_size = datasets[name]["val_size"]

    dataset = PlanetoidData(
        dataset_str=dataset_run, dataset_path=dataset_path, val_size=val_size
    )

    # adj = dataset._sparse_data["sparse_adj"]
    features = dataset._sparse_data["features"]
    labels = dataset._dense_data["y_all"]

    # n_nodes, n_feats = features.shape[0], features.shape[1]
    num_classes = labels.shape[-1]

    # G = cg.csrgraph(adj, threads=0)
    # G.set_threads(0)  # number of threads to use. 0 is full use
    # edge = nx.from_scipy_sparse_matrix(adj)  # indices + edge_weight
    X = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(np.argmax(labels, 1), dtype=torch.long)
    return (X, label, num_classes, datasets)


def get_order(ratio: list, masked_index: torch.Tensor, total_node_num: int, seed: int = 1234567):
    random.seed(seed)

    masked_node_num = len(masked_index)
    shuffle_criterion = list(range(masked_node_num))
    random.shuffle(shuffle_criterion)

    # train_val_test_list=[int(i) for i in ratio.split('-')]
    train_val_test_list = ratio
    tvt_sum = sum(train_val_test_list)
    tvt_ratio_list = [i/tvt_sum for i in train_val_test_list]
    train_end_index = int(tvt_ratio_list[0]*masked_node_num)
    val_end_index = train_end_index+int(tvt_ratio_list[1]*masked_node_num)

    train_mask_index = shuffle_criterion[:train_end_index]
    val_mask_index = shuffle_criterion[train_end_index:val_end_index]
    test_mask_index = shuffle_criterion[val_end_index:]

    train_mask = torch.zeros(total_node_num, dtype=torch.bool)
    train_mask[masked_index[train_mask_index]] = True
    val_mask = torch.zeros(total_node_num, dtype=torch.bool)
    val_mask[masked_index[val_mask_index]] = True
    test_mask = torch.zeros(total_node_num, dtype=torch.bool)
    test_mask[masked_index[test_mask_index]] = True

    return (train_mask, val_mask, test_mask)


def get_whole_mask(y, ratio: list = [48, 32, 20], seed: int = 1234567):
    y_have_label_mask = y != -1
    total_node_num = len(y)
    y_index_tensor = torch.tensor(list(range(total_node_num)), dtype=int)
    masked_index = y_index_tensor[y_have_label_mask]
    while True:
        (train_mask, val_mask, test_mask) = get_order(
            ratio, masked_index, total_node_num, seed)
        # if check_train_containing(train_mask,y):
        return (train_mask, val_mask, test_mask)
        # else:
        #     seed+=1


def load_data(dataset_name, round):
    numpy_x = np.load("/data/syf"+'/'+dataset_name+'/x.npy')
    x = torch.from_numpy(numpy_x).to(torch.float)
    numpy_y = np.load("/data/syf"+'/'+dataset_name+'/y.npy')
    y = torch.from_numpy(numpy_y).to(torch.long)
    # numpy_edge_index = np.load("/data/syf"+'/'+dataset_name+'/edge_index.npy')
    # edge_index = torch.from_numpy(numpy_edge_index).to(torch.long)
    (train_mask, val_mask, test_mask) = get_whole_mask(y, seed=round+1)

    lbl_set = []
    for lbl in y:
        if lbl not in lbl_set:
            lbl_set.append(lbl)
    num_classes = len(lbl_set)

    return x, y, num_classes, train_mask, val_mask, test_mask


class PAGG(MessagePassing):
    def __init__(self, feature_length, hidden_size, out_size, node_num, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super(PAGG, self).__init__()
        self.feature_length, self.hidden_size, self.out_size, self.node_num \
            = feature_length, hidden_size, out_size, node_num

        self.fc0 = torch.nn.Linear(feature_length, hidden_size)
        self.RNN = nn.RNN(hidden_size, hidden_size)
        self.fc2 = torch.nn.Linear(2*hidden_size, out_size)
        self.nei0 = torch.nn.Linear(hidden_size, hidden_size)
        self.nei1 = torch.nn.Linear(hidden_size, hidden_size)
        self.nei2 = torch.nn.Linear(hidden_size, hidden_size)
        self.nei3 = torch.nn.Linear(hidden_size, hidden_size)
        self.nei4 = torch.nn.Linear(hidden_size, hidden_size)
        # self.fc3 = torch.nn.Linear(hidden_size, out_size)
        torch.nn.init.xavier_uniform_(self.fc0.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.nei0.weight)
        torch.nn.init.xavier_uniform_(self.nei1.weight)
        torch.nn.init.xavier_uniform_(self.nei2.weight)
        torch.nn.init.xavier_uniform_(self.nei3.weight)
        # self.RNN_dict = {0: self.nei0,
        #                  1: self.nei1,
        #                  2: self.nei2,
        #                  3: self.nei3,
        #                  }

    def forward(self, X, neis, num_w, walk_len, indices, layer_type, indxx):
        split = sum(indices)
        # X = X.to(device)
        X = self.fc0(X)
        neis = neis.to(device)
        # layer_type = layer_type.to(device)
        # torch.Size([3480*4, 128])
        nei = X[neis].view(split*num_w*walk_len, self.hidden_size)
        # nei = nei.transpose(0, 1)  # (walk_len, split*num_w, self.hidden_size)
        # nei = torch.flip(nei, dims=[0])  # 最后节点是ego
        layer_type = layer_type.view(split*num_w*walk_len).to(device)

        nei0 = self.nei0(nei)
        nei1 = self.nei1(nei)
        nei2 = self.nei2(nei)
        nei3 = self.nei3(nei)
        nei4 = self.nei4(nei)
        neis_cat = torch.stack((nei0, nei1, nei2, nei3, nei4), dim=1)

        nei = neis_cat[indxx, layer_type].view(
            split*num_w, walk_len, self.hidden_size).transpose(0, 1)
        # print(nei.shape)  # torch.Size([4, 3480, 128])
        nei = F.dropout(nei, p=0.9, training=self.training)
        nei, h_n = self.RNN(nei)
        h_n = h_n.transpose(0, 1).view(
            split, num_w, -1)  # [V, num_of_walks, H]

        h_n = torch.mean(h_n, dim=1)
        ego = X[indices]
        layer1 = torch.cat((ego, h_n), dim=1)  # [V, 2*H]
        # layer1 = F.relu(self.fc2(layer1))
        layer1 = F.dropout(layer1, p=0.9, training=self.training)
        dout = self.fc2(layer1)
        return dout

# def RNNReLUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
#     hy = F.relu(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh, b_hh))
#     return hy


class RNNMean(MessagePassing):
    def __init__(self, feature_length, hidden_size, out_size, node_num, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super(RNNMean, self).__init__()
        self.feature_length, self.hidden_size, self.out_size, self.node_num \
            = feature_length, hidden_size, out_size, node_num

        self.fc0 = torch.nn.Linear(feature_length, hidden_size)
        self.RNNCell_1 = nn.RNNCell(hidden_size, hidden_size)
        self.RNNCell_2 = nn.RNNCell(hidden_size, hidden_size)
        self.RNNCell_3 = nn.RNNCell(hidden_size, hidden_size)
        self.RNNCell_4 = nn.RNNCell(hidden_size, hidden_size)
        # self.RNN_dict = {1: self.RNNCell_1,
        #                  2: self.RNNCell_2,
        #                  3: self.RNNCell_3,
        #                  4: self.RNNCell_4,
        #                  }
        self.fc2 = torch.nn.Linear(2*hidden_size, out_size) 
        # self.fc3 = torch.nn.Linear(hidden_size, out_size)
        torch.nn.init.xavier_uniform_(self.fc0.weight)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        # torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, X, neis, num_w, walk_len, indices, layer_type):
        split = sum(indices)
        # X = X.to(device)
        X = self.fc0(X)
        neis = neis.to(device)
        # layer_type = layer_type.to(device)
        nei = X[neis].view(split*num_w, walk_len, self.hidden_size)

        nei = nei.transpose(0, 1)
        nei = torch.flip(nei, dims=[0]) 
        # print(nei.shape)  # torch.Size([4, 3480, 128])

        h_t = torch.zeros(split*num_w, self.hidden_size).to(device)
        # print(layer_type.shape)
        layer_type = layer_type.view(split*num_w, walk_len).to(device)
        # print(layer_type.shape)
        # one_hot = F.one_hot(layer_type, 4).view(
        #     split*num_w, walk_len, 4).float().to(device)
        for i in range(nei.size(0)):
            nei_step = nei[i]
            now = layer_type[:, i]
            h_1 = self.RNNCell_1(nei_step, h_t).unsqueeze(1)
            h_2 = self.RNNCell_2(nei_step, h_t).unsqueeze(1)
            h_3 = self.RNNCell_3(nei_step, h_t).unsqueeze(1)
            h_4 = self.RNNCell_4(nei_step, h_t).unsqueeze(1)
            h_t = torch.cat((h_1, h_2, h_3, h_4), dim=1)  # 0 node 1 walk
            # print(h_t.shape)  # torch.Size([3480, 4, 128])
            h_t = h_t[torch.arange(
                split*num_w, dtype=torch.long, device=device), now]

        h_t = h_t.view(
            split, num_w, -1)  # [V, num_of_walks, H]

        h_t = torch.mean(h_t, dim=1)
        ego = X[indices]
        layer1 = torch.cat((ego, h_t), dim=1)  # [V, 2*H]
        # layer1 = F.relu(self.fc2(layer1))
        layer1 = F.dropout(layer1, p=0.9, training=self.training)
        dout = self.fc2(layer1)
        return dout


def train_fixed_indices(X, Y, num_classes, mode, data_name, train_indices, val_indices, test_indices, num_w, hid_size, walk_len, walks, path_type_all):
    feature_length = X.shape[-1]
    node_num = Y.shape[0]
    # Construct the model
    if mode == 'our':
        predictor = RNNMean(feature_length, hid_size,
                            num_classes, node_num).to(device)
    elif mode == 'pagg':
        predictor = PAGG(feature_length, hid_size,
                             num_classes, node_num).to(device)
    elif mode == 'mlp':
        predictor = MLP(feature_length, hid_size, num_classes).to(device)
    elif mode == 'gat':
        predictor = GAT(feature_length, hid_size, num_classes).to(device)

    optimizer = torch.optim.Adam(
        predictor.parameters(), lr=lr, weight_decay=weight_decay)
    lossfunc = torch.nn.CrossEntropyLoss()

    # prep data
    X = X.to(device)
    # Y = Y.to(device)

    # Start training
    test_1f1, test_2f1, test_rec, test_prec, test_acc = 0, 0, 0, 0, 0
    max_val_acc = 0
    max_val_2f1 = 0
    val_2f1 = 0
    val_acc = 0  # validation
    train_bar = tqdm.tqdm(range(epochs), dynamic_ncols=True, unit='step')
    loss_collector, val_acc_coll = [], []
    if name in ['cora', 'citeseer', 'cornell', "Nba", 'wisconsin', 'texas']:
        neis_all = torch.tensor(walks, dtype=torch.long).view(
            1000, node_num, -1)
        path_type_all = torch.tensor(path_type_all, dtype=torch.long).view( 
            1000, node_num, num_w, walk_len)  # -torch.ones(epochs, node_num, num_w, walk_len).long()

    for epoch in train_bar:
        time1 = time.time()
        if name in ['Dblp', 'bgp', 'Electronics', 'film', 'pubmed']:
            walks = []
            path_type = []
            path_file = "{}_{}_{}_hr_{:04d}.txt".format(
                name, num_w, walk_len, epoch)
            # "/data/syf/rw/"+
            with open(path_file, "r") as p:
                for line in p:
                    info = list(map(int, line[1:-2].split(",")))
                    walks.append(info[:walk_len])
                    path_type.append(info[walk_len:])
            neis = torch.tensor(walks, dtype=torch.long).view(node_num, -1)
            # print(path_type[0])
            # print(len(neis), len(path_type), len(path_type[0]))
            path_type = torch.tensor(path_type, dtype=torch.long).view(
                node_num, num_w, walk_len)

        elif name in ['cora', 'citeseer', 'cornell', "Nba", 'wisconsin', 'texas']:
            neis = neis_all[epoch]
            path_type = path_type_all[epoch]
        predictor.train()
        # print(path_type[train_indices].shape)
        # node_set = list(set(neis[train_indices].reshape(-1).tolist()))
        indxx = torch.arange(
            sum(train_indices)*num_w*walk_len, dtype=torch.long, device=device)
        time2 = time.time()
        y_hat = predictor(X, neis[train_indices],
                          num_w, walk_len, train_indices, path_type[train_indices], indxx)  # transductive!! path_type[train_indices]
        loss = lossfunc(y_hat, Y[train_indices].to(device))
        # loss_collector.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        time3 = time.time()
        last_test_acc = 0
        with torch.no_grad():
            predictor.eval()
            # neis[val_indices] = neis[val_indices].to(device)
            # node_set = list(set(neis[val_indices].reshape(-1).tolist()))
            indxx = torch.arange(
                sum(val_indices)*num_w*walk_len, dtype=torch.long, device=device)
            y_hat = F.log_softmax(
                predictor(X, neis[val_indices], num_w, walk_len, val_indices, path_type[val_indices], indxx), dim=1)
            # neis[val_indices] = neis[val_indices].to('cpu')
            # total_val_loss = lossfunc(y_hat[val_indices], Y[val_indices]).item()
            y_hat_ = y_hat.cpu().max(1)[1]
            val_acc = accuracy_score(Y[val_indices], y_hat_)
            # val_1f1, val_2f1, val_rec, val_prec, val_acc = (
            #     f1_score(Y[val_indices], y_hat_, average="macro"),
            #     f1_score(Y[val_indices], y_hat_, average="micro"),
            #     recall_score(Y[val_indices], y_hat_, average="macro"),
            #     precision_score(Y[val_indices], y_hat_, average="macro"),
            #     accuracy_score(Y[val_indices], y_hat_))
            # print(data_name, "val_acc", val_acc)
            # val_acc_coll.append(val_acc)
            time4 = time.time()
            if max_val_acc < val_acc:
                # if val_acc > max_val_acc:
                max_val_acc = val_acc
                # max_val_2f1 = val_2f1
                # print("Save Model.")
                torch.save(predictor.state_dict(),
                           "models/" + save_file_name + ".pth")
                # test_acc = 0
                # for k in range(test_rw_round):
                #     ch = random.choice(list(range(epochs)))
                #     neis = neis_all[ch]
                # neis[test_indices] = neis[test_indices].to(device)
                # node_set = list(set(neis[test_indices].reshape(-1).tolist()))
                indxx = torch.arange(
                    sum(test_indices)*num_w*walk_len, dtype=torch.long, device=device)
                y_hat = F.log_softmax(
                    predictor(X, neis[test_indices], num_w, walk_len, test_indices, path_type[test_indices], indxx), dim=1)
                # neis[test_indices] = neis[test_indices].to('cpu')
                y_hat_ = y_hat.cpu().max(1)[1]
                # test_acc = accuracy_score(Y[test_indices], y_hat_)
                test_1f1, test_2f1, test_rec, test_prec, test_acc = (
                    f1_score(Y[test_indices], y_hat_, average="macro"),
                    f1_score(Y[test_indices], y_hat_, average="micro"),
                    recall_score(Y[test_indices], y_hat_, average="macro"),
                    precision_score(Y[test_indices], y_hat_, average="macro"),
                    accuracy_score(Y[test_indices], y_hat_))
            time5 = time.time()
            #     test_acc += test_tmp
            # test_acc /= test_rw_round
            train_bar.set_postfix(
                data=data_name, val_acc=val_acc, test_acc=test_acc, test_2f1=test_2f1, test_1f1=test_1f1)
        # print(
        #     f"before train: {time2-time1}; train: {time3-time2}; val: {time4-time3}; test: {time5-time4}")
        # gc.collect()
    return test_1f1, test_2f1, test_rec, test_prec, test_acc  # val_acc is a list


file = open("results/" + save_file_name + ".txt", "a")
for k in range(start, end):
    name = data_name[k]
    print(name)
    walks = []
    path_type = []
    if name in ['cora', 'citeseer', 'cornell', "Nba", 'wisconsin', 'texas']:
        # "/data/syf/rw/"+
        path_file = "{}_{}_{}_m.txt".format(
            name, num_of_walks[0], walk_length[0])
        # "/data/syf/rw/"+
        with open(path_file, "r") as p:
            for line in p:
                info = list(map(int, line[1:-2].split(",")))
                walks.append(info[:walk_length[0]])
                path_type.append(info[walk_length[0]:])
        print(len(walks), len(path_type), len(path_type[0]))

    avg_test_1f1, avg_test_2f1, avg_test_rec, avg_test_prec, avg_test_acc, \
        std_test_1f1, std_test_2f1, std_test_rec, std_test_prec, std_test_acc = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    test_1f1s, test_2f1s, test_recs, test_precs, test_accs = [], [], [], [], []
    # train_l,val_ac=np.empty(epochs),np.empty(epochs)
    train_l, val_ac = [], []
    if name not in ['Dblp', 'bgp', 'Nba', 'Electronics']:
        (X, Y, num_classes, datasets) = load_data_ranked(name)

    for i in range(rounds):
        print('round', i)
        if name in ['Dblp', 'bgp', 'Nba', 'Electronics']:
            (X, Y, num_classes, train_mask, val_mask, test_mask) = load_data(name, i)
        else:
            dataset_run = datasets[name]["dataset"]
            dataset_path = datasets[name]["dataset_path"][i]
            dataset_path = "datasets" / \
                Path(dataset_path)
            val_size = datasets[name]["val_size"]

            dataset = PlanetoidData(
                dataset_str=dataset_run, dataset_path=dataset_path, val_size=val_size
            )

            train_mask = dataset._dense_data["train_mask"]
            val_mask = dataset._dense_data["val_mask"]
            test_mask = dataset._dense_data["test_mask"]
        test_1f1, test_2f1, test_rec, test_prec, test_acc = train_fixed_indices(
            X, Y, num_classes, mode, name, train_mask, val_mask, test_mask, num_of_walks[k], hidden_size[k], walk_length[k], walks, path_type)
        test_recs.append(test_rec)
        test_accs.append(test_acc)
        test_1f1s.append(test_1f1)
        test_2f1s.append(test_2f1)
        test_precs.append(test_prec)
    # for i in range(rounds):
    #     plt.subplot(4, 4, i+1)
    #     plt.plot(train_l[i])
    # plt.savefig("/home/syf/workspace/results/train_loss_" +
    #             name+'_'+".png", dpi=200)
    # plt.clf()
    # for i in range(rounds):
    #     plt.subplot(4, 4, i+1)
    #     plt.plot(val_ac[i])
    # plt.savefig("/home/syf/workspace/results/val_acc_" +
    #             name+'_'+".png", dpi=200)
    # plt.clf()

    #     train_l+=train_loss
    #     val_ac+=val_acc
    # train_l/=rounds
    # val_ac/=rounds

    avg_test_rec = sum(test_recs) / rounds
    avg_test_acc = sum(test_accs) / rounds
    avg_test_1f1 = sum(test_1f1s) / rounds
    avg_test_2f1 = sum(test_2f1s) / rounds
    avg_test_prec = sum(test_precs) / rounds
    # avg_test_1f1 = sum(test_1f1s) / rounds

    std_test_rec = np.std(np.array(test_recs))
    std_test_acc = np.std(np.array(test_accs))
    std_test_1f1 = np.std(np.array(test_1f1s))
    std_test_2f1 = np.std(np.array(test_2f1s))
    std_test_prec = np.std(np.array(test_precs))

    print(name+"_"+str(num_of_walks[k])+"_" +
          str(walk_length[k])+'_'+str(hidden_size[k])+"\n")
    print(mode+" Avg for {}: acc{:.4f} ± {:.4f}\t prec{:.4f} ± {:.4f}\t rec{:.4f} ± {:.4f}\t maf1{:.4f} ± {:.4f}\t mif1{:.4f} ± {:.4f}\t ".format(
        name, avg_test_acc, std_test_acc, avg_test_prec, std_test_prec, avg_test_rec, std_test_rec, avg_test_1f1, std_test_1f1, avg_test_2f1, std_test_2f1))
    print(name+"_"+str(num_of_walks[k])+"_"+str(walk_length[k]
                                                )+'_'+str(hidden_size[k])+"\n", file=file)
    print(mode+" Avg for {}: acc{:.4f} ± {:.4f}\t prec{:.4f} ± {:.4f}\t rec{:.4f} ± {:.4f}\t maf1{:.4f} ± {:.4f}\t mif1{:.4f} ± {:.4f}\t ".format(
        name, avg_test_acc, std_test_acc, avg_test_prec, std_test_prec, avg_test_rec, std_test_rec, avg_test_1f1, std_test_1f1, avg_test_2f1, std_test_2f1), file=file)
