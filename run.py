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

import argparse
import pickle
import gzip
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('-c', '--cuda', type=str, default='cuda:0')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.005)
parser.add_argument('-dr', '--dropout', type=float, default=0.9)
parser.add_argument('-e', '--epoch', type=int, default=1000)
parser.add_argument('-wd', '--weight_decay', type=float, default=0.0005)
parser.add_argument('-r', '--round', type=int, default=10)
parser.add_argument('-hid', '--hidden_size', type=int, default=128)
parser.add_argument('-nw', '--num_of_walks', type=int, default=40)
parser.add_argument('-wl', '--walk_length', type=int, default=4)
parser.add_argument('-data', '--data_name', type=str, default='Cornell')
parser.add_argument('-mode', '--model_mode', type=str, default='pagg')
# parser.add_argument('-r', '--rand_seed', type=int, default=2)
args = parser.parse_args()

# Parameters
# batch_size = 16
lr = args.learning_rate
weight_decay = args.weight_decay
dropout=args.dropout
epochs = args.epoch
rounds = args.round
num_of_walks=args.num_of_walks
walk_length=args.walk_length
hidden_size=args.hidden_size
name =args.data_name
# start, end = 0, 1
mode = args.model_mode
save_file_name = "result_for_"+name
# splits_file_path = "geom-gcn/splits/"
paths_root="preprocess/"

# Seed
# random_seed = 1
# random.seed(random_seed)
# torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

# Compute Homophily

device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')


def load_data_ranked(name):
    '''
    Load data for Cora, Cornell, Pubmed and Citeseer
    '''
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
    '''
    work for "get_whole_mask"
    '''
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
    '''
    work for "load_data", random_spilt at [48, 32, 20] ratio
    '''
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
    '''
    Load data for Nba, Electronics, Bgp
    '''
    numpy_x = np.load("./other_data"+'/'+dataset_name+'/x.npy')
    x = torch.from_numpy(numpy_x).to(torch.float)
    numpy_y = np.load("./other_data"+'/'+dataset_name+'/y.npy')
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
    '''
    Simple implementation of PAGG
    '''

    def __init__(self, feature_length, hidden_size, out_size, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super(PAGG, self).__init__()
        self.feature_length, self.hidden_size, self.out_size \
            = feature_length, hidden_size, out_size

        self.fc0 = torch.nn.Linear(feature_length, hidden_size)
        self.RNN = nn.RNN(hidden_size, hidden_size)
        self.fc2 = torch.nn.Linear(2*hidden_size, out_size)
        self.nei0 = torch.nn.Linear(hidden_size, hidden_size)  # for root node
        self.nei1 = torch.nn.Linear(hidden_size, hidden_size)
        self.nei2 = torch.nn.Linear(hidden_size, hidden_size)
        # for 3-rd order neighbor
        self.nei3 = torch.nn.Linear(hidden_size, hidden_size)
        self.nei4 = torch.nn.Linear(hidden_size, hidden_size)

        torch.nn.init.xavier_uniform_(self.fc0.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.nei0.weight)
        torch.nn.init.xavier_uniform_(self.nei1.weight)
        torch.nn.init.xavier_uniform_(self.nei2.weight)
        torch.nn.init.xavier_uniform_(self.nei3.weight)

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
        nei = F.dropout(nei, p=dropout, training=self.training)
        nei, h_n = self.RNN(nei)
        h_n = h_n.transpose(0, 1).view(
            split, num_w, -1)  # [V, num_of_walks, H]

        h_n = torch.mean(h_n, dim=1)
        ego = X[indices]
        layer1 = torch.cat((ego, h_n), dim=1)  # [V, 2*H]
        # layer1 = F.relu(self.fc2(layer1))
        layer1 = F.dropout(layer1, p=dropout, training=self.training)
        dout = self.fc2(layer1)
        return dout

# def RNNReLUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
#     hy = F.relu(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh, b_hh))
#     return hy


def train_fixed_indices(X, Y, num_classes, mode, data_name, train_indices, val_indices, test_indices, num_w, hid_size, walk_len, walks, path_type_all):
    feature_length = X.shape[-1]
    node_num = Y.shape[0]
    # Construct the model
    if mode == 'pagg':
        predictor = PAGG(feature_length, hid_size, num_classes).to(device)
    # elif mode == 'mlp':
    #     predictor = MLP(feature_length, hid_size, num_classes).to(device)
    # elif mode == 'gat':
    #     predictor = GAT(feature_length, hid_size, num_classes).to(device)

    optimizer = torch.optim.Adam(
        predictor.parameters(), lr=lr, weight_decay=weight_decay)
    lossfunc = torch.nn.CrossEntropyLoss()

    # prep data
    X = X.to(device)
    # Y = Y.to(device)

    # Start training
    test_1f1, test_2f1, test_rec, test_prec, test_acc = 0, 0, 0, 0, 0
    max_val_acc = 0
    val_acc = 0  # validation
    train_bar = tqdm.tqdm(range(epochs), dynamic_ncols=True, unit='step')

    if name in ['Cora', 'Citeseer', 'Cornell', "Nba"]:  # nomarl datasets
        neis_all = torch.tensor(walks, dtype=torch.long).view(
            1000, node_num, -1)
        path_type_all = torch.tensor(path_type_all, dtype=torch.long).view(
            1000, node_num, num_w, walk_len)

    for epoch in train_bar:
        # time1 = time.time()
        if name in ['Bgp', 'Electronics', 'Pubmed']:   # large datasets
            walks = []
            path_type = []
            path_file = paths_root+"{}_{}_{}_{:04d}.txt".format(
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

        elif name in ['Cora', 'Citeseer', 'Citeseer', "Nba"]:
            neis = neis_all[epoch]
            path_type = path_type_all[epoch]
        predictor.train()

        indxx = torch.arange(
            sum(train_indices)*num_w*walk_len, dtype=torch.long, device=device)
        # time2 = time.time()
        y_hat = predictor(X, neis[train_indices],
                          num_w, walk_len, train_indices, path_type[train_indices], indxx)  # transductive!! path_type[train_indices]
        loss = lossfunc(y_hat, Y[train_indices].to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # time3 = time.time()
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
            # time4 = time.time()
            if max_val_acc < val_acc:
                # if val_acc > max_val_acc:
                max_val_acc = val_acc
                # max_val_2f1 = val_2f1
                # print("Save Model.")
                torch.save(predictor.state_dict(),
                           "saved_models/" + save_file_name + ".pth")
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
            # time5 = time.time()
            #     test_acc += test_tmp
            # test_acc /= test_rw_round
            train_bar.set_postfix(
                data=data_name, val_acc=val_acc, test_acc=test_acc, test_2f1=test_2f1, test_1f1=test_1f1)
        # print(
        #     f"before train: {time2-time1}; train: {time3-time2}; val: {time4-time3}; test: {time5-time4}")
        # gc.collect()
    return test_1f1, test_2f1, test_rec, test_prec, test_acc  # val_acc is a list


file = open("results/" + save_file_name + ".txt", "a")
print(name)
walks = []
path_type = []
if name in ['Cora', 'Citeseer', 'Citeseer', "Nba"]:
    path_file = paths_root+"{}_{}_{}_m.txt".format(
        name, num_of_walks, walk_length)

    with open(path_file, "r") as p:
        for line in tqdm.tqdm(p):
            info = list(map(int, line[1:-2].split(",")))
            walks.append(info[:walk_length])
            path_type.append(info[walk_length:])
    print(len(walks), len(path_type))

avg_test_1f1, avg_test_2f1, avg_test_rec, avg_test_prec, avg_test_acc, \
    std_test_1f1, std_test_2f1, std_test_rec, std_test_prec, std_test_acc = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
test_1f1s, test_2f1s, test_recs, test_precs, test_accs = [], [], [], [], []

if name not in ['Bgp', 'Nba', 'Electronics']:
    (X, Y, num_classes, datasets) = load_data_ranked(name)

for i in range(rounds):
    print('round', i)
    if name in ['Bgp', 'Nba', 'Electronics']:
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
        X, Y, num_classes, mode, name, train_mask, val_mask, test_mask, num_of_walks, hidden_size, walk_length, walks, path_type)
    test_recs.append(test_rec)
    test_accs.append(test_acc)
    test_1f1s.append(test_1f1)
    test_2f1s.append(test_2f1)
    test_precs.append(test_prec)

avg_test_rec = sum(test_recs) / rounds
avg_test_acc = sum(test_accs) / rounds
avg_test_1f1 = sum(test_1f1s) / rounds
avg_test_2f1 = sum(test_2f1s) / rounds
avg_test_prec = sum(test_precs) / rounds

std_test_rec = np.std(np.array(test_recs))
std_test_acc = np.std(np.array(test_accs))
std_test_1f1 = np.std(np.array(test_1f1s))
std_test_2f1 = np.std(np.array(test_2f1s))
std_test_prec = np.std(np.array(test_precs))

print(name+"_"+str(num_of_walks)+"_" +
        str(walk_length)+'_'+str(hidden_size)+"\n")
print(mode+" Avg for {}: acc{:.4f} ± {:.4f}\t prec{:.4f} ± {:.4f}\t rec{:.4f} ± {:.4f}\t maf1{:.4f} ± {:.4f}\t mif1{:.4f} ± {:.4f}\t ".format(
    name, avg_test_acc, std_test_acc, avg_test_prec, std_test_prec, avg_test_rec, std_test_rec, avg_test_1f1, std_test_1f1, avg_test_2f1, std_test_2f1))
print(name+"_"+str(num_of_walks)+"_"+str(walk_length
                                            )+'_'+str(hidden_size)+"\n", file=file)
print(mode+" Avg for {}: acc{:.4f} ± {:.4f}\t prec{:.4f} ± {:.4f}\t rec{:.4f} ± {:.4f}\t maf1{:.4f} ± {:.4f}\t mif1{:.4f} ± {:.4f}\t ".format(
    name, avg_test_acc, std_test_acc, avg_test_prec, std_test_prec, avg_test_rec, std_test_rec, avg_test_1f1, std_test_1f1, avg_test_2f1, std_test_2f1), file=file)
