import math
import torch
from torch_geometric.utils import to_undirected
import numpy as np
from scipy.io import mmread
from torch_geometric.data import Data
import scipy.sparse

def load_pytg(link, node):
    left, right, weight = [], [], []
    with open(link, 'r') as file:
        for line in file:
            line = np.array(line[:-1].split('\t'))
            left.append(line[0].astype(int))
            right.append(line[1].astype(int))
            if len(line)>2:
                weight.append(line[2].astype(float))

    edge_index = torch.tensor([left, right], dtype=torch.long)
    node_attri = {}
    with open(node, 'r') as file:
        for line in file:
            line = line[:-1].split('\t')
            tmp = line[3].split(',')
            tmp.append(line[2])
            node_attri[int(line[0])] = np.array(tmp).astype(np.float32)
    data_x = torch.tensor([node_attri[k] for k in range(len(node_attri))], dtype=torch.float)
    data = Data(x=data_x, edge_index=edge_index)
    if len(weight)>2:
        data.edge_wt = torch.tensor(weight, dtype=torch.float)
    return data

def out_of_sample_test(data, link_val, link_test, neg_mask_file):
    assert 'batch' not in data  # No batch-mode.

    # validation edges
    left_n, right_n, left_p, right_p = [], [], [], []
    with open(link_val, 'r') as file:
        for line in file:
            line = np.array(line[:-1].split('\t'))
            if(line[2]=='0'):
                left_n.append(line[0].astype(int))
                right_n.append(line[1].astype(int))
            if(line[2]=='1'):
                left_p.append(line[0].astype(int))
                right_p.append(line[1].astype(int))

    data.val_pos_edge_index = torch.tensor([left_p, right_p], dtype=torch.long)
    data.val_neg_edge_index = torch.tensor([left_n, right_n], dtype=torch.long)
    # test edges
    left_n, right_n, left_p, right_p = [], [], [], []
    with open(link_test, 'r') as file:
        for line in file:
            line = np.array(line[:-1].split('\t'))
            if(line[2]=='0'):
                left_n.append(line[0].astype(int))
                right_n.append(line[1].astype(int))
            if(line[2]=='1'):
                left_p.append(line[0].astype(int))
                right_p.append(line[1].astype(int))

    data.test_pos_edge_index = torch.tensor([left_p, right_p], dtype=torch.long)
    data.test_neg_edge_index = torch.tensor([left_n, right_n], dtype=torch.long)

    neg_mask = mmread(neg_mask_file)
    values = neg_mask.data
    indices = np.vstack((neg_mask.row, neg_mask.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = neg_mask.shape
    neg_adj_mask = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
    data.train_neg_adj_mask = neg_adj_mask

    return data

def out_of_sample_data(data, neg_mask_file, setting):
    neg_mask = mmread(neg_mask_file)
    values = neg_mask.data
    indices = np.vstack((neg_mask.row, neg_mask.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = neg_mask.shape
    neg_adj_mask = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

    # validation edges
    data.val_pos_edge_index = torch.empty(2, 0, dtype=torch.long)
    left_n, right_n = torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    for pt in data.val_prot:
        # getting positive and negative edges
        tmp = (data.edge_index[0,:] == pt).nonzero(as_tuple=True)[0]
        data.val_pos_edge_index = torch.cat((data.val_pos_edge_index, data.edge_index[:,tmp]), 1)
        tmp2 = neg_adj_mask[pt,:].nonzero(as_tuple=True)[0]
        # subsample negative edges
        if setting=='transfer':
            perm = torch.randperm(tmp2.size(0))[:min(tmp.size(0), tmp2.size(0))]
            tmp2 = tmp2[perm]
        right_n = torch.cat((right_n, tmp2), 0)
        left_n = torch.cat(((left_n, pt * torch.ones(tmp2.size(0), dtype=torch.long))), 0)
        # removing added edges from data
        tmp1 = data.edge_index[0,:] != pt
        data.edge_index = data.edge_index[:,tmp1]
        neg_adj_mask[pt,:] = neg_adj_mask[:,pt] = 0

    data.val_neg_edge_index = torch.stack([left_n, right_n])

    # test edges
    if setting=='induc':   # check for transfer learning case
        data.test_pos_edge_index = torch.empty(2, 0, dtype=torch.long)
        left_n, right_n = torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        for pt in data.test_prot:
            # getting positive and negative edges
            tmp = (data.edge_index[0,:] == pt).nonzero(as_tuple=True)[0]
            data.test_pos_edge_index = torch.cat((data.test_pos_edge_index, data.edge_index[:,tmp]), 1)
            tmp2 = neg_adj_mask[pt,:].nonzero(as_tuple=True)[0]
            right_n = torch.cat((right_n, tmp2), 0)
            left_n = torch.cat(((left_n, pt * torch.ones(tmp2.size(0), dtype=torch.long))), 0)
            # removing added edges from data
            tmp1 = data.edge_index[0,:] != pt
            data.edge_index = data.edge_index[:,tmp1]
            neg_adj_mask[pt,:] = neg_adj_mask[:,pt] = 0

        data.test_neg_edge_index = torch.stack([left_n, right_n])

    # remove unconnected nodes
    rna_train = set(data.edge_index[1,:].tolist())
    rna_all = set((data.x[:,100] == 0).nonzero(as_tuple=True)[0].tolist())
    rna_rem = rna_all - rna_train

    tmp = [i for i, e in enumerate(data.val_pos_edge_index[1,:].tolist()) if e in rna_rem]
    tmp1 = [True] * data.val_pos_edge_index.size(1)
    for i in tmp:
        tmp1[i] = False
    data.val_pos_edge_index = data.val_pos_edge_index[:,tmp1]
    tmp = [i for i, e in enumerate(data.val_neg_edge_index[1,:].tolist()) if e in rna_rem]
    tmp1 = [True] * data.val_neg_edge_index.size(1)
    for i in tmp:
        tmp1[i] = False
    data.val_neg_edge_index = data.val_neg_edge_index[:,tmp1]

    if 'data.test_prot' in locals():
        tmp = [i for i, e in enumerate(data.test_pos_edge_index[1,:].tolist()) if e in rna_rem]
        tmp1 = [True] * data.test_pos_edge_index.size(1)
        for i in tmp:
            tmp1[i] = False
        data.test_pos_edge_index = data.test_pos_edge_index[:,tmp1]
        tmp = [i for i, e in enumerate(data.test_neg_edge_index[1,:].tolist()) if e in rna_rem]
        tmp1 = [True] * data.test_neg_edge_index.size(1)
        for i in tmp:
            tmp1[i] = False
        data.test_neg_edge_index = data.test_neg_edge_index[:,tmp1]

    neg_adj_mask[list(rna_rem),:] = neg_adj_mask[:,list(rna_rem)] = 0
    data.train_neg_adj_mask = neg_adj_mask
    data.train_pos_edge_index = data.edge_index

    return data

def my_neg_sampling(edge_index, num_nodes, neg_mask, negative_rate):
    r"""Samples random negative edges of a graph given using a negative adjacency mask"""
    num_neg_samples = int(negative_rate * edge_index.size(1))

    # Handle '|V|^2 - |E| < |E|'.
    size = num_nodes * num_nodes
    num_neg_samples = min(num_neg_samples, size - edge_index.size(1))

    row, col = edge_index

    neg_row, neg_col = neg_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:num_neg_samples]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    return torch.stack([neg_row, neg_col], dim=0)

def sample_lt_edges(data, num_samp):
    r"""Sample low throughput edges for train and validation proteins"""
    lt_edges = torch.empty(2, 0, dtype=torch.long)
    for pt in data.val_prot:
        idx = (data.val_pos_edge_index[0,:] == pt).nonzero(as_tuple=True)[0]
        lt_edges = torch.cat((lt_edges, data.val_pos_edge_index[:,idx[torch.multinomial(torch.ones(idx.size(0), dtype=torch.float), num_samp)]]), 1)

    for pt in data.test_prot:
        idx = (data.test_pos_edge_index[0,:] == pt).nonzero(as_tuple=True)[0]
        lt_edges = torch.cat((lt_edges, data.test_pos_edge_index[:,idx[torch.multinomial(torch.ones(idx.size(0), dtype=torch.float), num_samp)]]), 1)

    neg_adj_mask[list(rna_rem),:] = neg_adj_mask[:,list(rna_rem)] = 0
    data.train_neg_adj_mask = neg_adj_mask
    data.lt_pos_edge_index = torch.cat((data.edge_index, lt_edges), 1)
    data.lt_pos_edge_index = to_undirected(data.lt_pos_edge_index)

    return data

def my_structured_negative_sampling(data):
    r"""Samples a negative edge :obj:`(i,k)` for every positive edge
    :obj:`(i,j)` in the graph given by :attr:`data.edge_index`.
    """
    row, col = data.edge_index
    neg_mask = data.train_neg_adj_mask
    prots = data.prot_in_net
    prot_id = prots.nonzero(as_tuple=False).t()[0,]
    deg_prot = data.deg_prot
    right_n = torch.empty(0, dtype=torch.long).to('cpu')

    for pt in range(sum(prots)):
        neg_col = neg_mask[prot_id[pt],:].nonzero(as_tuple=False).t()[0,].to('cpu')
        weights = torch.ones_like(neg_col, dtype=torch.float)
        right_n = torch.cat((right_n, neg_col[torch.multinomial(weights, deg_prot[pt], replacement=True)]), -1)

    return torch.stack([row.to('cpu'), right_n], dim=0)

def eval_hits(y_pred_pos, y_pred_neg, K):
    '''
        compute Hits@K
        For each positive target node, the negative target nodes are the same.
        y_pred_neg is an array.
        rank y_pred_pos[i] against y_pred_neg for each i
    '''

    if len(y_pred_neg) < K:
        return 1

    kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
    hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)

    return hitsK
