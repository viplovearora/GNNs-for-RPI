import math
import torch
from torch_geometric.utils import to_undirected
import numpy as np
from scipy.io import mmread
from torch_geometric.data import Data
import pandas as pd

def predefined_test(data, test, neg_mask_file, val_ratio=0.09):
    assert 'batch' not in data  # No batch-mode.

    num_nodes = data.num_nodes
    row, col = data.edge_index

    # Negative edges.
    left_n, right_n, left_p, right_p = [], [], [], []
    with open(test, 'r') as file:
        for line in file:
            line = np.array(line[:-1].split('\t'))
            if(line[2]=='0'):
                left_n.append(line[0].astype(int))
                right_n.append(line[1].astype(int))
            if(line[2]=='1'):
                left_p.append(line[0].astype(int))
                right_p.append(line[1].astype(int))

    left_n =  torch.tensor(left_n)
    right_n = torch.tensor(right_n)
    left_p =  torch.tensor(left_p)
    right_p = torch.tensor(right_p)
    neg_mask = mmread(neg_mask_file)
    values = neg_mask.data
    indices = np.vstack((neg_mask.row, neg_mask.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = neg_mask.shape

    neg_adj_mask = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

    perm = torch.randperm(left_n.size(0))
    data.train_neg_adj_mask = neg_adj_mask
    n_v = int(math.floor(val_ratio * (data.train_pos_edge_index.size(1) + left_p.size(0))))

    row, col = left_n[:n_v], right_n[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)
    row, col = left_n[n_v:], right_n[n_v:]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)
    row, col = left_p[:n_v], right_p[:n_v]
    data.val_pos_edge_index = torch.stack([row, col], dim=0)
    row, col = left_p[n_v:], right_p[n_v:]
    data.test_pos_edge_index = torch.stack([row, col], dim=0)

    return data

def load_pytg(link, node, val_ratio=0.05):
    left, right, weight = [], [], []
    with open(link, 'r') as file:
        for line in file:
            line = np.array(line[:-1].split('\t'))
            left.append(line[0].astype(int))
            right.append(line[1].astype(int))
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
    data.edge_wt = torch.tensor(weight, dtype=torch.float)
    data.train_pos_edge_index = torch.tensor([left, right], dtype=torch.long)
    return data

def load_edges(link, node, val_ratio=0.05):
    left, right, weight = [], [], []
    with open(link, 'r') as file:
        for line in file:
            line = np.array(line[:-1].split('\t'))
            left.append(line[0].astype(int))
            right.append(line[1].astype(int))
            weight.append(line[2].astype(float))

    edge_index = torch.tensor([left, right], dtype=torch.long)
    # node_attri = {}
    prot = []
    with open(node, 'r') as file:
        for line in file:
            line = line[:-1].split('\t')
            # tmp = line[3].split(',')
            prot = line[2].astype(int)
            # tmp.append(line[2])
            # node_attri[int(line[0])] = np.array(tmp).astype(np.float32)
    # data_x = torch.tensor([node_attri[k] for k in range(len(node_attri))], dtype=torch.float)
    data = Data(prot=prot, edge_index=edge_index)
    data.edge_wt = torch.tensor(weight, dtype=torch.float)
    data.train_pos_edge_index = torch.tensor([left, right], dtype=torch.long)
    # data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)
    return data

def load_features(data, feat_prot, feat_rna):
    data.x_prot = torch.tensor(pd.read_csv(feat_prot).values, dtype=torch.float)
    data.x_rna = torch.tensor(pd.read_csv(feat_rna).values, dtype=torch.float)
    return data

def my_neg_sampling(edge_index, num_nodes, neg_mask, negative_rate):
    r"""Samples random negative edges of a graph given using a negative adjacency mask
    """

    num_neg_samples = int(negative_rate * edge_index.size(1))

    # Handle '|V|^2 - |E| < |E|'.
    size = num_nodes * num_nodes
    num_neg_samples = min(num_neg_samples, size - edge_index.size(1))

    row, col = edge_index

    neg_row, neg_col = neg_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:num_neg_samples]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    return torch.stack([neg_row, neg_col], dim=0)

def my_structured_negative_sampling(data):
    r"""Samples a negative edge :obj:`(i,k)` for every positive edge
    :obj:`(i,j)` in the graph given by :attr:`data.edge_index`.
    """
    row, col = data.edge_index
    num_nodes = data.num_nodes
    neg_mask = data.train_neg_adj_mask
    prots = data.x[:,100]==1
    prot_id = prots.nonzero(as_tuple=False).t()[0,]
    deg_prot = torch.bincount(row)
    prots = prots[:deg_prot.size(0)]
    deg_prot = deg_prot[prots]
    right_n = torch.empty(0, dtype=torch.long).to('cpu')

    for pt in range(sum(prots)):
        neg_col = neg_mask[prot_id[pt],:].nonzero(as_tuple=False).t()[0,].to('cpu')
        weights = torch.ones_like(neg_col, dtype=torch.float)
        # perm = torch.randperm(neg_col.size(0))[:deg_prot[pt]]
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
        return {'hits@{}'.format(K): 1.}

    kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
    hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)

    return hitsK

def eval_mrr(y_pred_pos, y_pred_neg):   # not considering right now
    '''
        compute mrr
        y_pred_neg is an array with shape (batch size, num_entities_neg).
        y_pred_pos is an array with shape (batch size, )
    '''

    y_pred = torch.cat([y_pred_pos.view(-1,1), y_pred_neg], dim = 1)
    argsort = torch.argsort(y_pred, dim = 1, descending = True)
    ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
    ranking_list = ranking_list[:, 1] + 1
    hits1_list = (ranking_list <= 1).to(torch.float)
    hits3_list = (ranking_list <= 3).to(torch.float)
    hits10_list = (ranking_list <= 10).to(torch.float)
    mrr_list = 1./ranking_list.to(torch.float)

    return {'hits@1_list': hits1_list,
             'hits@3_list': hits3_list,
             'hits@10_list': hits10_list,
             'mrr_list': mrr_list}

def my_train_test(data, neg_mask_file, val_ratio=0.09, test_ratio=0.2):
    assert 'batch' not in data  # No batch-mode.

    num_nodes = data.num_nodes
    row, col = data.edge_index
    data.edge_index = None

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # Negative edges.
    neg_mask = mmread(neg_mask_file)
    values = neg_mask.data
    indices = np.vstack((neg_mask.row, neg_mask.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = neg_mask.shape

    neg_adj_mask = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:n_v + n_t]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data
