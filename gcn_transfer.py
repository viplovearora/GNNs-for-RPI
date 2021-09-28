import argparse
import torch
import math
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from random import random

from utils_induc import *
parser = argparse.ArgumentParser()
parser.add_argument('--val_per', type=int, default=10, help='what percentage of the proteins to add to the validation set')
parser.add_argument('--cos', action='store_true', help='cosine similarity or inverse distance-based similarity')
parser.add_argument('--struct_neg', action='store_true', help='use structured negative sampling instead of uniformly at random')
parser.add_argument('--rna_seq', action='store_true', help='append RNA-seq to GNN embedding before prediction')
parser.add_argument('--rank_loss', action='store_true', help='use a ranking-based loss function, will only work with structured negative sampling')
parser.add_argument('--use_ap', action='store_true', help='use average precision for model selection')
args = parser.parse_args()

link = 'Data/HepG2_PCA/all_links.dat'
node = 'Data/HepG2_PCA/node.dat'
if args.rna_seq:
    node = 'Data/HepG2_PCA/node_seq.dat'
data = load_pytg(link, node)
if args.rna_seq:
    data.TPM_val = data.x[:,100].unsqueeze(1)
    data.x = torch.cat((data.x[:,:100], data.x[:,101:]), 1)
prot_id = data.x[:,100] == 1
prots = prot_id.nonzero()[:,0]
num_prots = prots.size(0)
val_prot_num = torch.multinomial(torch.ones(num_prots, dtype=torch.float), math.ceil(num_prots*args.val_per/100))
val_prot = prots[val_prot_num]  # select val_per % of preoteins randomly
neg_mask_file = 'Data/HepG2_PCA/neg_mask.mtx'
data.val_prot = val_prot
data = out_of_sample_data(data, neg_mask_file, setting='transfer')
best_mod = 'best_model' + str(random()) + '.pt'

link_transfer = 'Data/K562_transfer/link.dat'
node_transfer = 'Data/K562_transfer/node.dat'
if args.rna_seq:
    node_transfer = 'Data/K562_transfer/node_seq.dat'
data_transfer = load_pytg(link_transfer, node_transfer)
if args.rna_seq:
    data_transfer.TPM = data_transfer.x[:,100].unsqueeze(1)
    data_transfer.x = torch.cat((data_transfer.x[:,:100], data_transfer.x[:,101:]), 1)
    data.TPM_test = data_transfer.TPM
test_prot = (data_transfer.x[:,100] == 1).nonzero().tolist()

neg_mask_transfer = 'Data/K562_transfer/neg_mask.mtx'

def read_neg_mask(neg_mask_file):
    neg_mask = mmread(neg_mask_file)
    values = neg_mask.data
    indices = np.vstack((neg_mask.row, neg_mask.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = neg_mask.shape
    neg_adj_mask = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
    return neg_adj_mask

data_transfer.neg_adj_mask = read_neg_mask(neg_mask_transfer)

data.test_pos_edge_index = data_transfer.edge_index
data.test_neg_edge_index = my_neg_sampling(edge_index=data_transfer.edge_index, num_nodes=data_transfer.num_nodes, neg_mask=data_transfer.neg_adj_mask, negative_rate=1)

cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
pdist = torch.nn.PairwiseDistance(p=2)

if args.cos:
    wt_test_prot = torch.empty(len(test_prot), num_prots)
    for pt in range(len(test_prot)):
        a = cos(data_transfer.x[test_prot[pt],:], data.x[prot_id,:])
        a[val_prot_num] = 0      # setting 0 for val prots
        a[a<0] = 0
        wt_test_prot[pt,:] = F.normalize(a, p=1, dim=0)

    wt_val_prot = torch.empty(len(val_prot), num_prots)
    for pt in range(len(val_prot)):
        a = cos(data.x[val_prot[pt],:], data.x[prot_id,:])
        a[val_prot_num] = 0      # setting 0 for val prots
        a[a<0] = 0
        wt_val_prot[pt,:] = F.normalize(a, p=1, dim=0)
else:
    wt_test_prot = torch.empty(len(test_prot), num_prots)
    for pt in range(len(test_prot)):
        a = 1/pdist(data_transfer.x[test_prot[pt],:], data.x[prot_id,:])
        a[val_prot_num] = 0      # setting 0 for val prots
        wt_test_prot[pt,:] = F.normalize(a, p=1, dim=0)

    wt_val_prot = torch.empty(len(val_prot), num_prots)
    for pt in range(len(val_prot)):
        a = 1/pdist(data.x[val_prot[pt],:], data.x[prot_id,:])
        a[val_prot_num] = 0      # setting 0 for val prots
        wt_val_prot[pt,:] = F.normalize(a, p=1, dim=0)

data.wt_val_prot = wt_val_prot
data.test_prot = test_prot
data.wt_test_prot = wt_test_prot
data.prot_id_val = prot_id
data.prot_id_test = torch.cat((prot_id, torch.zeros(47)==1), 0)

if args.struct_neg:
    row, col = data.edge_index
    prot_in_net = data.x[:,100]==1
    prot_in_net[data.val_prot] = False
    data.prot_in_net = prot_in_net
    prot_in_net_id = prot_in_net.nonzero(as_tuple=False).t()[0,]
    data.deg_prot = torch.bincount(row)[prot_in_net_id]
print(data)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, 50)
        self.conv2 = GCNConv(50, 10)

    def encode(self):
        x = self.conv1(data.x, data.train_pos_edge_index)
        x = x.relu()
        return self.conv2(x, data.train_pos_edge_index)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_rl(self, z, edge_index):
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.015)

def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def train():
    model.train()

    if args.struct_neg:
        neg_edge_index = my_structured_negative_sampling(data).to(device)
    else:
        neg_edge_index = my_neg_sampling(edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes, neg_mask=data.train_neg_adj_mask, negative_rate=2)

    optimizer.zero_grad()
    z = model.encode()
    if args.rna_seq:
        z = torch.cat((z, data.TPM_val), 1)
    if args.rank_loss:
        logits_pos = model.decode_rl(z, data.train_pos_edge_index)
        logits_neg = model.decode_rl(z, neg_edge_index)
        target = torch.ones(neg_edge_index.size(1))
        loss_fn = torch.nn.MarginRankingLoss(margin=2.5)
        loss = loss_fn(logits_pos, logits_neg, target.to(device))
    else:
        link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
        link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
        loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def test():
    model.eval()
    perfs = []
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        prots = data[f'{prefix}_prot']
        wt_prots = data[f'wt_{prefix}_prot']

        z = model.encode()
        if prefix == "test":
            z = torch.cat((z, torch.empty(47, z.size(1)).to(device)), 0)
        for pt in range(len(prots)):
            z[prots[pt],:] = torch.matmul(torch.transpose(z[data[f'prot_id_{prefix}'],:], 0, 1), wt_prots[pt,:])
        if args.rna_seq:
            z = torch.cat((z, data[f'TPM_{prefix}']), 1)
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        if args.use_ap:
            perfs.append(average_precision_score(link_labels.cpu(), link_probs.cpu()))
        else:
            perfs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
    return perfs

best_val_perf = test_perf = 0
for epoch in range(1, 1001):
    train_loss = train()
    val_perf, tmp_test_perf = test()
    if val_perf > best_val_perf:
        best_val_perf = val_perf
        test_perf = tmp_test_perf
        torch.save(model.state_dict(), best_mod)
        log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_loss, best_val_perf, test_perf))

model = Net()
model.load_state_dict(torch.load(best_mod))
model.to(device)
model.eval()
z = model.encode()
z = torch.cat((z, torch.empty(47, z.size(1)).to(device)), 0)
prots = data.test_prot
wt_prots = data.wt_test_prot
for pt in range(len(prots)):
    z[prots[pt],:] = torch.matmul(torch.transpose(z[data.prot_id_test,:], 0, 1), wt_prots[pt,:])
if args.rna_seq:
    z = torch.cat((z, data.TPM_test), 1)
link_logits = model.decode(z, data.test_pos_edge_index, data.test_neg_edge_index)
link_probs = link_logits.sigmoid()
link_labels = get_link_labels(data.test_pos_edge_index, data.test_neg_edge_index)

auc = roc_auc_score(link_labels.cpu().detach(), link_probs.cpu().detach())
ap = average_precision_score(link_labels.cpu().detach(), link_probs.cpu().detach())
logits_pos = model.decode_rl(z, data.test_pos_edge_index)
logits_neg = model.decode_rl(z, data.test_neg_edge_index)
y_pred_pos = logits_pos.sigmoid()
y_pred_neg = logits_neg.sigmoid()
hits50 = eval_hits(y_pred_pos, y_pred_neg, 50)
hits10 = eval_hits(y_pred_pos, y_pred_neg, 10)
hits100 = eval_hits(y_pred_pos, y_pred_neg, 100)
res = [args.val_per, args.cos, args.rna_seq, args.struct_neg, args.rank_loss, auc, ap, hits10, hits50, hits100]
# print(res)

import csv
if args.use_ap:
    with open('results/res_HepG2_transfer_ap.csv', 'a', encoding='UTF8', newline='') as fd:
        writer = csv.writer(fd)
        writer.writerow(res)
else:
    with open('results/res_HepG2_transfer.csv', 'a', encoding='UTF8', newline='') as fd:
        writer = csv.writer(fd)
        writer.writerow(res)
