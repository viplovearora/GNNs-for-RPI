import argparse
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from random import random

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--weighted', action='store_true', help='use a weighted network')
parser.add_argument('--struct_neg', action='store_true', help='use structured negative sampling instead of uniformly at random')
parser.add_argument('--rna_seq', action='store_true', help='append RNA-seq to GNN embedding before prediction')
parser.add_argument('--rank_loss', action='store_true', help='use a ranking-based loss function, will only work with structured negative sampling')
parser.add_argument('--easy', action='store_true', help='easy test negative edges')
parser.add_argument('--embed_size', type=int, default=10)
args = parser.parse_args()

link = 'Data/K562_PCA/link.dat'
node = 'Data/K562_PCA/node.dat'
if args.rna_seq:
    node = 'Data/K562_PCA/node_seq.dat'
data = load_pytg(link, node)
test = 'Data/K562_PCA/link.dat.test'
neg_mask_file = 'Data/K562_PCA/neg_mask.mtx'
data = predefined_test(data, test, neg_mask_file)
if args.rna_seq:
    data.TPM = data.x[:,100].unsqueeze(1)
    data.x = torch.cat((data.x[:,:100], data.x[:,101:]), 1)
best_mod = 'RBP_lp/best_model' + str(random()) + '.pt'
print(data)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, 50)
        self.conv2 = GCNConv(50, args.embed_size)
#
    def encode(self):
        if args.weighted:
            x = self.conv1(data.x, data.train_pos_edge_index, data.edge_wt)
            x = x.relu()
            return self.conv2(x, data.train_pos_edge_index, data.edge_wt)
        else:
            x = self.conv1(data.x, data.train_pos_edge_index)
            x = x.relu()
            return self.conv2(x, data.train_pos_edge_index)
#
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
# print(model)

def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def train():
    model.train()
#
    if args.struct_neg:
        neg_edge_index = my_structured_negative_sampling(data).to(device)
    else:
        neg_edge_index = my_neg_sampling(edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes, neg_mask=data.train_neg_adj_mask, negative_rate=2)
#
    optimizer.zero_grad()
    z = model.encode()
    if args.rna_seq:
        z = torch.cat((z, data.TPM), 1)
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
#
    return loss


@torch.no_grad()
def test():
    model.eval()
    perfs = []
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
#
        z = model.encode()
        if args.rna_seq:
            z = torch.cat((z, data.TPM), 1)
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
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

model_best = Net()
model_best.load_state_dict(torch.load(best_mod))
model_best.to(device)
model_best.eval()
z = model_best.encode()
if args.rna_seq:
    z = torch.cat((z, data.TPM), 1)
link_logits = model_best.decode(z, data.test_pos_edge_index, data.test_neg_edge_index)
link_probs = link_logits.sigmoid()
link_labels = get_link_labels(data.test_pos_edge_index, data.test_neg_edge_index)

from sklearn.metrics import average_precision_score
auc = roc_auc_score(link_labels.cpu().detach(), link_probs.cpu().detach())
ap = average_precision_score(link_labels.cpu().detach(), link_probs.cpu().detach())
logits_pos = model_best.decode_rl(z, data.test_pos_edge_index)
logits_neg = model_best.decode_rl(z, data.test_neg_edge_index)
y_pred_pos = logits_pos.sigmoid()
y_pred_neg = logits_neg.sigmoid()
hits50 = eval_hits(y_pred_pos, y_pred_neg, 50)
hits10 = eval_hits(y_pred_pos, y_pred_neg, 10)
hits100 = eval_hits(y_pred_pos, y_pred_neg, 100)
res = [data.test_pos_edge_index.size(1), args.weighted, args.rna_seq, args.struct_neg, args.rank_loss, auc, ap, hits10, hits50, hits100]
# print(res)

import csv
with open('results/res_K562_trans.csv', 'a', encoding='UTF8', newline='') as fd:
    writer = csv.writer(fd)
    writer.writerow(res)
