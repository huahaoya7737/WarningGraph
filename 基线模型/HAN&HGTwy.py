

import os
import torch
import pickle
import numpy as np
import copy
import os.path as osp
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch import nn
from model.model_han import HANConv
from utils.loader import DNS
from typing import Dict, List, Union
from torch_geometric.datasets import IMDB,DBLP
from torch_geometric.nn import HGTConv
from utils.util import plot_confusion_matrix,attack_graph_feature
from sklearn.metrics import f1_score,classification_report ,confusion_matrix
from utils.util import MIEstimator,Attention,attack_graph,get_split,subgraph_sampling,plot_confusion_matrix,attack_graph_feature,attack_label,attack_graph2,attack_graph3,attack_graph4
from utils.PyGCL.GCL.eval.logistic_regression import LREvaluator


device = torch.device('cuda')

with open('/data/sda/haoqingyu/DNS/data_hash_multi.pickle','rb') as f:
        data = pickle.load(f)
print(data)
data_orignal = copy.deepcopy(data)
data_orignal = data_orignal.to(device)

# #节点扰动
#data = attack_graph_feature(data,0.3)
#边扰动
# data = attack_graph(data,0.6)
# data = attack_graph2(data,0.6)
# data = attack_graph3(data,0.6)
# data = attack_graph4(data,0.6)

id = (data['Hash'].train_mask | data['Hash'].test_mask | data['Hash'].val_mask).detach().cpu()



def mask(data, node_type, tr=0.2, vr=0.1):

    num_test = tr
    num_val = vr
    n_nodes = len(data[node_type].x)
    perm = torch.randperm(n_nodes)
    test_idx = perm[:int(n_nodes * num_test)]
    val_idx = perm[int(n_nodes * num_test):int(n_nodes * (num_test + num_val))]
    train_idx = perm[int(n_nodes * (num_test + num_val)):]
    for v, idx in [('train', train_idx), ('test', test_idx), ('val', val_idx)]:
        mask = torch.zeros(n_nodes, dtype=torch.bool)
        mask[idx] = True
        data[node_type][f'{v}_mask'] = mask
    
    return data

train_r = 0.4
test = 0.9 - train_r
val = 0.1

# data = mask(data, 'Hash', test, val)


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

class HAN(nn.Module):
    def __init__(self, in_channels: Union[int, Dict[str, int]],
                 out_channels: int, hidden_channels=256, heads=2):
        super().__init__()
        self.han_conv = HANConv(in_channels, hidden_channels, heads=heads,
                                dropout=0.1, metadata=data.metadata())
        self.han_conv_2 = HANConv(hidden_channels, hidden_channels, heads=heads,
                                dropout=0.1, metadata=data.metadata())
        self.lin = nn.Linear(hidden_channels, 20)
        self.lin_out = nn.Linear(32, out_channels)

    def forward(self, x_dict, edge_index_dict):
        out = self.han_conv(x_dict, edge_index_dict)
        out = self.han_conv_2(out, edge_index_dict)
        out = self.lin(out['Hash'])
        return out, F.log_softmax(out)


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:   
            self.lin_dict[node_type] = nn.Linear(data[node_type].x.size(1), hidden_channels)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(), num_heads, group='sum')
            self.convs.append(conv)
        self.lin = nn.Linear(hidden_channels, 32)
        self.lin_out = nn.Linear(32, out_channels)

    def forward(self, x_dict, edge_index_dict):

        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        z = x_dict['Hash']
        out = self.lin(z)
        # return out, F.log_softmax(self.lin_out(out))
        return out, F.log_softmax(out)

model = HAN(in_channels=data.num_node_features,out_channels=20)
#model = HGT(hidden_channels=256, out_channels=20,num_layers=2,num_heads=2)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data, model = data.to(device), model.to(device)

with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.001)

uniform_txt = open("han_uniform.txt","wt")
def train(epoch) -> float:
    model.train()
    optimizer.zero_grad()
    _, out = model(data.x_dict, data.edge_index_dict)
    mask = data['Hash'].train_mask
    if(epoch % 10000 == 0):
        uniform_l = uniform_loss(_).item()
        print(uniform_l)
        uniform_txt.write(str(uniform_l)+"\n")
    y = data['Hash'].y[mask]
    y = torch.tensor(y,dtype = torch.int64)
    loss = F.cross_entropy(out[mask], y)
    loss.backward()
    optimizer.step()
    return float(loss),_


@torch.no_grad()
def test() -> List[float]:
    model.eval()
    _,out = model(data.x_dict, data.edge_index_dict)
    pred = out.argmax(dim=-1)
    accs = []
    result = {}
    test_mask = data['Hash'].test_mask
    pred = pred[test_mask].cpu().detach().numpy()
    test_micro = f1_score(data['Hash'].y[test_mask].cpu().detach().numpy(), pred,average='micro')
    cr = classification_report(data['Hash'].y[test_mask].cpu().detach().numpy(), pred,digits=4)
    cm = confusion_matrix(data['Hash'].y[test_mask].cpu(), pred)
    result['micro_f1'] = test_micro
    result['cr'] = cr
    result['cm'] = cm
    return result

best_val_acc = 0
best_test_acc = 0
best_cr = 0
best_f1 = 0
best_cm = 0
max_f1 = 0
best_cr = 0
best_cm = np.array([])
start_patience = patience = 200
log = open("han_log.csv",'wt')
res = open("han_res.csv","wt")
for epoch in range(1, 501):
    print('开始训练：')
    loss,_ = train(epoch)
    if(epoch % 10 == 0):
        test_result = test()
        print(test_result['micro_f1'])
        if(test_result['micro_f1'] > max_f1):
            max_f1 = test_result['micro_f1']
            if(test_result['micro_f1'] > max_f1):
                max_f1 = test_result['micro_f1']
                best_cr= test_result['cr']
                best_cm = test_result['cm']
                print(max_f1)
                print(best_cr)
print("best f1 socre:", max_f1)
print(test_result['cr'])

