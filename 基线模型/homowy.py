from dataclasses import dataclass
import torch
import time
import copy
import random
import pickle
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from utils.loader import DNS
import torch.nn.functional as F
import torch_geometric.transforms as T
from utils.PyGCL.GCL.eval.logistic_regression import LREvaluator
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from sklearn.metrics import f1_score,classification_report ,confusion_matrix
from utils.util import plot_confusion_matrix,attack_label
from utils.loader import DNS, to_homogeneous
from utils.util import MIEstimator,Attention,attack_graph,get_split,subgraph_sampling,plot_confusion_matrix,attack_graph_feature,attack_label,attack_graph2,attack_graph3,attack_graph4

device = torch.device('cuda')
with open('/data/sda/haoqingyu/DNS/data_hash_multi.pickle','rb') as f:
        data = pickle.load(f)
data_orignal = copy.deepcopy(data)


#节点扰动
#data = attack_graph_feature(data,0.6)

# #边扰动
# data = attack_graph(data,0.6)
# data = attack_graph2(data,0.6)
# data = attack_graph3(data,0.6)
# data = attack_graph4(data,0.6)

data = to_homogeneous(data).to(device)
data_orignal = to_homogeneous(data_orignal).to(device)

# id = (data.train_mask | data.test_mask | data.val_mask).detach().cpu()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

class GCN(torch.nn.Module):

    def __init__(self, features, hidden, classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(features, hidden)  # shape（输入的节点特征维度 * 中间隐藏层的维度）
        self.conv2 = GCNConv(hidden, hidden)  # shaape（中间隐藏层的维度 * 节点类别）
        self.lin = torch.nn.Linear(hidden, classes)

    def forward(self, data):
        # 加载节点特征和邻接关系
        x, edge_index = data.x, data.edge_index
        
        # 传入卷积层
        x = self.conv1(x, edge_index)
        
        x = F.relu(x)  # 激活函数
        x = F.dropout(x, training=self.training)  # dropout层，防止过拟合
        x = self.conv2(x, edge_index)  # 第二层卷积层
        print("x is : ",x)
        # 将经过两层卷积得到的特征输入log_softmax函数得到概率分布
        return x , F.log_softmax(self.lin(x))
        # return x


class GraphSAGE(torch.nn.Module):

    def __init__(self, feature, hidden, classes):
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(feature, hidden)  # 定义两层GraphSAGE层
        self.sage2 = SAGEConv(hidden, classes)
        # self.sage2 = SAGEConv(hidden, 16)
        # self.lin = torch.nn.Linear(16, classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.sage1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)
        return x 
        # return x , F.log_softmax(self.lin(x))


class GAT(torch.nn.Module):
    def __init__(self, features, hidden, classes, heads):
        super(GAT, self).__init__()
        self.gat1 = GATConv(features, hidden, heads) 
        self.gat2 = GATConv(hidden*heads, hidden)  
        self.lin = torch.nn.Linear(hidden, classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)

        return x


def test(model,data):
    model.eval()
    # _, out = model(data)
    _,out = model(data)
    pred = out.argmax(dim=1) 
    result = {}
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    test_micro = f1_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(),average='micro')
    cr = classification_report(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(),digits=4)
    cm = confusion_matrix(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())
    result['micro_f1'] = test_micro
    result['cr'] = cr
    result['cm'] = cm
    return result 

def test_2(data):
    model.eval()
    z=model(data)
    split=get_split(num_samples=z.size()[0], train_ratio=0.7, test_ratio=0.2)
    y = torch.tensor(data.y,dtype = torch.int64)
    result = LREvaluator()(z, y, split)
    return result
    
    
    

#model = GAT(-1, 512, 20, heads=2).to(device)  # 定义GAT
model = GCN(data.num_node_features, 256, 20).to(device)
# model = GraphSAGE(data.num_node_features, 512, 2).to(device)
#model = GraphSAGE(-1, 512, 2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)
# 定义损失函数
criterion = nn.CrossEntropyLoss()

best = 0
best_cr = 0
max_f1 = 0
model.train()
log = open("gsage_log.txt",'wt')
res = open("gsage_res.csv","wt")
uniform_txt = open("homo.txt","wt")
start = time.time()
with tqdm(total=500, desc='(T)') as pbar:
    for epoch in range(1,501):
        optimizer.zero_grad()
        # a=
        # print(data.x)
        # out = model(data)
        _,out = model(data)
        y = torch.tensor(data.y[data.train_mask],dtype = torch.int64)
        # loss = F.nll_loss(out[data.train_mask], y)
        loss = criterion (out[data.train_mask], y)
        log.write(str(epoch)+":"+str(loss.item())+","+str(time.time()-start)+"\n")
        loss.backward()
        optimizer.step()
        if(epoch % 100 == 0):
            test_result = test(model,data)
            res.write(str(test_result['micro_f1'])+"\n")
            if(test_result['micro_f1'] > max_f1):
                best_f1 = test_result['micro_f1']
                best_cr = test_result['cr']
            print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}')

print("best f1 socre:", max_f1)
# print(test_result['cr'])
end = time.time()
print('------------------')
print(best_cr)




