import torch
import pickle
import copy
import os.path as osp
import GCL.losses as L
import torch_geometric.transforms as T

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import SingleBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform
from torch_geometric.datasets import Planetoid

import torch_geometric.transforms as T
from utils.loader import DNS,to_homogeneous
from sklearn.metrics import f1_score,classification_report ,confusion_matrix
from utils.PyGCL.GCL.eval.logistic_regression import LREvaluator

from sklearn.metrics import f1_score,classification_report ,confusion_matrix
from sklearn.linear_model import LogisticRegression
from utils.util import plot_confusion_matrix
from utils.util import MIEstimator,Attention,attack_graph,get_split,subgraph_sampling,plot_confusion_matrix,attack_graph_feature,attack_label,attack_graph2,attack_graph3,attack_graph4

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()
    
def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.activations.append(nn.PReLU(hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv, act in zip(self.layers, self.activations):
            z = conv(z, edge_index, edge_weight)
            z = act(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, 16)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        g = self.project(torch.sigmoid(z.mean(dim=0, keepdim=True)))
        zn = self.encoder(*self.corruption(x, edge_index))
        return z, g, zn
    
    def projector(self,z):
        return self.lin(z)

uniform_txt = open("dgi_uniform.txt","wt")
def train(encoder_model, contrast_model, data, optimizer,epoch):
    encoder_model.train()
    optimizer.zero_grad()
    z, g, zn = encoder_model(data.x, data.edge_index)
    if(epoch % 10000 == 0):
        uniform_l = uniform_loss(encoder_model.projector(z)).item()
        print(uniform_l)
        uniform_txt.write(str(uniform_l)+"\n")
    loss = contrast_model(h=z, g=g, hn=zn)
    loss.backward()
    optimizer.step()
    return loss


def test(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index)
    split = get_split(num_samples=z.size()[0], train_ratio=0.01, test_ratio=0.89)
    y = torch.tensor(data.y,dtype = torch.int64)
    result = LREvaluator()(z, y, split)
    return result,z

def test_2(encoder_model,data):
    result = {}
    clf = LogisticRegression()
    train_mask = data.train_mask
    test_mask = data.test_mask
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index)
    clf.fit(z[train_mask].detach().cpu().numpy(),data.y[train_mask].detach().cpu().numpy())
    pred = clf.predict(z[test_mask].detach().cpu().numpy())
    test_micro = f1_score(data.y[test_mask].detach().cpu().numpy(), pred,average='micro')
    cr = classification_report(data.y[test_mask].detach().cpu().numpy(), pred,digits=4)
    cm = confusion_matrix(data.y[test_mask].detach().cpu().numpy(), pred)
    result['micro_f1'] = test_micro
    result['cr'] = cr
    result['cm'] = cm

    return result,z

def main():
    
    device = torch.device('cuda')
    with open('/data/sda/haoqingyu/DNS/data_hash_multi.pickle','rb') as f:
        data = pickle.load(f)
    print(data)
    data_orignal = copy.deepcopy(data)

    # #节点扰动
    #data = attack_graph_feature(data,0.6)
    #边扰动
    # data = attack_graph(data,0.2)
    # data = attack_graph2(data,0.2)
    # data = attack_graph3(data,0.2)
    # data = attack_graph4(data,0.2)
    


    # train_r = 0.001
    # test = 0.9 - train_r
    # val = 0.1
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

    # data = mask(data, 'domain_node', test, val).to('cpu')

    data = to_homogeneous(data).to(device)
    data_orignal = to_homogeneous(data_orignal).to(device)

    id = (data.train_mask | data.test_mask | data.val_mask).detach().cpu()



    gconv = GConv(input_dim=data.num_features, hidden_dim=256, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, hidden_dim=256).to(device)
    contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.001)


    max_f1 = 0
    best_cr = 0
    log = open("dgi_log.txt",'wt')
    res = open("dgi_res.csv","wt")
    import time
    start = time.time()
    with tqdm(total=500, desc='(T)') as pbar:
        for epoch in range(1, 501):
            loss = train(encoder_model, contrast_model, data, optimizer, epoch)
            log.write(str(epoch)+":"+str(loss)+","+str(time.time()-start)+"\n")
            if( epoch % 10 == 0):
                test_result,_ = test_2(encoder_model,data)
                res.write(str(test_result['micro_f1'])+"\n")
                if(test_result['micro_f1'] > max_f1):
                    max_f1 = test_result['micro_f1']
                    best_cr = test_result['cr']
            #         best_cm = test_result['cm']
            #     print(f'(E): F1Mi={test_result["micro_f1"]:.4f}')
            #     print(best_cr)
            #     # print(f'(E): F1Mi={test_result["micro_f1"]:.4f}')
            # pbar.set_postfix({'loss': loss})
            # pbar.update()


    print("best f1 socre:",max_f1)
    print(best_cr)
    # plot_confusion_matrix(best_cm)

    


if __name__ == '__main__':
    main()