import os
import time
import copy
import torch
import pickle
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import to_hetero

from tqdm import tqdm
from utils.loader import DNS, to_homogeneous
from torch.optim import Adam,AdamW
from GCL.models import BootstrapContrast
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.loader import DataLoader
from utils.util import get_split,RFEvaluator,LazyEvaluator
from torch_geometric.datasets import IMDB
from utils.PyGCL.GCL.eval.logistic_regression import LREvaluator
from utils.util import plot_confusion_matrix

from sklearn.metrics import f1_score,classification_report ,confusion_matrix
from sklearn.linear_model import LogisticRegression
from utils.util import MIEstimator,Attention,attack_graph,get_split,subgraph_sampling,plot_confusion_matrix,attack_graph_feature,attack_label,attack_graph2,attack_graph3,attack_graph4
def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()
    
def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def bgrl_loss(z1, h2):
  # z1: online representation and prediction, shape: (N, D)
  # h2: target representation, shape: (N, D)
  # N: number of nodes, D: dimension of representation
  # returns: scalar loss value
  z1 = F.normalize(z1, dim=1) # normalize along feature dimension
  h2 = F.normalize(h2, dim=1) # normalize along feature dimension
  cos_sim = torch.sum(z1 * h2, dim=1) # element-wise product and sum, shape: (N,)
  loss = -torch.mean(cos_sim) # negative mean cosine similarity
  return loss

class Normalize(torch.nn.Module):
    def __init__(self, dim=None, norm='batch'):
        super().__init__()
        if dim is None or norm == 'none':
            self.norm = lambda x: x
        if norm == 'batch':
            self.norm = torch.nn.BatchNorm1d(dim)
        elif norm == 'layer':
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)

class MLP(torch.nn.Module):
 
    def __init__(self,num_i,num_h,num_o):
        super(MLP,self).__init__()
        
        self.linear1=torch.nn.Linear(num_i,num_h)
        self.relu=torch.nn.ReLU()
        self.linear2=torch.nn.Linear(num_h,num_h) #2个隐层
        self.relu2=torch.nn.ReLU()
        self.linear3=torch.nn.Linear(num_h,num_o)
  
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2,
                 encoder_norm='batch', projector_norm='batch'):
        super(GConv, self).__init__()
        self.activation = torch.nn.PReLU()
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

        self.batch_norm = Normalize(hidden_dim, norm=encoder_norm)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=projector_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.batch_norm(z)
        return z, self.projection_head(z)

class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2,
                 encoder_norm='batch', projector_norm='batch'):
        super(GAT, self).__init__()
        self.activation = torch.nn.PReLU()
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()
        self.layers.append(GATConv(input_dim, hidden_dim, heads=2))
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(hidden_dim, hidden_dim,  heads=2))

        self.batch_norm = Normalize(hidden_dim, norm=encoder_norm)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=projector_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.batch_norm(z)
        return z, self.projection_head(z)


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, dropout=0.2, predictor_norm='batch'):
        super(Encoder, self).__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.augmentor = augmentor
        self.mlp = MLP(256,128,256)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))
        self.lin = torch.nn.Linear(512, 16)
    
    def project(self,z):
        return self.lin(z)
        
    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)

            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(self.get_target_encoder().parameters(), self.online_encoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        h1, h1_online = self.online_encoder(x, edge_index, edge_weight)
        h2, h2_online = self.online_encoder(x, edge_index, edge_weight)

        h1_pred = self.predictor(h1_online)
        h2_pred = self.predictor(h2_online)

        with torch.no_grad():
            _, h1_target = self.get_target_encoder()(x, edge_index, edge_weight)
            _, h2_target = self.get_target_encoder()(x, edge_index, edge_weight)

        return h1, h2, h1_pred, h2_pred, h1_target, h2_target

uniform_txt = open("bgrl_uniform.csv","wt")

def train(encoder_model, contrast_model, data, optimizer, epoch):
    encoder_model.train()
    optimizer.zero_grad()
    h1, h2, h1_pred, h2_pred, h1_target, h2_target = encoder_model(data.x, data.edge_index, data.edge_attr)
    # loss = contrast_model(h1_pred=h1_pred, h2_pred=h2_pred, h1_target=h1_target.detach(), h2_target=h2_target.detach())
    loss = (bgrl_loss(h1_target,h1_pred) + bgrl_loss(h2_target,h2_pred)) / 2
    if(epoch % 1000 == 0):
        align_l = align_loss(h1,h2).item()
        uniform_l = uniform_loss(encoder_model.project(torch.cat([h1, h2], dim=1))).item()
        print(align_l)
        print(uniform_l)
        uniform_txt.write(str(uniform_l)+","+str(align_l)+"\n")
    loss.backward()
    optimizer.step()
    encoder_model.update_target_encoder(0.99)
    return loss.item()

def test(encoder_model, data):
    encoder_model.eval()
    h1, h2, _, _, _, _ = encoder_model(data.x, data.edge_index)
    z = torch.cat([h1, h2], dim=1)
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
    h1, h2, _, _, _, _ = encoder_model(data.x, data.edge_index)
    z = torch.cat([h1, h2], dim=1)
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

    best_f1 = 0
    best_cr = 0
    device = torch.device('cuda')
    with open('/data/sda/haoqingyu/DNS/data_hash_multi.pickle','rb') as f:
        data = pickle.load(f)
    print(data)
    data_orignal = copy.deepcopy(data)
    #边扰动
    # data = attack_graph(data,0.6)
    # data = attack_graph2(data,0.6)
    # data = attack_graph3(data,0.6)
    # data = attack_graph4(data,0.6)
    #节点扰动
    # data = attack_graph_feature(data,0.6)

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

    # train_r = 0.001
    # test = 0.9 - train_r
    # val = 0.1
   

    data = to_homogeneous(data).to(device)
    data_orignal = to_homogeneous(data_orignal).to(device)
    print(data)
    id = (data.train_mask | data.test_mask | data.val_mask).detach().cpu()
    
    aug1 = A.Compose([A.EdgeRemoving(pe=0.2), A.NodeDropping(pn=0.2)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.2), A.NodeDropping(pn=0.2)])

    gconv = GConv(input_dim=data.num_node_features, hidden_dim=128, num_layers=2).to(device)
    gatconv = GAT(input_dim=data.num_features,hidden_dim=256,num_layers=2).to(device)
    # gatconv = to_hetero(gatconv, data.metadata(), aggr='sum')
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=128).to(device)
    contrast_model = BootstrapContrast(loss=L.BootstrapLatent(), mode='L2L').to(device)

    optimizer = AdamW(encoder_model.parameters(), lr=0.002)
    max_f1 = 0
    best_cr = 0
    log = open("bgrl_log.txt",'wt')
    res = open("res.csv","wt")
    start = time.time()
    with tqdm(total=200, desc='(T)') as pbar:
        for epoch in range(1, 201):
            loss = train(encoder_model, contrast_model, data, optimizer, epoch)
            log.write(str(epoch)+":"+str(loss)+","+str(time.time()-start)+"\n")
            pbar.set_postfix({'loss': loss})
            pbar.update()
            if( epoch % 10 == 0):
                test_result,z = test_2(encoder_model, data)
                res.write(str(test_result['micro_f1'])+"\n")
                if(test_result['micro_f1'] > best_f1):
                    best_f1 = test_result['micro_f1']
                    best_cr = test_result['cr']
                    best_cm = test_result['cm']
                print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}')
                # print("类别的准确率：",test_result['classification_report'])
    print(best_cr)
if __name__ == '__main__':
    main()