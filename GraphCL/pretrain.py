import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, normalize
import torch, os, random, copy
import numpy as np
import gc
from torch.nn.utils import clip_grad_norm_
from graph_aug import mask_nodes, mask_edges, permute_edges, drop_nodes, subgraph
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from torch_ema import ExponentialMovingAverage

from modules.dualgraph.gnn import GNN
from modules.dualgraph.mol import smiles2graphwithface

from torch_geometric.data import Dataset, InMemoryDataset
from modules.dualgraph.dataset import DGData
from torch_geometric.loader import DataLoader

from modules.ogb.utils.features import (
    allowable_features,
    atom_to_feature_vector,
    bond_to_feature_vector,
    atom_feature_vector_to_dict,
    bond_feature_vector_to_dict,
)

from rdkit import Chem
import numpy as np
from modules.dualgraph.graph import get2DConformer, Graph, getface

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
seed_everything(2023)

device = 'cuda'

class CustomDataset(InMemoryDataset):
    def __init__(self, root='dataset_path', transform=None, pre_transform=None, df=None, target_type='pretrain', mode='train'):
        self.df = df
        self.target_type = target_type
        self.mode = mode
        super().__init__(root, transform, pre_transform, df)
        

    @property
    def raw_file_names(self):        
        return [f'raw_{i+1}.pt' for i in range(self.df.shape[0])]

    @property
    def processed_file_names(self):
        return [f'data_{i+1}.pt' for i in range(self.df.shape[0])]        

    def len(self):
        return self.df.shape[0]

    def get(self, idx):
        sid = self.sid_list[idx]
        dset = self.dataset_list[idx]
        # origin_graph =  torch.load(f'graph_pt/{sid}.pt')
        smiles = self.smiles_list[idx]
        origin_graph = self.get_graph(smiles)
        mask_graph1 = mask_edges(mask_nodes(copy.deepcopy(origin_graph), 0.3), 0.15)
        mask_graph2 = mask_edges(mask_nodes(copy.deepcopy(origin_graph), 0.15), 0.3)         

        return (origin_graph, mask_graph1, mask_graph2)

    def get_graph(self, smiles):
        data = DGData()

        graph = smiles2graphwithface(smiles)

        data.__num_nodes__ = int(graph["num_nodes"])
        data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
        data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
        data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)

        data.ring_mask = torch.from_numpy(graph["ring_mask"]).to(torch.bool)
        data.ring_index = torch.from_numpy(graph["ring_index"]).to(torch.int64)
        data.nf_node = torch.from_numpy(graph["nf_node"]).to(torch.int64)
        data.nf_ring = torch.from_numpy(graph["nf_ring"]).to(torch.int64)
        data.num_rings = int(graph["num_rings"])
        data.n_edges = int(graph["n_edges"])
        data.n_nodes = int(graph["n_nodes"])
        data.n_nfs = int(graph["n_nfs"])
        
        return data

    def process(self):                
        self.sid_list = self.df['MOL_ID'].values
        self.dataset_list = self.df['dataset_name'].values
        self.smiles_list = self.df['smiles'].values

triplet_loss = nn.TripletMarginLoss(margin=0.0, p=2)
criterion = nn.CrossEntropyLoss()

class MedModel(torch.nn.Module):

    def __init__(self):
        super(MedModel, self).__init__()
        self.ddi = True
        self.gnn = GNN(
                        mlp_hidden_size = 512,
                        mlp_layers = 4,
                        latent_size = 128,
                        use_layer_norm = False,
                        use_face=True,
                        ddi=self.ddi,
                        dropedge_rate = 0.1,
                        dropnode_rate = 0.1,
                        dropout = 0.1,
                        dropnet = 0.1,
                        global_reducer = "sum",
                        node_reducer = "sum",
                        face_reducer = "sum",
                        graph_pooling = "sum",
                        node_attn = True,
                        face_attn = True                        
                        )                
        self.proj = nn.Sequential(
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        )
    def forward(self, batch):
        mol = self.gnn(batch).squeeze(1)
        return self.proj(mol)

data = pd.read_parquet('GraphCL/data.parquet')

train_dataset = CustomDataset(df = data)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers = 8)

model = MedModel().to(device)
optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=30, verbose=False)

def loss_cl(x1, x2):
    T = 0.1
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss

best_loss = 1e6
start = 2
for epoch in range(1, 31):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader):
        batch = [bat.to(device) for bat in batch]
        outputs = [model(bat) for  bat in batch]        
        origin_output = outputs[0]
        
        mask_cl_loss = loss_cl(outputs[1], outputs[2])                    
        mask_t_loss = triplet_loss(outputs[0], outputs[1], outputs[2])

        loss = mask_cl_loss + mask_t_loss * 0.1

        optim.zero_grad()
        loss.backward() 
        optim.step()
        ema.update()
        
        train_loss += loss.cpu().item()
        
        gc.collect()
        torch.cuda.empty_cache()

    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(model.gnn.state_dict(), f'gnn_pretrain.pt')
    
    scheduler.step()
    print(f'EPOCH : {epoch} | train_loss : {train_loss/len(train_loader):.4f}')    