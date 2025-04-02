import pandas as pd
import argparse
import torch, os, random
import numpy as np
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from torch_ema import ExponentialMovingAverage
from matplotlib import pyplot as plt
from glob import glob
from modules.dualgraph.mol import smiles2graphwithface
from modules.dualgraph.gnn import GNN
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import train_test_split
import torch_geometric
from torch_geometric.data import Dataset, InMemoryDataset
from modules.dualgraph.dataset import DGData
from torch_geometric.loader import DataLoader
import dgl
import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)
    torch_geometric.seed_everything(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--save_path", default='ckpt', type=str)        
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--mode", type=str, default='MetaboGNN', 
                        choices=['MetaboGNN', 'Base', 'Scratch'])
    
    args = parser.parse_args()
    return args


class CustomDataset(InMemoryDataset):
    def __init__(self, root='dataset_path', transform=None, pre_transform=None, df=None, target_type='MLM', mode='train'):
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
        return len(self.graph_list)

    def get(self, idx):
        return self.graph_list[idx]

    def process(self):        
        smiles_list = self.df["SMILES"].values
        targets_list = self.df[['MLM', 'HLM']].values
        data_list = []
        for i in range(len(smiles_list)):
            data = DGData()
            smiles = smiles_list[i]
            targets = targets_list[i]
            graph = smiles2graphwithface(smiles)

            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.y = torch.Tensor([targets])

            data.ring_mask = torch.from_numpy(graph["ring_mask"]).to(torch.bool)
            data.ring_index = torch.from_numpy(graph["ring_index"]).to(torch.int64)
            data.nf_node = torch.from_numpy(graph["nf_node"]).to(torch.int64)
            data.nf_ring = torch.from_numpy(graph["nf_ring"]).to(torch.int64)
            data.num_rings = int(graph["num_rings"])
            data.n_edges = int(graph["n_edges"])
            data.n_nodes = int(graph["n_nodes"])
            data.n_nfs = int(graph["n_nfs"])        

            data_list.append(data)
        self.smiles_list = smiles_list  
        self.graph_list = data_list
        self.targets_list = targets_list

class MetaboGNN(torch.nn.Module):

    def __init__(self, mode):
        super(MetaboGNN, self).__init__()
        self.mode = mode
        self.ddi = True
        self.gnn = GNN(mlp_hidden_size = 512, mlp_layers = 2, latent_size = 128, use_layer_norm = False,
                        use_face=True, ddi=self.ddi, dropedge_rate = 0.1, dropnode_rate = 0.1, dropout = 0.1,
                        dropnet = 0.1, global_reducer = "sum", node_reducer = "sum", face_reducer = "sum", graph_pooling = "sum",                        
                        node_attn = True, face_attn = True)
        if self.mode != 'Scratch':
            state_dict=  torch.load('GraphCL/gnn_pretrain.pt', map_location='cpu')
            self.gnn.load_state_dict(state_dict, strict=False)

        self.fc1 = nn.Sequential(
                    nn.LayerNorm(128),
                    nn.Linear(128, 128,),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.1),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                    )
        self.fc2 = nn.Sequential(
                    nn.LayerNorm(128),
                    nn.Linear(128, 128,),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.1),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                    )

        self.fc1[-1].weight.data.normal_(mean=0.0, std=0.01)
        self.fc2[-1].weight.data.normal_(mean=0.0, std=0.01)

    def forward(self, batch):
        mol = self.gnn(batch)

        out1 = torch.sigmoid(self.fc1(mol).squeeze(1)) * 100        
        if self.mode == 'MetaboGNN':
            out2 = (torch.sigmoid(self.fc2(mol).squeeze(1))-0.5) * 200
        else:
            out2 = torch.sigmoid(self.fc1(mol).squeeze(1)) * 100        

        return out1, out2

def correlation_score(y_true, y_pred):
    y_true_centered = y_true - torch.mean(y_true, dim=1)[:, None]
    y_pred_centered = y_pred - torch.mean(y_pred, dim=1)[:, None]
    cov_tp = torch.sum(y_true_centered * y_pred_centered, dim=1) / (y_true.shape[1] - 1)
    var_t = torch.sum(y_true_centered ** 2, dim=1) / (y_true.shape[1] - 1)
    var_p = torch.sum(y_pred_centered ** 2, dim=1) / (y_true.shape[1] - 1)
    return cov_tp / torch.sqrt(var_t * var_p)


def correlation_loss(pred, target):
    return -torch.mean(correlation_score(target.unsqueeze(0), pred.unsqueeze(0)))

def inference(model, test_loader, mode):
    if mode == 'MetaboGNN':
        test_pred_MLM, test_pred_RES = [], []
        test_true_MLM, test_true_HLM = [], []             

        for batch in test_loader:
            with torch.no_grad():                    
                pred_mlm, pred_res = model(batch.to(args.device))
            targets = batch.y.to(args.device)                
            target_mlm, target_hlm = targets[:, 0], targets[:, 1]

            test_pred_MLM += pred_mlm.cpu().tolist()
            test_pred_RES += pred_res.cpu().tolist()

            test_true_HLM += target_hlm.cpu().tolist()
            test_true_MLM += target_mlm.cpu().tolist()

        test_pred_MLM, test_pred_RES = np.array(test_pred_MLM), np.array(test_pred_RES)
        test_pred_HLM = test_pred_MLM - test_pred_RES
    else:
        test_pred_MLM, test_pred_HLM = [], []
        test_true_MLM, test_true_HLM = [], []             

        for batch in test_loader:
            with torch.no_grad():                    
                pred_mlm, pred_hlm = model(batch.to(args.device))
            targets = batch.y.to(args.device)                
            target_mlm, target_hlm = targets[:, 0], targets[:, 1]

            test_pred_MLM += pred_mlm.cpu().tolist()
            test_pred_HLM += pred_hlm.cpu().tolist()

            test_true_HLM += target_hlm.cpu().tolist()
            test_true_MLM += target_mlm.cpu().tolist()

        test_pred_MLM, test_pred_HLM = np.array(test_pred_MLM), np.array(test_pred_HLM)
    return test_pred_MLM, test_true_MLM, test_pred_HLM, test_true_HLM

def main(args):
    seed_everything(args.seed)
    data = pd.read_csv('data/train_paper.csv', index_col=None)
    test = pd.read_csv('data/test_paper.csv', index_col=None)
    
    test['MLM_raw'], test['HLM_raw'] = test['MLM_raw'].str.replace('<', '').str.replace('>', ''), test['HLM_raw'].str.replace('<', '').str.replace('>', '')
    test['MLM'], test['HLM'] = test['MLM_raw'].astype(float), test['HLM_raw'].astype(float)

    train, valid = train_test_split(data, test_size=0.2, random_state=args.seed)
    train, valid = train.reset_index(drop=True), valid.reset_index(drop=True)

    train_dataset = CustomDataset(df = train, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers = 8)

    valid_dataset = CustomDataset(df = valid, mode='test')
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers = 8)

    test_dataset = CustomDataset(df = test, mode='test', target_type='MLM')
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers = 8) 

    model = MetaboGNN(mode=args.mode).to(args.device)
    mse_loss = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=1, verbose=False)
    best_val_loss = 1e6

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    for epoch in range(45):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(args.device)
            targets = batch.y.to(args.device)
            target_mlm, target_hlm = targets[:, 0], targets[:, 1]

            if args.mode == 'MetaboGNN':
                pred_mlm, pred_res = model(batch)                                        
                loss2 = mse_loss(pred_mlm-pred_res, target_hlm)
            else:
                pred_mlm, pred_hlm = model(batch)
                loss2 = mse_loss(pred_hlm, target_hlm)

            loss1 = mse_loss(pred_mlm, target_mlm) 

            loss = (loss1 + loss2)/2

            optim.zero_grad()
            loss.backward()
            grad = clip_grad_norm_(model.parameters(), 1000)
            optim.step()
            ema.update()
            
            train_loss += loss.cpu().item()
        
        model.eval()                    
        valid_preds_mlm, valid_label_mlm, valid_label_hlm, valid_preds_hlm = inference(model, valid_loader, args.mode)
        
        mlm_rmse = mean_squared_error(valid_preds_mlm, valid_label_mlm) ** (1/2)
        hlm_rmse = mean_squared_error(valid_preds_hlm, valid_label_hlm) ** (1/2)

        val_loss = (mlm_rmse + hlm_rmse)/2
        print(f"EPOCH {epoch:02d} | TRAIN LOSS: {train_loss / len(train_loader):.1f}")

        if val_loss < best_val_loss:
            print()
            print(f"\n🔥 Validation loss improved: {best_val_loss:.2f} → {val_loss:.2f}")
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{args.save_path}/{args.seed}_{args.mode}.pt')

            test_pred_MLM, test_true_MLM, test_pred_HLM, test_true_HLM = inference(model, test_loader, args.mode)

            test_mlm_rmse = mean_squared_error(test_pred_HLM, test_true_HLM) ** 0.5
            test_hlm_rmse = mean_squared_error(test_pred_MLM, test_true_MLM) ** 0.5

            print(f"🔍 Evaluating on test set:")
            print(f"    ▶ TEST MLM RMSE : {test_mlm_rmse:.2f}")
            print(f"    ▶ TEST HLM RMSE : {test_hlm_rmse:.2f}\n")

            print()

        scheduler.step()
        

if __name__ == '__main__':
    args = parse_args()
    main(args)