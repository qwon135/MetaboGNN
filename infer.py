import pandas as pd
import argparse
import torch, os, random
import numpy as np
from torch import nn
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from modules.dualgraph.mol import smiles2graphwithface
from modules.dualgraph.gnn import GNN
import torch_geometric
from torch_geometric.data import InMemoryDataset
from modules.dualgraph.dataset import DGData
from torch_geometric.loader import DataLoader
import dgl
import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed=42):
    """모든 난수 생성기의 시드를 설정하는 함수"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch_geometric.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CustomDataset(InMemoryDataset):
    """분자 그래프 데이터셋 클래스"""
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
        """SMILES 문자열을 그래프로 변환하는 처리 함수"""
        smiles_list = self.df["SMILES"].values
        targets_list = self.df[['MLM', 'HLM']].values
        data_list = []
        
        for i in range(len(smiles_list)):
            data = DGData()
            smiles = smiles_list[i]
            targets = targets_list[i]
            graph = smiles2graphwithface(smiles)

            # 그래프 속성 설정
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
        if self.mode == 'MetaoGNN':
            out2 = (torch.sigmoid(self.fc2(mol).squeeze(1))-0.5) * 200
        else:
            out2 = torch.sigmoid(self.fc1(mol).squeeze(1)) * 100        

        return out1, out2
def evaluate_model(model, test_loader, device, model_name):
    """모델 평가 함수"""
    model.eval()
    test_pred_MLM, test_pred_HLM = [], []
    test_true_MLM, test_true_HLM = [], []
    
    for batch in test_loader:
        with torch.no_grad():
            if model_name == "MetaboGNN":
                pred_mlm, pred_res = model(batch.to(device))
                pred_hlm = pred_mlm - pred_res
            else:
                pred_mlm, pred_hlm = model(batch.to(device))
                
        targets = batch.y.to(device)
        target_mlm, target_hlm = targets[:, 0], targets[:, 1]

        test_pred_MLM.extend(pred_mlm.cpu().tolist())
        test_pred_HLM.extend(pred_hlm.cpu().tolist())
        test_true_MLM.extend(target_mlm.cpu().tolist())
        test_true_HLM.extend(target_hlm.cpu().tolist())

    mlm_rmse = mean_squared_error(test_pred_MLM, test_true_MLM) ** 0.5
    hlm_rmse = mean_squared_error(test_pred_HLM, test_true_HLM) ** 0.5
    
    return mlm_rmse, hlm_rmse

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    seed_everything(args.seed)
    device = args.device
    
    # 데이터 로드
    test = pd.read_csv('data/test_paper.csv', index_col=None)
    test['MLM_raw'] = test['MLM_raw'].str.replace('<', '').str.replace('>', '')
    test['HLM_raw'] = test['HLM_raw'].str.replace('<', '').str.replace('>', '')
    test['MLM'] = test['MLM_raw'].astype(float)
    test['HLM'] = test['HLM_raw'].astype(float)
    
    test_dataset = CustomDataset(df=test, mode='test', target_type='MLM')
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)
    
    # 결과 저장 딕셔너리
    results = {'Model': [], 'MLM RMSE': [], 'HLM RMSE': []}
    
    # 모델 1: Scratch 평가
    model_scratch = MetaboGNN(mode='Scratch').to(device)
    model_scratch.load_state_dict(torch.load('ckpt/2025_Scratch.pt'))
    mlm_rmse, hlm_rmse = evaluate_model(model_scratch, test_loader, device, "Scratch")
    results['Model'].append('Scratch')
    results['MLM RMSE'].append(mlm_rmse)
    results['HLM RMSE'].append(hlm_rmse)
    
    # 모델 2: Base 평가
    model_base = MetaboGNN(mode='Base').to(device)
    model_base.load_state_dict(torch.load('ckpt/2025_base.pt'))
    mlm_rmse, hlm_rmse = evaluate_model(model_base, test_loader, device, "Base")
    results['Model'].append('Base')
    results['MLM RMSE'].append(mlm_rmse)
    results['HLM RMSE'].append(hlm_rmse)
    
    # 모델 3: MetaboGNN 평가
    model_metabognn = MetaboGNN(mode='MetaboGNN').to(device)
    model_metabognn.load_state_dict(torch.load('ckpt/2025.pt'))
    mlm_rmse, hlm_rmse = evaluate_model(model_metabognn, test_loader, device, "MetaboGNN")
    results['Model'].append('MetaboGNN')
    results['MLM RMSE'].append(mlm_rmse)
    results['HLM RMSE'].append(hlm_rmse)
    
    # 결과 표시
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # 결과 시각화
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(results['Model']))
    
    plt.bar(index, results['MLM RMSE'], bar_width, label='MLM RMSE', color='skyblue')
    plt.bar(index + bar_width, results['HLM RMSE'], bar_width, label='HLM RMSE', color='salmon')
    
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.title('Model Comparison')
    plt.xticks(index + bar_width/2, results['Model'])
    plt.legend()
    plt.savefig('Performance.png')
    plt.close()

if __name__ == '__main__':
    main()