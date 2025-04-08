import os
import time
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from modules.dualgraph.dataset import DGData
from modules.dualgraph.mol import smiles2graphwithface
from infer import MetaboGNN, seed_everything
import argparse
import warnings
warnings.filterwarnings("ignore")

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list):
        self.smiles_list = smiles_list
        self.graphs = []
        self.process()

    def process(self):
        for smi in self.smiles_list:
            data = DGData()
            graph = smiles2graphwithface(smi)

            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).long()
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).long()
            data.x = torch.from_numpy(graph["node_feat"]).long()

            data.ring_mask = torch.from_numpy(graph["ring_mask"]).bool()
            data.ring_index = torch.from_numpy(graph["ring_index"]).long()
            data.nf_node = torch.from_numpy(graph["nf_node"]).long()
            data.nf_ring = torch.from_numpy(graph["nf_ring"]).long()
            data.num_rings = int(graph["num_rings"])
            data.n_edges = int(graph["n_edges"])
            data.n_nodes = int(graph["n_nodes"])
            data.n_nfs = int(graph["n_nfs"])
            self.graphs.append(data)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='Path to input CSV file with SMILES column')
    parser.add_argument('--smiles', type=str, help='Single or comma-separated SMILES string(s)')
    parser.add_argument('--model_ckpt', type=str, default='ckpt/2025_MetaboGNN.pt', help='Path to trained model checkpoint')
    parser.add_argument('--save_dir', type=str, default='./outputs', help='Directory to save the prediction CSV')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    seed_everything(42)

    # --- Load SMILES ---
    if args.csv:
        df = pd.read_csv(args.csv)
        if 'SMILES' not in df.columns:
            raise ValueError("CSV must contain a 'SMILES' column.")
        smiles_list = df['SMILES'].tolist()
    elif args.smiles:
        smiles_list = [s.strip() for s in args.smiles.split(',')]
    else:
        raise ValueError("You must provide either --csv or --smiles.")

    # --- Load model ---
    device = torch.device(args.device)
    model = MetaboGNN(mode='MetaboGNN').to(device)
    model.load_state_dict(torch.load(args.model_ckpt, map_location=device))
    model.eval()

    # --- Prepare dataset ---
    dataset = InferenceDataset(smiles_list)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # --- Predict ---
    all_results = []
    if args.smiles:
        print(f"{'SMILES':<45} {'MLM_Pred':>10} {'HLM_Pred':>10}")

    for i, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            pred_mlm, pred_res = model(batch)
            pred_hlm = pred_mlm - pred_res
            for smi, mlm, hlm in zip(smiles_list[i * 64:(i + 1) * 64], pred_mlm.tolist(), pred_hlm.tolist()):
                if args.smiles:
                    print(f"{smi:<45} {mlm:>10.2f} {hlm:>10.2f}")
                all_results.append({'SMILES': smi, 'MLM_Pred': round(mlm, 2), 'HLM_Pred': round(hlm, 2)})

    # --- Save to CSV ---
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"MetaboGNN_result_{timestamp}.csv"
    full_path = os.path.join(args.save_dir, filename)
    pd.DataFrame(all_results).to_csv(full_path, index=False)

    print(f"\n✅ Prediction results saved to: {full_path}")

if __name__ == '__main__':
    main()
