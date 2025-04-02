# MetaboGNN

MetaboGNN is a Graph Neural Network-based framework for predicting liver metabolic stability using molecular graph representations and cross-species experimental data.

This code was developed as part of our study:  
**"MetaboGNN: Predicting Liver Metabolic Stability with Graph Neural Networks and Cross-Species Data"**  
(submitted to *Journal of Cheminformatics*)

---

## 📦 Installation

We recommend using Python 3.9+ and a virtual environment.

```bash
git clone https://github.com/qwon135/MetaboGNN.git
cd MetaboGNN
```
### ⚠️ CUDA Runtime Requirement

This project uses PyTorch with GPU acceleration, and requires:

✅ NVIDIA GPU

✅ Driver version ≥ 520 (supports CUDA 11.8)

✅ CUDA 11.8 Runtime

You do not need to install the full CUDA Toolkit.
The runtime only is enough.

✅ You can check your driver version with:
```bash
nvidia-smi
```

### 🔗 Download CUDA 11.8 Runtime:
If `nvidia-smi` does not work or your driver is outdated, install the latest version here:
👉 https://developer.nvidia.com/cuda-11-8-0-download-archive

### 🧪 Environment Setup (with conda)
If you are using conda:

```bash
conda create -n metabo_gnn python=3.9
conda activate metabo_gnn

conda install pytorch==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c dglteam/label/th21_cu118 dgl
conda install conda-forge::pytorch_geometric

conda install pytorch-scatter=2.1.2 -c pyg
conda install pytorch-sparse=0.6.18 -c pyg

pip install -r requirements.txt
```


## Project Structure

```bash
MetaboGNN/
│
├── Benchmark/               # Baseline methods (e.g., MS_BACL, PredMS) from previous studies for performance comparison
├── data/                    # Dataset for liver metabolic stability prediction 
├── EdgeShaper/              # Utilities for bond-level interpretability and visualization
├── GraphCL/                 # Code for self-supervised pretraining 
├── modules/                 # Core GNN model components and training utilities
├── edgeshaper.ipynb         # Jupyter notebook to visualize bond-level model interpretation
├── finetune.py              # Fine-tuning with pretrained model on cross-species metabolic stability prediction
├── finetune_scratch.py      # Training from scratch (no pretraining or cross-species)
├── finetune_base.py         # Evaluation of pretrained model without cross-species fine-tuning
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

```

## How to Run
### 1. Pretraining

The dataset used for pretraining is not included in this repository due to licensing constraints and file size. However, it can be downloaded from the following link:

📁 [Download pretraining dataset](https://drive.google.com/drive/folders/1Vowev9pZtRBFOXA_zCN9YTLO9ECIKEV7?usp=sharing)

After downloading, please place the extracted files inside the `GraphCL/` directory so that the training script can access them.

We also provide a pretrained model checkpoint (`GraphCL/gnn_pretrain.pt`) for users who wish to skip this step.

To perform self-supervised pretraining from scratch, run:

```bash
PYTHONPATH=. python GraphCL/pretrain.py
```

### 2. Fine-tuning Experiments

We provide three fine-tuning scenarios for ablation and comparison:

```bash
python finetune.py
# Main experiment: Fine-tuning pretrained GNN on cross-species metabolic stability task

python finetune_scratch.py
# Ablation: Training from scratch without pretraining for comparison

python finetune_base.py
# Representation-only: Evaluates pretrained GNN without cross-species fine-tuning
```

### 3. Model Interpretability

We provide a Jupyter notebook that visualizes how the model interprets molecular structures, focusing on bond-level features.

- `edgeshaper.ipynb`: Highlights important chemical bonds based on attention weights or gradient-based signals.
  - Helps identify which bonds are most influential in predicting liver metabolic stability.
  - Requires a fine-tuned model (stored in the `ckpt/` directory).

https://drive.google.com/drive/folders/1Vowev9pZtRBFOXA_zCN9YTLO9ECIKEV7?usp=sharing