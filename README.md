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
pip install -r requirements.txt

If you are using conda:

conda create -n metabo_gnn python=3.9
conda activate metabo_gnn
pip install -r requirements.txt
```
## Project Structure

```bash
MetaboGNN/
│
├── data/                 # Data files (may need to be downloaded separately)
├── models/               # GNN model implementations
├── utils/                # Utility functions
├── configs/              # Configuration YAML files
├── train.py              # Main training script
├── inference.py          # Evaluation / Prediction
├── requirements.txt      # Required Python packages
└── README.md             # This file
```

## How to Run
### 1. Pretraining

The dataset used for pretraining is not included in this repository due to licensing restrictions.
Details about the data sources can be found in the associated manuscript.

Instead, we provide the pretrained model checkpoint (GraphCL/gnn_pretrain.pt) for reproducibility and downstream usage.

You can skip this step if you only wish to fine-tune the model or evaluate it using the provided checkpoint.

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
