## GraphCL Pretraining
### Overview
- This repository contains code for pretraining graph neural networks using the GraphCL (Graph Contrastive Learning) framework. Our implementation focuses on molecular graph representation learning.
### 333333Dataset
- The pretraining dataset consists of 2,000,000 drug-like molecules sampled from the ZINC database. The data selection criteria follow the methodology outlined in the MoleculeNet paper. 
- These molecules were specifically chosen for their structural similarity to known drugs, making them ideal for pharmaceutical applications.
### Data Source
- Base Dataset: ZINC database
- Sample Size: 2,000,000 molecules
- Selection Criteria: Drug-like molecular properties
- Reference: MoleculeNet paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC5868307/

### Pretraining Details
- The pretraining process follows the GraphCL framework, which uses contrastive learning to learn molecular representations.
