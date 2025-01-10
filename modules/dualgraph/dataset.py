from torch_geometric.data import InMemoryDataset
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data
import torch
import os.path as osp
from torch_sparse import SparseTensor
import re
import os
import shutil
from ..ogb.utils.url import decide_download, download_url, extract_zip
from .mol import smiles2graphwithface
import numpy as np
import io



class DGData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if isinstance(value, SparseTensor):
            return (0, 1)
        elif bool(re.search("(index|face)", key)):
            return -1
        elif bool(re.search("(nf_node|nf_ring)", key)):
            return -1
        return 0

    def __inc__(self, key, value, *args, **kwargs):
        if bool(re.search("(ring_index|nf_ring)", key)):
            return int(self.num_rings)
        elif bool(re.search("(index|face|nf_node)", key)):
            return self.num_nodes
        else:
            return 0

