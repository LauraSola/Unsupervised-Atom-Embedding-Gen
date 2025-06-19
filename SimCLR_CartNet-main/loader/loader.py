# Copyright Universitat Politècnica de Catalunya 2024 https://imatge.upc.edu
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import torch
#from dataset.datasetADP import DatasetADP
#from dataset.datasetEnergies import DatasetEnergies
#from dataset.utils import compute_knn, compute_inf_potnet, compute_levels
from torch_geometric.graphgym.config import cfg
from torch_geometric.loader import DataLoader
import random
import os.path as osp
import numpy as np
from tqdm import tqdm
from dataset.datasetMP import DatasetMP
import glob

def create_loader():
    """
    Create data loader object

    Returns: List of PyTorch data loaders

    """

    if cfg.data == "MP":
        dat = glob.glob(osp.join(cfg.dataset_path, "*.cif"))
        print("Totes les dades a train, tamany = ", len(dat))
        
        dataset_train = DatasetMP(root=cfg.dataset_path, data=dat[:8], radius=cfg.radius, max_neigh=cfg.max_neighbours, augment=cfg.augment, name="MP_train")
        
    else:
        raise Exception("Dataset not implemented")
   
    train_loader = DataLoader(dataset_train, batch_size=cfg.batch, persistent_workers=True, shuffle=True, num_workers=cfg.workers, pin_memory=True)
        
    return train_loader
