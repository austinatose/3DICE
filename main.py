import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, precision_recall_curve, auc, roc_curve
from model import Model
from solver import Solver
from config.cfg import get_cfg_defaults
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
import glob
from functools import lru_cache

from dataset import MyDataset, collate_fn

def main():
    cfg = get_cfg_defaults()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(cfg)



    print(sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable parameters")

if __name__ == "__main__":
    main()