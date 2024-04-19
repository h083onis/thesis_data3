import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from model import Model
from code_transformer import CodeChangeTransformer
from dataset import CodeChangeDataset
from utils import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from utils import Logger
from sklearn.metrics import roc_auc_score
import gc

logger = Logger()

class ModelCCT(Model):
    
    def __init__(self):
        super.__init__()
    
    def train(self, filepath, valid_idx):
        trin_dataset = CodeChangeDataset(
            filepath,
            spm_modelpath = self.params['spm_modelpath'],
            max_file=self.params['max_file'],
            max_line_len=self.params['max_line_len']
            )
        
        self.model = CodeChangeTransformer(self.params)
    def predict(self):
        self.model.eval()