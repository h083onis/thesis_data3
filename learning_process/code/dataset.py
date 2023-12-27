import torch
import numpy as np

class CNNDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        features, labels = dataset[0], dataset[1]
        self.features_values = features
        self.labels = labels

    # len()を使用すると呼ばれる
    def __len__(self):
        return len(self.features_values)

    # 要素を参照すると呼ばれる関数    
    def __getitem__(self, idx):
        features_x = torch.LongTensor(self.features_values[idx])
        labels =  torch.as_tensor(int(self.labels[idx]), dtype=torch.long)
        return features_x, labels


class TFDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        features, masks, labels = dataset[0], dataset[1], dataset[2]
        self.features_values = features
        self.masks_values = masks
        self.labels = labels

    # len()を使用すると呼ばれる
    def __len__(self):
        return len(self.features_values)

    # 要素を参照すると呼ばれる関数    
    def __getitem__(self, idx):
        features_x = torch.LongTensor(self.features_values[idx])
        masks_x = torch.BoolTensor(self.masks_values[idx])
        labels =  torch.as_tensor(int(self.labels[idx]), dtype=torch.long)
        # labels = torch.LongTensor(self.labels[idx])
        return features_x, masks_x, labels