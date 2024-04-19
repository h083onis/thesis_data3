from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from spm_processor import SpmProcessor
from torchtext.vocab import build_vocab_from_iterator

class CodeChangeDataset(Dataset):
    def __init__(self, data_filepath, spm_modelpath, kind='train', max_file=5, max_line_len=512):
        if not isinstance(data_filepath, pd.DataFrame):
            self.df = pd.read_csv(data_filepath, header=0, index_col=0)
            self.df = self.df[self.df['kinds'] == kind]
        self.df = data_filepath
        self.commit = self.df['commit_id']
        self.data = self.df['input_data'].apply(eval)
        self.label = self.df['label']
        self.code_type = ['<cpp>','<java>','<python>']
        self.max_file = max_file
        self.max_line_len = max_line_len
        self.sp = SpmProcessor()
        self.sp.initializer(model_file=spm_modelpath)
        self.corpus = self.sp.generate_vocab()
        self.code_type_idx = [self.corpus[code_token] for code_token in self.code_type]
        self.features, self.segment_label, self.file_mask = self.process(self.data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.LongTensor(np.array(self.features[idx], dtype=int)), \
                torch.LongTensor(np.array(self.segment_label[idx], dtype=int)), \
                torch.BoolTensor(self.file_mask[idx]), \
                torch.as_tensor(int(self.label[idx]), dtype=torch.long)
        
    def process(self, data):
        features_list = []
        segment_label_list = []
        file_mask_list = []
        for files in data:
            file_mask = np.zeros(self.max_file)
            if len(files) < self.max_file:
                file_mask[len(files):] = 1
                [files.append(['','']) for i in range(self.max_file-len(files))]
            elif len(files) > self.max_file:
                files = files[:self.max_file]
            input_data = []
            segment_data = []
            data_list = [input_data, segment_data]
            [data_list[i].append(result) for lines in files for i, result in enumerate(self.transform(lines))]
            features_list.append(input_data)
            segment_label_list.append(segment_data)
            file_mask_list.append(file_mask)
        return features_list, segment_label_list, file_mask_list
    
    def transform(self, lines):
        lines2idx_list = [self.sp.encode(line) for line in lines]
        if len(lines2idx_list[0]) >= self.max_line_len and len(lines2idx_list[1]) >= self.max_line_len:
            lines2idx_list = [lines2idx[:self.max_line_len] for lines2idx in lines2idx_list]
        elif len(lines2idx_list[0]) < self.max_line_len and len(lines2idx_list[1]) > self.max_line_len:
            lines2idx_list[1] = lines2idx_list[1][:self.max_line_len+self.max_line_len-len(lines2idx_list[0])]
        elif len(lines2idx_list[0]) > self.max_line_len and len(lines2idx_list[1]) < self.max_line_len:
            lines2idx_list[0] = lines2idx_list[0][:self.max_line_len+self.max_line_len-len(lines2idx_list[1])]
        
        segment_label = np.concatenate([np.full(len(lines2idx_list[i]), i+1, dtype=int) for i in range(0, len(lines2idx_list))])
        lines2idx = lines2idx_list[0] + lines2idx_list[1]
        pad_id = self.corpus['<pad>']
        if len(lines2idx) <  2 * self.max_line_len:
            lines2idx += [pad_id] * (2 * self.max_line_len-len(lines2idx))
            segment_label = np.concatenate([segment_label, np.zeros(2 * self.max_line_len-len(segment_label))])

        return lines2idx, segment_label


class CNNDataset(Dataset):
    def __init__(self, data_filepath, spm_modelpath, kind='train', max_lines = 10, max_line_len=512):
        if not isinstance(data_filepath, pd.DataFrame):
            self.df = pd.read_csv(data_filepath, header=0, index_col=0)
            self.df = self.df[self.df['kinds'] == kind]
        self.df = data_filepath
        self.commit = self.df['commit_id']
        self.data = self.df['input_data'].apply(eval)
        self.labels = self.df['label']
        self.code_type = ['<cpp>','<java>','<python>']
        self.max_line_len = max_line_len
        self.max_lines = max_lines
        self.sp = SpmProcessor()
        self.sp.initializer(model_file=spm_modelpath)
        self.corpus = self.sp.generate_vocab()
        self.code_type_idx = [self.corpus[code_token] for code_token in self.code_type]
        self.features = self.process(self.data)

    # len()を使用すると呼ばれる
    def __len__(self):
        return len(self.features_values)

    # 要素を参照すると呼ばれる関数    
    def __getitem__(self, idx):
        features_x = torch.LongTensor(np.array(self.features[idx], dtype=int))
        labels =  torch.as_tensor(int(self.labels[idx]), dtype=torch.long)
        return features_x, labels
    
    def process(self, data):
        features_list = []
        [features_list.append(result) for files in data for lines in files for result in self.transform(lines)]
        if len(features_list) > self.max_lines:
            features_list = features_list[:self.max_lines]
        return features_list
    
    def transform(self, lines):
        lines2idx_list = [self.sp.encode(line) for line in lines]
        pad_id = self.corpus['<pad>']
        [lines2idx_list[i].extend([pad_id] * (self.max_line_len-len(lines2idx_list[i]))) if len(lines2idx_list[i]) < self.max_line_len else lines2idx_list[i][:self.max_line_len] for i in range(len(lines2idx_list))]
        return lines2idx_list

class CodeChangeDataset2(Dataset):
    def __init__(self, data_filepath, spm_modelpath=None, max_lines=64, max_line_len=32, kind='train'):
        self.df = pd.read_csv(data_filepath, header=0, index_col=0)
        self.corpus = self.generate_corpus(self.df[self.df['kinds'] == 'train']['input_data'].apply(eval))
        self.df = self.df[self.df['kinds'] == kind]
        self.df.reset_index(inplace=True, drop=True)
        self.commit = self.df['commit_id']
        self.data = self.df['input_data'].apply(eval)
        self.labels = self.df['label']
        self.max_line_len = max_line_len
        self.max_lines = max_lines

    # len()を使用すると呼ばれる
    def __len__(self):
        return len(self.data)

    # 要素を参照すると呼ばれる関数    
    def __getitem__(self, idx):
        features, lines_mask = self.process(self.data[idx])
        features_x = torch.LongTensor(np.array(features, dtype=int))
        lines_mask = torch.BoolTensor(lines_mask)
        labels =  torch.as_tensor(int(self.labels[idx]), dtype=torch.long)
        return features_x, lines_mask, labels
    
    def process(self, data):
        features_list = [self.transform(line) for line in data]
        lines_mask = np.zeros(self.max_lines)
        if len(features_list) > self.max_lines:
            features_list = features_list[:self.max_lines]
        elif len(features_list) < self.max_lines:
            [features_list.extend([[self.corpus['<pad>']]*self.max_line_len]) for i in range(self.max_lines-len(features_list))]
            lines_mask[len(data):] = 1
        return features_list
    
    def transform(self, line):
        lines2idx = [self.corpus[token] for token in line.split(' ')]
        pad_id = self.corpus['<pad>']
        if len(lines2idx) > self.max_line_len:
            lines2idx = lines2idx[:self.max_line_len]
        elif len(lines2idx) < self.max_line_len:
            lines2idx += [pad_id] * (self.max_line_len-len(lines2idx))
        return lines2idx
    
    def generate_corpus(self, data):
        sentences = [line.split(' ') for lines in data for line in lines if line != '']
        corpus = build_vocab_from_iterator(
            iterator=sentences,
            min_freq=3,
            specials=['<unk>', '<pad>']
        )
        corpus.set_default_index(corpus['<unk>'])
        return corpus
    
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
    
    
