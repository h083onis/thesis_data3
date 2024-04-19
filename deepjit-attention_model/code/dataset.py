import pandas as pd
import torch
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
import numpy as np
from torch.utils.data import Dataset
class CNNDataset(Dataset):
    def __init__(self, data_filepath, spm_modelpath=None, max_lines=64, max_line_len=32, kind='train'):
        self.df = pd.read_csv(data_filepath, header=0, index_col=0)
        self.corpus = self.generate_corpus(self.df[self.df['kinds'] == 'train']['input_data'].apply(eval))
        self.df = self.df[self.df['kinds'] == kind]
        self.df.reset_index(inplace=True, drop=True)
        self.commit = self.df['commit_id']
        self.data = self.df['input_data'].apply(eval)
        self.labels = self.df['label']
        # self.code_type = ['<cpp>','<java>','<python>']
        self.max_line_len = max_line_len
        self.max_lines = max_lines
        
        # self.code_type_idx = [self.corpus[code_token] for code_token in self.code_type]


    # len()を使用すると呼ばれる
    def __len__(self):
        return len(self.data)

    # 要素を参照すると呼ばれる関数    
    def __getitem__(self, idx):
        features = self.process(self.data[idx])
        features_x = torch.LongTensor(np.array(features, dtype=int))
        labels =  torch.as_tensor(int(self.labels[idx]), dtype=torch.long)
        return features_x, labels
    
    def process(self, data):
        features_list = [self.transform(line) for line in data]
        if len(features_list) > self.max_lines:
            features_list = features_list[:self.max_lines]
        elif len(features_list) < self.max_lines:
            features_list.extend([[self.corpus['<pad>']]*self.max_line_len for i in range(self.max_lines-len(features_list))])
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
    
    
class CNNDataset2(Dataset):
    def __init__(self, data_filepath, spm_modelpath=None, max_lines=64, max_line_len=32, kind='train'):
        self.df = pd.read_csv(data_filepath, header=0, index_col=0)
        self.corpus = self.generate_corpus(self.df[self.df['kinds'] == 'train']['input_data'].apply(eval))
        self.df = self.df[self.df['kinds'] == kind]
        self.df.reset_index(inplace=True, drop=True)
        self.commit = self.df['commit_id']
        self.data = self.df['input_data'].apply(eval)
        self.labels = self.df['label']
        # self.code_type = ['<cpp>','<java>','<python>']
        self.max_line_len = max_line_len
        self.max_lines = max_lines
        
        # self.code_type_idx = [self.corpus[code_token] for code_token in self.code_type]


    # len()を使用すると呼ばれる
    def __len__(self):
        return len(self.data)

    # 要素を参照すると呼ばれる関数    
    def __getitem__(self, idx):
        features, segment = self.process(self.data[idx])
        features_x = torch.LongTensor(np.array(features, dtype=int))
        segment = torch.LongTensor(np.array(segment, dtype=int))
        labels =  torch.as_tensor(int(self.labels[idx]), dtype=torch.long)
        return features_x, segment, labels
    
    def process(self, data):
        features_list = []
        segments_list = []
        data_list = [features_list, segments_list]
        [data_list[i].append(result) for line in data for i, result in enumerate(self.transform(line))]
        if len(features_list) > self.max_lines:
            features_list = features_list[:self.max_lines]
            segments_list = segments_list[:self.max_lines]
        elif len(features_list) < self.max_lines:
            features_list.extend([[self.corpus['<pad>']]*self.max_line_len for i in range(self.max_lines-len(features_list))])
            segments_list.extend([[0]*self.max_line_len for i in range(self.max_lines-len(segments_list))])
        return features_list, segments_list
    
    def transform(self, line):
        lines2idx = [self.corpus[token] for token in line.split(' ')]
        pad_id = self.corpus['<pad>']
        added_code_id = self.corpus['<added_code>']
        deleted_code_id = self.corpus['<deleted_code>']
        pos_id = np.where((np.array(lines2idx) == added_code_id) | (np.array(lines2idx) == deleted_code_id))[0][0]
        segment_label = np.ones(len(lines2idx))
        segment_label[pos_id:] = 2
        
        if len(lines2idx) > self.max_line_len:
            lines2idx = lines2idx[:self.max_line_len]
            segment_label = segment_label[:self.max_line_len]
        elif len(lines2idx) < self.max_line_len:
            lines2idx += [pad_id] * (self.max_line_len-len(lines2idx))
            segment_label = np.concatenate([segment_label, np.zeros(self.max_line_len-len(segment_label))])
        return lines2idx, segment_label
    
    def generate_corpus(self, data):
        sentences = [line.split(' ') for lines in data for line in lines if line != '']
        corpus = build_vocab_from_iterator(
            iterator=sentences,
            min_freq=3,
            specials=['<unk>', '<pad>']
        )
        corpus.set_default_index(corpus['<unk>'])
        return corpus