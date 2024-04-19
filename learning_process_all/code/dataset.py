from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from spm_processor import SpmProcessor
from torchtext.vocab import build_vocab_from_iterator

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
    
    
    
class CodeAndMsgDataset(torch.utils.data.Dataset):
    def __init__(self, data_filepath, params, kind='train'):
        self.params = params
        self.df = pd.read_csv(data_filepath, header=0, index_col=0)
        self.code_corpus = self.generate_corpus(self.df[self.df['kinds'] == 'train']['code_data'].apply(eval))
        self.df = self.df[self.df['kinds'] == kind]
        self.df.reset_index(inplace=True, drop=True)
        self.commit = self.df['commit_id']
        self.code_data = self.df['code_data'].apply(eval)
        self.msg_data = self.df['msg_data'].apply(eval)
        self.labels = self.df['label']
        self.msg_sp = SpmProcessor()
        self.msg_sp.initializer(model_file=self.params['msg_spm_modelpath'])
        self.msg_corpus = self.msg_sp.generate_vocab()
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        code_feature = self.process_code(self.code_data[idx]) 
        msg_feature = self.process_msg(self.msg_data[idx])
        code_feature = torch.LongTensor(np.array(code_feature, dtype=int))
        msg_feature = torch.LongTensor(np.array(msg_feature, dtype=int))
        labels =  torch.as_tensor(int(self.labels[idx]), dtype=torch.long)
        return code_feature, msg_feature, labels
    
    def process_code(self, data):
        features_list = [self.transform_code(line) for line in data]
        lines_mask = np.zeros(self.params['code_max_lines'])
        if len(features_list) > self.max_lines:
            features_list = features_list[:self.params['code_max_lines']]
        elif len(features_list) < self.max_lines:
            [features_list.extend([[self.corpus['<pad>']]*self.params['code_max_line_len']]) for i in range(self.params['code_max_lines']-len(features_list))]
            lines_mask[len(data):] = 1
        return features_list
    
    def transform_code(self, line):
        lines2idx = [self.code_corpus[token] for token in line.split(' ')]
        pad_id = self.code_corpus['<pad>']
        if len(lines2idx) > self.params['code_max_line_len']:
            lines2idx = lines2idx[:self.params['code_max_line_len']]
        elif len(lines2idx) < self.params['code_max_line_len']:
            lines2idx += [pad_id] * (self.params['code_max_line_len']-len(lines2idx))
        return lines2idx
    
    def process_msg(self, data):
        sent2idx = [self.msg_sp.encode(sent) for sent in data]
        text = []
        [text.extend(sent) for sent in sent2idx]
        pad_id = self.msg_corpus['<pad>']
        if len(text) > self.params['msg_max_len']:
            text = text[:self.params['msg_max_len']]
        elif len(text) < self.params['msg_max_len']:
            text += [pad_id] * (self.params['msg_max_len']-len(text))
        return text
    
    def generate_corpus(self, data):
        sentences = [line.split(' ') for lines in data for line in lines if line != '']
        corpus = build_vocab_from_iterator(
            iterator=sentences,
            min_freq=3,
            specials=['<unk>', '<pad>']
        )
        corpus.set_default_index(corpus['<unk>'])
        return corpus
    
class CodeAndMsgDataset2(torch.utils.data.Dataset):
    def __init__(self, data_filepath, params, kind='train'):
        self.params = params
        self.df = pd.read_csv(data_filepath, header=0, index_col=0)
        
        self.df = self.df[self.df['kinds'] == kind]
        self.df.reset_index(inplace=True, drop=True)
        self.commit = self.df['commit_id']
        self.code_data = self.df['code_data'].apply(eval)
        self.msg_data = self.df['msg_data'].apply(eval)
        self.metrics_data = self.df[['num_added', 'num_deleted']]
        self.labels = self.df['label']
        self.msg_sp = SpmProcessor()
        self.msg_sp.initializer(model_file=self.params['msg_spm_modelpath'])
        self.msg_corpus = self.msg_sp.generate_vocab()
        self.code_sp = SpmProcessor()
        self.code_sp.initializer(model_file=self.params['code_spm_modelpath'])
        self.code_corpus = self.msg_sp.generate_vocab()
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        code_feature = self.process_code(self.code_data[idx]) 
        msg_feature = self.process_msg(self.msg_data[idx])
        code_feature = torch.LongTensor(np.array(code_feature, dtype=int))
        msg_feature = torch.LongTensor(np.array(msg_feature, dtype=int))
        metrics_feature = torch.FloatTensor(self.metrics_data[idx])
        labels =  torch.as_tensor(int(self.labels[idx]), dtype=torch.long)
        return code_feature, msg_feature, metrics_feature, labels
    
    def process_code(self, data):
        features_list = [self.transform_code(line) for line in data]
        lines_mask = np.zeros(self.params['code_max_lines'])
        if len(features_list) > self.max_lines:
            features_list = features_list[:self.params['code_max_lines']]
        elif len(features_list) < self.max_lines:
            [features_list.extend([[self.corpus['<pad>']]*self.params['code_max_line_len']]) for i in range(self.params['code_max_lines']-len(features_list))]
            lines_mask[len(data):] = 1
        return features_list
    
    def transform_code(self, line):
        lines2idx = self.code_sp.encode(line)
        pad_id = self.code_corpus['<pad>']
        if len(lines2idx) > self.params['code_max_line_len']:
            lines2idx = lines2idx[:self.params['code_max_line_len']]
        elif len(lines2idx) < self.params['code_max_line_len']:
            lines2idx += [pad_id] * (self.params['code_max_line_len']-len(lines2idx))
        return lines2idx
    
    def process_msg(self, data):
        sent2idx = [self.msg_sp.encode(sent) for sent in data]
        text = []
        [text.extend(sent) for sent in sent2idx]
        pad_id = self.msg_corpus['<pad>']
        if len(text) > self.params['msg_max_len']:
            text = text[:self.params['msg_max_len']]
        elif len(text) < self.params['msg_max_len']:
            text += [pad_id] * (self.params['msg_max_len']-len(text))
        return text
    
    def generate_corpus(self, data):
        sentences = [line.split(' ') for lines in data for line in lines if line != '']
        corpus = build_vocab_from_iterator(
            iterator=sentences,
            min_freq=3,
            specials=['<unk>', '<pad>']
        )
        corpus.set_default_index(corpus['<unk>'])
        return corpus