from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from spm_processor import SpmProcessor

class CodeChangeDataset(Dataset):
    def __init__(self, data_filepath, spm_modelpath, max_file=5, max_line_len=512):
        self.df = pd.read_csv(data_filepath, header=0, index_col=0)
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
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features, segment_label, file_mask = self.process(self.data[idx])
        for feature in features:
            print(len(feature))
        return torch.LongTensor(np.array(features)), \
                torch.LongTensor(np.array(segment_label)), \
                torch.BoolTensor(file_mask), \
                torch.as_tensor(int(self.label[idx]), dtype=torch.long)
        
    def process(self, data):
        file_mask = np.zeros(self.max_file)
        if len(data) < self.max_file:
            file_mask[len(data):] = 1
            [data.append(['','']) for i in range(self.max_file-len(data))]
        elif len(data) > self.max_file:
            data = data[:self.max_file]
        input_data = []
        segment_data = []
        data_list = [input_data, segment_data]
        [data_list[i].append(result) for lines in data for i, result in enumerate(self.transform(lines))]
        return input_data, segment_data, file_mask
    
    def transform(self, lines):
        lines2idx_list = [self.sp.encode(line) for line in lines]
        print(len(lines2idx_list[0]), len(lines2idx_list[1]))
        if len(lines2idx_list[0]) >= self.max_line_len and len(lines2idx_list[1]) >= self.max_line_len:
            lines2idx_list = [lines2idx[:self.max_line_len] for lines2idx in lines2idx_list]
        elif len(lines2idx_list[0]) < self.max_line_len and len(lines2idx_list[1]) > self.max_line_len:
            lines2idx_list[1] = lines2idx_list[1][:self.max_line_len+self.max_line_len-len(lines2idx_list[0])]
        elif len(lines2idx_list[0]) > self.max_line_len and len(lines2idx_list[1]) < self.max_line_len:
            lines2idx_list[0] = lines2idx_list[0][:self.max_line_len+self.max_line_len-len(lines2idx_list[1])]
        print(len(lines2idx_list[0]) , len(lines2idx_list[1]))
        segment_label = np.concatenate([np.full(len(lines2idx_list[i]), i+1, dtype=int) for i in range(0, len(lines2idx_list))])
        lines2idx = lines2idx_list[0] + lines2idx_list[1]
        pad_id = self.corpus['<pad>']
        if len(lines2idx) <  2 * self.max_line_len:
            lines2idx += [pad_id] * (2 * self.max_line_len-len(lines2idx))
            segment_label = np.concatenate([segment_label, np.zeros(2 * self.max_line_len-len(segment_label))])

        return lines2idx, segment_label
    
    # def label_segment(self, data):
    #     idx_list = [np.where(np.array(data,dtype=int) == idx)[0] for idx in self.code_type_idx]
    #     segment_label = np.zeros(len(data), dtype=int)
    #     [segment_label[:idx+1].fill(i) for i, idx in reversed(list(enumerate(idx_list, 1)))]
    #     return segment_label


# class CodeChangeDataset(Dataset):
#     def __init__(self, data_filepath, spm_modelpath, max_file=5, max_line_len=512):
#         self.df = pd.read_csv(data_filepath, header=0, index_col=0)
#         self.commit = self.df['commit_id']
#         self.data = self.df['input_data'].apply(eval)
#         self.label = self.df['label']
#         self.specials = ['<unk>','<s>','</s>','<pad>','<cpp>','<java>','<python>','<sep>']
#         self.max_file = max_file
#         self.max_line_len = max_line_len
#         self.sp = SpmProcessor()
#         self.sp.initializer(model_file=spm_modelpath)
#         self.vocabs = self.sp.generate_vocab()
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         features, segment_label, line_mask, file_mask = self.process(self.data[idx])
#         return torch.LongTensor(np.array(features)), \
#                 torch.LongTensor(np.array(segment_label)), \
#                 torch.BoolTensor(line_mask), \
#                 torch.BoolTensor(file_mask), \
#                 torch.as_tensor(int(self.label[idx]), dtype=torch.long)
        
#     def process(self, data):
#         file_mask = np.zeros(self.max_file)
#         if len(data) < self.max_file:
#             file_mask[self.max_file-len(data)-1:] = 1
#             [data.append(['','']) for i in range(self.max_file-len(data))]
#         elif len(data) > self.max_file:
#             data = data[:self.max_file]
#         input_data = []
#         line_mask = [[0 if line != '' else 1 for line in lines] for lines in data]
#         segment_data = []
#         data_list = [input_data, segment_data]
#         [data_list[i].append(result) for lines in data for i, result in enumerate(self.transform(lines))]
#         return input_data, segment_data, line_mask, file_mask
               
#     def transform(self, lines):
#         tokenized_list = []
#         segment_label_list = []
#         lines_list = [tokenized_list, segment_label_list]
#         [lines_list[i].append(result) for line in lines for i, result in enumerate(self.line2idx(line))]
#         return lines_list 
    
#     def line2idx(self, line):
#         line2idx = self.sp.encode(line)
        
#         pad_id = self.vocabs['<pad>']
#         if len(line2idx) < self.max_line_len:
#             line2idx += [pad_id] * (self.max_line_len-len(line2idx))
#             segment_label = self.label_segment(line2idx)
#         elif len(line2idx) > self.max_line_len:
#             segment_label = self.label_segment(line2idx)
#             line2idx = line2idx[:self.max_line_len]
#             segment_label = segment_label[:self.max_line_len]
#         return line2idx, segment_label
    
#     def label_segment(self, data):
#         sep_id = self.vocabs['<sep>']
#         idx_list = np.where(np.array(data,dtype=int) == sep_id)[0]
#         segment_label = np.zeros(len(data), dtype=int)
#         segment_label[:] = 0
#         [segment_label[:idx+1].fill(i) for i, idx in reversed(list(enumerate(idx_list, 1)))]
#         return segment_label

class CNNDataset(Dataset):
    def __init__(self, data_filepath, spm_modelpath, max_lines = 10, max_line_len=512, kind='train'):
        if not isinstance(data_filepath, pd.DataFrame):
            self.df = pd.read_csv(data_filepath, header=0, index_col=0)
            self.df = self.df[self.df['kinds'] == kind]
        else:
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
        features_list = []
        [features_list.extend(self.transform(lines)) for lines in data]
        if len(features_list) > self.max_lines:
            features_list = features_list[:self.max_lines]
        elif len(features_list) < self.max_lines:
            [features_list.extend([[self.corpus['<pad>']]*self.max_line_len]) for i in range(self.max_lines-len(features_list))]
        return features_list
    
    def transform(self, lines):
        lines2idx_list = [self.sp.encode(line) for line in lines]
        print(lines2idx_list)
        pad_id = self.corpus['<pad>']
        lines2idx_list2 = []
        for lines2idx in lines2idx_list:
            if lines2idx == []:
                continue
            if len(lines2idx) > self.max_line_len:
                lines2idx_list2.append(lines2idx[:self.max_line_len])
            elif len(lines2idx) < self.max_line_len:
                lines2idx += [pad_id] * (self.max_line_len-len(lines2idx))
                lines2idx_list2.append(lines2idx)
            else:
                lines2idx_list2.append(lines2idx)
        return lines2idx_list2
        
        