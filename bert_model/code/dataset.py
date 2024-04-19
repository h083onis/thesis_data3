from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
import torch

class BertDataset(Dataset):
    def __init__(self, data_filepath, params, kind='train'):
        self.params = params
        self.df = pd.read_csv(data_filepath, header=0, index_col=0)
        self.df = self.df[self.df['kinds'] == kind]
        self.df.reset_index(inplace=True, drop=True)
        self.commit = self.df['commit_id']
        self.msg_data = self.df['msg'].apply(eval)
        self.labels = self.df['label']
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        special_tokens_dict = {'additional_special_tokens': ['<id>', '<num>']}
        self.tokenizer.add_special_tokens(
            special_tokens_dict=special_tokens_dict)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        encoded_dict = self.tokenizer.encode_plus(
            '[CLS] ' + ' [SEP] '.join(self.msg_data[idx]) + ' [SEP]',
            add_special_tokens = False,
            max_length = self.params['max_len'],
            pad_to_max_length = True,
            return_attention_mask = True,
            return_tensors='pt',
        )
        input_ids = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask']
        label = torch.as_tensor(int(self.labels[idx]), dtype=torch.long)
        return input_ids, attention_mask, label