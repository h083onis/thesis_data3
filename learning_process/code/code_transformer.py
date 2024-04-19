import math
import torch
from torch import nn, Tensor
from encoder import TransformerEncoder
from embedding import PositionalEmbedding, SegmentEmbedding

class CodeChangeTransformer:
    def __init__(self, params, n_change_line_types=2):
        super().__init__()
        self.params = params
        
        n_tokens = params['n_tokens']
        d_model = params['d_model']
        n_head = params['n_head']
        d_hid = params['d_hid']
        n_layers = params['n_layers']
        n_class = params['n_classes']
        dropout = params['dropout']
        pad_id = params['pad_id']
        self.pad_id = pad_id
        self.file_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.files_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = PositionalEmbedding(d_model, dropout)
        self.seg_embedding = SegmentEmbedding(n_change_line_types, d_model, pad_id)
        self.file_encoder = TransformerEncoder(n_layers, d_model, d_hid, n_head, dropout)
        self.files_encoder = TransformerEncoder(n_layers, d_model, d_hid, n_head, dropout)
        self.d_model = d_model
        self.Embedding = nn.Embedding(n_tokens, d_model)
        self.linear = nn.Linear(d_model, n_class)
    
    def forward(self, dataset):
        src = dataset[0]
        src_segment_label = dataset[1]
        src_files_key_mask = dataset[3]
        
        self.batch_file_token = self.file_token(src.shape[0], -1, -1, -1) * math.sqrt(self.d_model)
        
        
        out = (self.Embedding(src) + self.seg_embedding(src_segment_label)) * math.sqrt(self.d_model)
        
        out = torch.cat([self.batch_file_token, out], dim=2)
        src_file_key_padding_mask = (src == self.pad_id).transpose(0,1).to(dataset.device)
        src_file_key_padding_mask = torch.cat([torch.zeros(src_file_key_padding_mask.shape[0],
                                                           src_file_key_padding_mask.shape[1],
                                                           1,
                                                           dtype=torch.bool,
                                                           device=dataset.device),
                                               src_file_key_padding_mask],
                                              dim=2)
        out = self.file_encoder(out, src_file_key_padding_mask)
        out = out[:,:,0,:]
        
        self.batch_files_token = self.files_token(out.shape[0], -1, -1) * math.sqrt(self.d_model)
        out = torch.cat([self.batch_files_token, out], dim=1)
        src_files_key_mask = torch.cat([torch.zeros(src_files_key_mask.shape[0], 1, dtype=torch.bool, device=dataset.device), src_files_key_mask], dim=1)
        out = self.files_encoder(out, src_files_key_mask)
        out = out[:,0,:]
        out = self.linear(out)
        return out
        
        
        
        