import math
import torch
from torch import nn, Tensor
from cc_file_encoder import FileEncoder
from cc_files_encoder import FilesEncoder
from cc_line_encoder import LineEncoder
from embedding import PositionalEmbedding

class CodeChangeTransformer(nn.Module):
    def __init__(self, params, n_change_line_types=3):
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
        self.line_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.lines_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = PositionalEmbedding(d_model, dropout)
        self.line_encoder = FileEncoder(n_layers, d_model, d_hid, n_head, dropout)
        self.lines_encoder = FilesEncoder(n_layers, d_model, d_hid, n_head, dropout)
        self.d_model = d_model
        self.Embedding = nn.Embedding(n_tokens, d_model)
        self.linear = nn.Linear(d_model, n_class)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, dataset):
        src = dataset[0]
        print(src.shape)
        src_lines_key_mask = dataset[1]
        
        out = self.dropout((self.Embedding(src) + self.pos_embedding(src))) * math.sqrt(self.d_model)
        self.batch_file_token = self.line_token.expand(src.shape[0], src.shape[1], -1, -1) * math.sqrt(self.d_model)
        
        out = torch.cat([self.batch_file_token, out], dim=2)
        
        src_line_key_padding_mask = (src == self.pad_id).to(out.device)
        src_line_key_padding_mask = torch.cat([torch.zeros(src_line_key_padding_mask.shape[0],
                                                           src_line_key_padding_mask.shape[1],
                                                           1,
                                                           dtype=torch.bool,
                                                           device=out.device),
                                               src_line_key_padding_mask],
                                              dim=2)
        out = self.line_encoder(out, src_line_key_padding_mask)
        out = out[:,:,0,:]
        
        self.batch_lines_token = self.lines_token(out.shape[0], -1, -1) * math.sqrt(self.d_model)
        out = torch.cat([self.batch_lines_token, out], dim=1)
        src_lines_key_mask = torch.cat([torch.zeros(src_lines_key_mask.shape[0], 1, dtype=torch.bool, device=out.device), src_files_key_mask], dim=1)
        out = self.lines_encoder(out, src_lines_key_mask)
        out = out[:,0,:]
        out = self.linear(out)
        return out
        
        
        
        