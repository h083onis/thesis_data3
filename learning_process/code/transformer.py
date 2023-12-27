import math
import torch
from torch import nn, Tensor
from typing import List
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.transpose(0,1))

    #[batch,len,dim]->[batch,len,dim]
    def forward(self, x):
        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, params:dict, device):
        super().__init__()
        self.params = params
        
        ntokens = params['ntokens']
        d_model = params['d_model']
        nhead= params['nhead']
        d_hid = params['d_hid']
        nlayers = params['nlayers']
        n_class = params['n_classes']
        dropout = params['dropout']
        self.device = device
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.encoder = nn.Embedding(ntokens, d_model)
        self.linear = nn.Linear(d_model, n_class)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, dataset:List[Tensor]) -> Tensor:
        """
        Args:
            dataset[0]: Transformerへの入力データ
            dataset[1]: 入力データにかけるマスク
        Returns:
            Transformerの出力
        """
        src = dataset[0]
        src_key_padding_mask = dataset[1]
        batch_class_token = self.class_token.expand(src.shape[0], -1, -1)
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = torch.cat([batch_class_token, src], dim=1)
        src = self.pos_encoder(src)
        src_key_padding_mask = torch.cat([torch.zeros(src_key_padding_mask.shape[0], 1, dtype=torch.bool, device=self.device), src_key_padding_mask], dim=1)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = output[:,0,:]
        output = self.linear(output)
        return output
