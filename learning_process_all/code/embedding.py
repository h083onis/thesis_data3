import torch
import math
from torch import nn

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.transpose(0,1))

    #[batch,file,len,dim]->[batch,file,len,dim]
    def forward(self, size):
        return self.pe[:,:size,:]
    
class SegmentEmbedding(nn.Embedding):
    def __init__(self, num_sentence=3, embed_size=512, padding_idx=0):
        super().__init__(num_sentence, embed_size, padding_idx)