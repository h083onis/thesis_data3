import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules.normalization import LayerNorm

class SoftmaxAttention(nn.Module):
    def __init__(self, head_dim, dropout=0.1):
        super().__init__()
        self.head_dim = head_dim
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        #[batch, num_head, len, head_dim] 
        logit = torch.einsum('bhld, bhmd->bhlm', q, k) / math.sqrt(self.head_dim)
        print(logit.shape)
        if mask != None:
            logit = logit + mask
        # print(logit)
        attention_weight = F.softmax(logit, dim=-1)
        attention_weight = self.dropout(attention_weight)
        x = torch.einsum('bhlm, bhmd->bhld', attention_weight, v)
        
        return x
    
class SelfAttention(nn.Module):
    def __init__(self, dim, num_head):
        super().__init__()
        
        self.dim = dim
        self.num_head = num_head
        
        assert dim % num_head == 0, print("ASSERT #dim % num_head == 0")
        
        self.head_dim = int(dim / num_head)
        
        self.w_q = nn.Linear(self.dim, self.dim, bias=False)
        self.w_k = nn.Linear(self.dim, self.dim, bias=False)
        self.w_v = nn.Linear(self.dim, self.dim, bias=False)
        self.linear = nn.Linear(self.dim, self.dim, bias = False)
        
        self.attention = SoftmaxAttention(self.head_dim)
    
    def forward(self, x, mask):
        #[batch, len, dim]
        q = self.split_heads(self.w_q(x))
        k = self.split_heads(self.w_k(x))
        v = self.split_heads(self.w_v(x))
        
        attention_out = self.attention(q.float(), k.float(), v.float(), mask.float())
        attention_out = self.combine_heads(attention_out)
        return self.linear(attention_out)
    
    #[batch,len,dim]->[batch,num_head,len,head_dim]  
    def split_heads(self, x):
        #[batch,len,dim]->[batch,len,head,head_dim]
        x = x.reshape(x.size(0), x.size(1), self.num_head, self.head_dim)
        #[batch,len,num_head,head_dim]->[batch,num_head,len,head_dim] 
        x = x.transpose(1, 2)
        return x
    
    #[batch,head,len,num_head_dim]->[batch,len,dim]
    def combine_heads(self, x):
        x = x.transpose(1, 2)
        x = x.reshape(x.size(0), x.size(1), self.num_head * self.head_dim)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_head, hid, dropout):
        super().__init__()
        
        self.attention = SelfAttention(dim, num_head)
        self.feed = nn.Sequential(
                    nn.Linear(dim,hid),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                    nn.Linear(hid,dim),
                )
        self.norm = LayerNorm(dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x = self.dropout(self.attention(x, mask)) + x
        x = self.norm(x)
        x = self.dropout(self.feed(x)) + x
        x = self.norm(x)
        return x
    
class FilesEncoder(nn.Module):
    def __init__(self, num_layer, dim, hid, num_head, dropout=0.5):
        super().__init__()
        
        self.encoders = nn.ModuleList([TransformerEncoderLayer(dim, num_head, hid, dropout)for i in range(num_layer)])
        
    def forward(self, x, key_padding_mask=None):
        if key_padding_mask != None:
            mask = torch.zeros_like(key_padding_mask, dtype=torch.bool).to(x.device)
            mask = mask.masked_fill_(key_padding_mask, float("-inf"))
            # mask = mask.reshape(mask.shape[0], 1, 1, mask.shape[1])
        print(mask.shape)
        for layer in self.encoders:
            x = layer(x, mask=mask)
        return x
    #   def forward(self, x, key_padding_mask=None):
    #     if key_padding_mask != None:
    #         mask = torch.zeros(key_padding_mask.shape[0], key_padding_mask.shape[1]).to(x.device)
    #         mask = mask.masked_fill_(key_padding_mask, float("-inf"))
    #         mask = mask.reshape(mask.shape[0], 1, 1, mask.shape[1])
    #     # print(mask)
    #     for layer in self.encoders:
    #         x = layer(x, mask=mask)
    #     return x
            
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    