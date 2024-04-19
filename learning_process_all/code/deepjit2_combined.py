import torch.nn as nn
import torch
from embedding import PositionalEmbedding
from msg_encoder import TransformerEncoder
import torch.nn.functional as F
import math

class DeepJIT2(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        # code layers
        V_code = params['code_n_tokens']
        Dim = d_model = params['d_model']

        Ci = 1  # input of convolutional layer
        Co = params['n_filters']  # output of convolutional layer
        self.n_filters = Co
        Ks = params['filter_sizes']  # kernel sizes
        self.filter_sizes = Ks
        hid = params['hid']

        # CNN-2D for commit code
        self.embedding_code = nn.Embedding(V_code, Dim)
        self.convs_code_line = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])
        self.convs_code_file = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Co * len(Ks))) for K in Ks])

        # msg layers
        ntokens = params['msg_n_tokens']
        # d_model = params['d_model']
        nhead= params['n_head']
        d_hid = params['d_hid']
        nlayers = params['n_layers']
        n_class = params['n_classes']
        self.emb_dropout = nn.Dropout(params['msg_dropout'])
        self.pad_id = params['msg_pad_id']
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = PositionalEmbedding(d_model)
        self.transformer_encoder = TransformerEncoder(nlayers, d_model, d_hid, nhead, params['dropout'])
        self.d_model = d_model
        self.embedding_msg = nn.Embedding(ntokens, d_model)
        
        self.dropout = nn.Dropout(params['dropout'])
        self.fc1 = nn.Linear(len(Ks) * Co + d_model + params['n_metrics'], hid)
        self.fc2 = nn.Linear(hid, n_class)
        
    def forward_msg(self, x, convs):
        # note that we can use this function for commit code line to get the information of the line
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        return x

    def forward_code(self, x, convs_line, convs_hunks):
        n_batch, n_file = x.shape[0], x.shape[1]
        x = x.reshape(n_batch * n_file, x.shape[2], x.shape[3])

        # apply cnn 2d for each line in a commit code
        x = self.forward_msg(x=x, convs=convs_line)

        # apply cnn 2d for each file in a commit code
        x = x.reshape(n_batch, n_file, self.n_filters * len(self.filter_sizes))
        x = self.forward_msg(x=x, convs=convs_hunks)
        return x
    
    def forward(self, dataset):
        code = dataset[0]
        x_code = self.embedding_code(code)
        x_code = self.forward_code(x_code, self.convs_code_line, self.convs_code_file)
        
        x_msg = dataset[1]
        src_key_padding_mask = (x_msg == self.pad_id).to(x_msg.device)
        src_key_padding_mask = torch.cat([torch.zeros(src_key_padding_mask.shape[0], 1, dtype=torch.bool, device=x_msg.device), src_key_padding_mask], dim=1)
        pos = self.pos_embedding(x_msg.size(1))
        batch_class_token = self.class_token.expand(x_msg.shape[0], -1, -1)
        x_msg = self.emb_dropout(self.embedding_msg(x_msg) + pos)
        x_msg = torch.cat([batch_class_token, x_msg], dim=1) * math.sqrt(self.d_model)
        x_msg = self.transformer_encoder(x_msg, src_key_padding_mask)
        x_msg = x_msg[:,0,:]
        
        x_commit = torch.cat([x_code, x_msg, dataset[2]], dim=1)
        x_commit = self.dropout(x_commit)
        x_commit = self.fc1(x_commit)
        x_commit = F.relu(x_commit)
        x_commit = self.fc2(x_commit)
        return x_commit
        