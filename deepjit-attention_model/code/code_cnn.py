import torch.nn as nn
import torch
import torch.nn.functional as F
from attention_layer import AttentionLayer



class CodeCNN(nn.Module):
    def __init__(self, params):
        super(CodeCNN, self).__init__()

        V_code = params['n_tokens']
        Dim = params['dim']
        Class = params['n_classes']        

        Ci = 1  # input of convolutional layer
        Co = params['n_filters']  # output of convolutional layer
        self.n_filters = Co
        Ks = params['filter_sizes']  # kernel sizes
        self.filter_sizes = Ks
        
        dropout = params['dropout']
        hid = params['hid']
        num_head = params['num_head']
        
        self.line_token = nn.Parameter(torch.randn(1, 1, Dim))
        conv_line_out_sizes = [(Dim-K)+1 for K in Ks]
        self.linears_code_line = nn.ModuleList([nn.Linear(size, Dim) for size in conv_line_out_sizes])
        self.line_attention_pool = AttentionLayer(Dim, num_head)
        
        self.lines_token = nn.Parameter(torch.randn(1, 1, Dim))
        conv_lines_out_sizes = [((Co*len(Ks)+Dim)-K)+1 for K in Ks]
        self.linears_code_line = nn.ModuleList([nn.Linear(size, Dim) for size in conv_lines_out_sizes])
        self.lines_attention_pool = AttentionLayer(Dim, num_head)

        # CNN-2D for commit code
        self.embed_code = nn.Embedding(V_code, Dim)
        self.convs_code_line = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])
        self.convs_code_lines = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Co * len(Ks) + Dim)) for K in Ks])

        
        # other information
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks) * Co + Dim, hid)  # hidden units
        self.fc2 = nn.Linear(hid, Class)
        
    def forward_line(self, x):
        # note that we can use this function for commit code line to get the information of the line
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs_code_line]  # [(N, Co, W), ...]*len(Ks)
        x_max_pool = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        out_max_pool = torch.cat(x_max_pool, 1)
        x_attention_matrix = [self.linears_code_line[i](w) for i, w in enumerate(x)]
        x_attention_matrix = torch.cat(x_attention_matrix, 1)
        batch_line_token = self.line_token.expand(x_attention_matrix.shape[0], -1, -1)
        x_attention_matrix = torch.cat([batch_line_token, x_attention_matrix], dim=1)
        x_attention_pool = self.line_attention_pool(x_attention_matrix)
        out_attention_pool = x_attention_pool[:,0,:]
        out = torch.cat(out_max_pool, out_attention_pool, 1)
        return out
    
    def forward_lines(self, x, convs):
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in convs]
        x_max_pool = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        out_max_pool = torch.cat(x_max_pool, 1)
        x_attention_matrix = [self.linears_code_line[i](w) for i, w in enumerate(x)]
        x_attention_matrix = torch.cat(x_attention_matrix, 1)
        batch_lines_token = self.lines_token.expand(x_attention_matrix.shape[0], -1, -1)
        x_attention_matrix = torch.cat([batch_lines_token, x_attention_matrix], dim=1)
        x_attention_pool = self.lines_attention_pool(x_attention_matrix)
        out_attention_pool = x_attention_pool[:,0,:]
        out = torch.cat(out_max_pool, out_attention_pool, 1)
        return out

    def forward_code(self, x):
        n_batch, n_file = x.shape[0], x.shape[1]
        x = x.reshape(n_batch * n_file, x.shape[2], x.shape[3])

        # apply cnn 2d for each line in a commit code
        x = self.forward_line(x)

        # apply cnn 2d for each file in a commit code
        x = x.reshape(n_batch, n_file, self.n_filters * len(self.filter_sizes))
        x = self.forward_lines(x)
        return x

    def forward(self, code):
        x_code = self.embed_code(code)
        x_code = self.forward_code(x_code)
        x_code = self.dropout(x_code)
        out = self.fc1(x_code)
        out = F.relu(out)
        out = self.fc2(out)
        return out