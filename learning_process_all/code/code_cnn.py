import torch.nn as nn
import torch
import torch.nn.functional as F


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

        # CNN-2D for commit code
        self.embed_code = nn.Embedding(V_code, Dim)
        self.convs_code_line = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])
        self.convs_code_file = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Co * len(Ks))) for K in Ks])

        # other information
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, hid)  # hidden units
        self.fc2 = nn.Linear(hid, Class)

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

    def forward(self, code):
        x_code = self.embed_code(code)
        x_code = self.forward_code(x_code, self.convs_code_line, self.convs_code_file)
        x_code = self.dropout(x_code)
        out = self.fc1(x_code)
        out = F.relu(out)
        out = self.fc2(out)
        return out