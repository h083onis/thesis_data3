import argparse
import torch
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model_bert import ModelBert

def torch_seed(seed=100):
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.benchmark = False
    


def read_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-s', '--spm_modelfile', type=str)
    parser.add_argument('-i', '--input_datafile', type=str)
    parser.add_argument('-name', '--run_name', type=str)
    return parser

if __name__ == '__main__':
    params_cct ={
        'batch_size': 64,
        'lr' : 0.0001,
        'weight_decay':0.00001,
        'epochs':100,
        'patience':5,
        'max_len':512,
    }
    # 乱数初期化
    torch_seed()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    params = read_args().parse_args()
    model = ModelBert(params.run_name, params_cct, device)
    df = pd.read_csv(params.input_datafile, index_col=0, header=0)
    train_df = df[df['kinds']=='train']
    train_idx, valid_idx = train_test_split(train_df.index, test_size=0.1)
    model.train(df, train_idx, valid_idx)
    test_df = df[df[df['kinds']=='test']]
    pred = model.predict(test_df)
    pred_df = pd.DataFrame(
            data={
                'commit_id':test_df['commit_id'],
                'pred':pred
                },
            columns=['commit_id','pred']
        )
    pred_df.to_csv(params.name+'.csv')
    