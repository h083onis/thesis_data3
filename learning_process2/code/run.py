import argparse
import json
import pickle
import torch
import random
import numpy as np
from runner import Runner
from model_lgb import ModelLGB
from model_cnn import ModelCNN
from model_tf import ModelTF
from make_dataset import ProcessData

# PyTorch乱数固定用

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
    parser.add_argument('-i_c_dict','--input_code_dict', type=str)
    parser.add_argument('-i_m_dict','--input_msg_dict', type=str)
    parser.add_argument('-i_tr_data', '--input_train_data', type=str, required=True)
    parser.add_argument('-i_te_data', '--input_test_data', type=str, required=True)
    parser.add_argument('-s', '--setting', choices=['lgb','code_cnn','msg_tf', 'ensemble'], required=True)
    parser.add_argument('-c', '--cv_type', choices=['time','random'], required=True)
    parser.add_argument('-p', '--project', choices=['openstack','qt'], required=True)
    parser.add_argument('--max_code_len', type=int, default=32)
    parser.add_argument('--max_code_lines', type=int, default=64)
    parser.add_argument('--max_msg_len', type=int, default=512)
    return parser
    
if __name__ == '__main__':
    #不均衡対策openstack lgb
    # openstack_params_lgb = {
    #     'objective': 'binary',
    #     'metric': 'binary_logloss',
    #     'verbosity': -1,
    #     'feature_pre_filter': False,
    #     'lambda_l1': 2.808e-06,
    #     'lambda_l2': 4.025e-07,
    #     'num_leaves': 3,
    #     'feature_fraction': 0.4,
    #     'bagging_fraction': 0.9875,
    #     'bagging_freq': 1,
    #     'min_child_samples': 20,
    #     'random_state': 100,
    #     'num_iterations': 10000,
    #     'early_stopping_rounds': 10,
    # }
    
    openstack_params_lgb = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.01,
        'verbosity': -1,
        'num_iterations': 10000,
        'feature_pre_filter': False,
        'lambda_l1': 0.01,
        'lambda_l2': 0.0006,
        'num_leaves': 4,
        'feature_fraction': 0.48,
        'bagging_fraction': 0.68,
        'bagging_freq': 3, 
        'min_child_samples': 50,
        'random_state': 100,
        'early_stopping_rounds': 10,
    }
 
    #不均衡対策qt lgb
    # qt_params_lgb ={
    #     'objective': 'binary',
    #     'metric': 'binary_logloss',
    #     'learning_rate':0.1,
    #     'verbosity': -1,
    #     'feature_pre_filter': False,
    #     'lambda_l1': 0.006,
    #     'lambda_l2': 0.004,
    #     'num_leaves': 8,
    #     'feature_fraction': 0.4,
    #     'bagging_fraction': 0.866,
    #     'bagging_freq': 6,
    #     'min_child_samples': 50,
    #     'random_state': 100,
    #     'num_iterations': 1000,
    #     'early_stopping_rounds': 10,
    # }
    qt_params_lgb = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.01,
        'verbosity': -1,
        'random_state': 100,
        'num_iterations': 10000,
        'early_stopping_rounds': 10,
        'feature_pre_filter': False,
        'lambda_l1': 0.02,
        'lambda_l2': 8.48,
        'num_leaves': 9,
        'feature_fraction': 0.4,
        'bagging_fraction': 0.82,
        'bagging_freq': 1,
        'min_child_samples': 50,
        'early_stopping_rounds': 10,
    }

    params_tf ={
       'batch_size': 64,
       'lr' : 0.0001,
       'weight_decay':0.001,
       'd_model':256,
       'nhead':8,
       'nlayers':3,
       'd_hid':256,
       'n_classes': 2,
       'dropout':0.5,
        'epochs':100,
        'patience':5,
    }
    params_cnn ={
        'batch_size': 64,
        'lr' : 0.0001,
        'weight_decay':0.00001,
        'dim':64,
        'n_filters':64,
        'filter_sizes': [1,2,3],
        'hid':512,
        'n_classes': 2,
        'dropout':0.5,
        'epochs':100,
        'patience':5,
    }
    
    openstack_params_ensemble={
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'feature_pre_filter': False,
        'lambda_l1': 0.5564294241223428,
        'lambda_l2': 0.0003146523379419709,
        'num_leaves': 3,
        'feature_fraction': 1.0,
        'bagging_fraction': 0.9858922751119072,
        'bagging_freq': 1,
        'min_child_samples': 20,
        'random_state': 100,
        'num_iterations': 10000,
        'early_stopping_rounds': 10,
    }
    
    qt_params_ensemble ={
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'feature_pre_filter': False,
        'lambda_l1': 1.0361489371157912e-06,
        'lambda_l2': 3.4322608718712727e-06,
        'num_leaves': 3,
        'feature_fraction': 0.8999999999999999,
        'bagging_fraction': 0.6388522327010289,
        'bagging_freq': 1,
        'min_child_samples': 50,
        'random_state': 100,
        'num_iterations': 10000,
        'early_stopping_rounds': 10,
    }

    
    # 乱数初期化
    torch_seed()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    params = read_args().parse_args()
    pd = ProcessData(params)
    tr_dataset, te_dataset, dict_len =  pd.make_dataset()
    
    if params.project == 'openstack':
        params_lgb = openstack_params_lgb
        params_ensemble = openstack_params_ensemble
    else:
        params_lgb = qt_params_lgb
        params_ensemble = qt_params_ensemble
        
    if params.setting == 'lgb':
        runner = Runner(params.project+'-'+params.setting, ModelLGB, tr_dataset, te_dataset, params_lgb, device)
        runner.run_train_cv(params.cv_type)
        runner.run_predict_cv()
    elif params.setting == 'code_cnn':
        params_cnn['ntokens'] = dict_len
        runner = Runner(params.project+'-'+params.setting, ModelCNN, tr_dataset, te_dataset, params_cnn, device)
        runner.run_train_cv(params.cv_type)
        runner.run_predict_cv()
    elif params.setting == 'msg_tf':
        params_tf['ntokens'] = dict_len
        runner = Runner(params.project+'-'+params.setting, ModelTF, tr_dataset, te_dataset, params_tf, device)
        runner.run_train_cv(params.cv_type)
        runner.run_predict_cv()
    elif params.setting == 'ensemble':
        runner = Runner(params.project+'-'+params.setting, ModelLGB, tr_dataset, te_dataset, params_ensemble, device)
        runner.run_train_cv(params.cv_type)
        runner.run_predict_cv()
    else:
        print('setting error')