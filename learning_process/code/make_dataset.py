import pandas as pd
import numpy as np
import json
import pickle
from typing import Optional, List, Union, Tuple

def padding(data:List, max_len:int, max_lines:int=None, type:str='msg') -> List:
    pad_data = []
    if type == 'msg':
        for value in data:
            if len(value) < max_len:
                pad_data.append(value + ['<pad>' for i in range(max_len-len(value))])
            elif len(value) > max_len:
                pad_data.append(value[:-(len(value)-max_len)])
            else:
                pad_data.append(value)
    else:
        for values in data:
            codes_list = []
            for i, value in enumerate(values, 1):
                if i > max_lines:
                    break
                if len(value) < max_len:
                    codes_list.append(value + ['<pad>' for i in range(max_len-len(value))])
                elif len(value) > max_len:
                    codes_list.append(value[:-(len(value)-max_len)])
                else:
                    codes_list.append(value)
            if len(codes_list) < max_lines:
                codes_list += [['<pad>' for i in range(max_len)] for i in range(max_lines-len(codes_list))]
            pad_data.append(codes_list)
    return pad_data

def assign_index(pad_data:List, vob_dict:dict, type:str='msg') -> List:
    word_list = []
    if type == 'msg':
        mask_list = []
        for value in pad_data:
            word_list2 = []
            mask_list2 = []
            for word in value:
                if word == '<pad>':
                    mask_list2.append(1)
                else:
                    mask_list2.append(0)
                if word in vob_dict.keys():
                    word_list2.append(vob_dict[word])
                else:
                    word_list2.append(vob_dict['<unk>'])
            word_list.append(word_list2)
            mask_list.append(mask_list2)
        return word_list, mask_list
    else:
        for values in pad_data:
            codes_list = [[vob_dict[word] if word in vob_dict.keys() else vob_dict['<unk>'] for word in lines] for lines in values]
            word_list.append(codes_list)                
        return word_list

def process_dataset(data:List, vob_dict:dict, params:dict, type:str) -> List:
    if type == 'msg':
        pad_data = padding(data, params.max_msg_len, type)
        word_list, mask_list = assign_index(pad_data, vob_dict, type)
        return word_list, mask_list
    else:
        pad_data = padding(data, params.max_code_len, params.max_code_lines, type)
        word_list = assign_index(pad_data, vob_dict, type)
        return word_list
    

class ProcessData():
    def __init__(self, params:dict) -> None:
        with open(params.input_train_data, 'rb') as f_train, open(params.input_test_data, 'rb') as f_test:
            tr_data = pickle.load(f_train)
            te_data = pickle.load(f_test)
        self.tr_data = tr_data
        self.te_data = te_data
        self.params = params
        
    def make_dataset(self) -> Tuple[List, List, Union[int, None]]:
        if self.params.setting == 'lgb':
            tr_data = [self.tr_data[1], self.tr_data[4], self.tr_data[5]]
            te_data = [self.te_data[4], self.te_data[5]]
            return tr_data, te_data, None
        elif self.params.setting == 'code_cnn':
            with open(self.params.input_code_dict, 'r') as f:
                vob_dict = json.load(f)
            tr_word_list = process_dataset(self.tr_data[3], vob_dict, self.params, type)
            te_word_list = process_dataset(self.te_data[3], vob_dict, self.params, type)
            tr_data = [self.tr_data[1], tr_word_list, self.tr_data[5]]
            te_data = [te_word_list, self.te_data[5]]
            return  tr_data, te_data, len(vob_dict)
        elif self.params.setting == 'msg_tf':
            with open(self.params.input_msg_dict, 'r') as f:
                vob_dict = json.load(f)
            tr_word_list, tr_mask_list = process_dataset(self.tr_data[2], vob_dict, self.params, type)
            te_word_list, te_mask_list = process_dataset(self.te_data[2], vob_dict, self.params, type)
            tr_data = [self.tr_data[1], tr_word_list, tr_mask_list, self.tr_data[5]]
            te_data = [te_word_list, te_mask_list, self.te_data[5]]
            return  tr_data, te_data, len(vob_dict)
        elif self.params.setting == 'ensemble':
            pred_list = []
            kinds = ['train', 'test']
            models = ['lgb', 'code_cnn', 'msg_tf']
            for kind in kinds:
                for model in models:
                    with open('../pred/'+self.params.project+'-'+model+'-'+self.params.cv_type+'-'+kind+'.pkl', 'rb') as f:
                        pred_list.append(pickle.load(f))
            tr_pred = [[pred_list[0][j], pred_list[1][j], pred_list[2][j]] for j in range(len(pred_list[0]))]
            te_pred = [[pred_list[3][j], pred_list[4][j], pred_list[5][j]] for j in range(len(pred_list[3]))]                
            tr_data = [self.tr_data[1], tr_pred, self.tr_data[5]]
            te_data = [te_pred, self.te_data[5]]
            return tr_data, te_data, None
            
            