import argparse
from data_processor import DataProcessor
import json
import pandas as pd
import re
from sacrebleu.tokenizers.tokenizer_intl import TokenizerV14International


def add_lines(msg_list, code_list, project_data, test_rate=0.1, df=None):
    for i, commit in enumerate(project_data):
        if not isinstance(df, pd.DataFrame):
            if i < len(project_data) - test_rate * len(project_data):
                commit['kinds'] = 'train'
                code_list.extend(commit['codes'])
                msg_list.extend(commit['msg'])
            else:
                commit['kinds'] = 'test'
        else:
            if df.at[commit['commit_id'], 'strata'] != df['strata'].max():
                commit['kinds'] = 'train'
                code_list.extend(commit['codes'])
                msg_list.extend(commit['msg'])
            else:
                commit['kinds'] = 'test'
                
    
def integrate_line(project, processor):
    path = '../resource/preprocess_data/openstack_no_tokenize/'+project
    with open(path+'.json', 'r', encoding='utf-8') as f, open(path+'.log', 'r', encoding='utf-8') as f_log:
        project_data  = json.load(f)
        log = f_log.readlines()
    commit_id_list = [commit_id.split(' - ')[1].split('\t')[0] for commit_id in log]
    project_data2 = []
    tokenizer = TokenizerV14International()
    for i, commit in enumerate(project_data):
        print(project, i)
        if commit['codes'] == [] or commit['commit_id'] in commit_id_list:
            continue
        commit_dict = {}
        commit_dict['commit_id'] = commit['commit_id']
        commit_dict['timestamp'] = commit['timestamp']
        commit_dict['msg'] = processor.process_msg(commit['msg'])
        commit_dict['codes'] = []
 
        for file in commit['codes']:
            filepath = tokenizer(file['filepath']).lower()
            filepath = re.sub('\d+', '<num>', filepath)
            for key, item in list(file.items())[1:]:
                if item != []:
                    # lines = [processor.process(file['filepath'], file['filepath']) + ' <'+key+'> '+ processor.process(line, file['filepath']) for line in item]
                    lines = [filepath + ' <'+key+'> '+ processor.process_code(line, file['filepath']) for line in item]
                    # lines = ['<'+key+'> '+ processor.process(line) for line in item]
                commit_dict['codes'].extend(lines)
                # print(commit_dict['codes'])
        project_data2.append(commit_dict)
    return project_data2
        

def main(params):
    processor = DataProcessor(params.main_project)
    msg_list = []
    code_list = []
    project_data = []
    # for project in params.project_list:
    #     project_data = integrate_line(project, processor)
    #     add_lines(lines_list, project_data)
    #     processor.data_to_csv(project_data, project)
        
    df = pd.read_csv('../resource/'+params.main_project+'.csv', index_col=0, header=0)
    df.set_index('commit_id', inplace=True)   
    project_data = integrate_line(params.main_project, processor)
    add_lines(msg_list, code_list, project_data, df=df)
    processor.data_to_csv(project_data, params.main_project)
    params.model_prefix = params.main_project + '_msg_' + str(params.vocab_size)
    params.user_defined_symbols = params.msg_user_defined_symbols
    processor.train_spm(msg_list, params)
    params.model_prefix = params.main_project + '_code_' + str(params.vocab_size)
    params.user_defined_symbols = params.code_user_defined_symbols
    processor.train_spm(code_list, params)
        
    
def read_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-json_file', type=str)
    parser.add_argument('-model_type', type=str, default='unigram')
    # parser.add_argument('-model_prefix', type=str, default='openstack_spm_4000')
    parser.add_argument('-vocab_size', type=int, default=2000)
    parser.add_argument('-character_coverage', type=int, default=1.0)
    # parser.add_argument('-user_defined_symbols', type=list, default=['<pad>','<cpp>','<java>','<python>','<sep>','<num>','<literal>','<added_code>','<deleted_code>'])
    parser.add_argument('-code_user_defined_symbols', type=list, default=['<pad>', '<num>','<literal>','<added_code>','<deleted_code>'])
    parser.add_argument('-msg_user_defined_symbols', type=list, default=['<pad>', '<num>', '<id>'])
    parser.add_argument('-type', type=str, default='code')
    parser.add_argument('-main_project', type=str, default='openstack')
    parser.add_argument('-project_list', type=list, default=['django', 'ansible','libcloud'])
    parser.add_argument('-test_rate', type=int, default=0.1)
    return parser


if __name__ == '__main__':
    params = read_args().parse_args()
    main(params)