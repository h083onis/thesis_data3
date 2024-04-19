import argparse
from data_processor import DataProcessor
import json
import pandas as pd

def check_ext(filepath):
    ext = filepath.split('.')[-1].lower()
    try:
        if ext == 'py':
            return '<python>'
        elif ext == 'java':
            return '<java>'
        elif ext == 'cpp' or ext == 'hpp' or ext == 'c' or ext == 'h' \
            or ext == 'cxx' or ext == 'hxx':
            return '<cpp>'
        else:
            raise ValueError('Discovered not setting source file extention!')
    except ValueError as e:
        print(e)

  
def add_lines(lines_list, project_data, test_rate=0.1, df=None):
    for i, commit in enumerate(project_data):
        if not isinstance(df, pd.DataFrame):
            if i < len(project_data) - test_rate * len(project_data):
                commit['kinds'] = 'train'
                [lines_list.append(line) for file in commit['codes'] for line in file]
            else:
                commit['kinds'] = 'test'
        else:
            if df.at[commit['commit_id'], 'strata'] != df['strata'].max():
                commit['kinds'] = 'train'
                [lines_list.append(line) for file in commit['codes'] for line in file]
            else:
                commit['kinds'] = 'test'
                
        
    
def integrate_line(project, processor):
    path = '../resource/preprocess_data/openstack_no_replace/'+project
    with open(path+'.json', 'r', encoding='utf-8') as f, open(path+'.log', 'r', encoding='utf-8') as f_log:
        project_data  = json.load(f)
        log = f_log.readlines()
    commit_id_list = [commit_id.split(' - ')[1].split('\t')[0] for commit_id in log]
    project_data2 = []
    for i, commit in enumerate(project_data):
        print(project, i)
        if commit['codes'] == [] or commit['commit_id'] in commit_id_list:
            continue
        commit_dict = {}
        commit_dict['commit_id'] = commit['commit_id']
        commit_dict['timestamp'] = commit['timestamp']
        commit_dict['codes'] = []
 
        for file in commit['codes']:
            file_list = []
            for key, item in list(file.items())[1:]:
                if item != []:
                    lines = [processor.process(line, file['filepath']) for line in item]
                    source = check_ext(file['filepath']) + ' <'+key+'> ' + ' <sep> '.join(lines) + ' <sep>'
                    file_list.append(source)
                else:
                    file_list.append('')
            commit_dict['codes'].append(file_list)
        project_data2.append(commit_dict)
    return project_data2
        

def main(params):
    processor = DataProcessor(params)
    lines_list = []
    project_data = []
    for project in params.project_list:
        project_data = integrate_line(project, processor)
        add_lines(lines_list, project_data)
        processor.data_to_csv(project_data, project)
        
    df = pd.read_csv('../resource/'+params.main_project+'.csv', index_col=0, header=0)
    df.set_index('commit_id', inplace=True)   
    project_data = integrate_line(params.main_project, processor)
    add_lines(lines_list, project_data, df=df)
    processor.data_to_csv(project_data, params.main_project)
    
    # processor.train_spm(lines_list)
        
    
def read_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-json_file', type=str)
    parser.add_argument('-model_type', type=str, default='unigram')
    parser.add_argument('-model_prefix', type=str, default='openstack_spm_8000')
    parser.add_argument('-vocab_size', type=int, default=8000)
    parser.add_argument('-character_coverage', type=int, default=1.0)
    parser.add_argument('-user_defined_symbols', type=list, default=['<pad>','<cpp>','<java>','<python>','<sep>','<num>','<literal>','<added_code>','<deleted_code>'])
    parser.add_argument('-type', type=str, default='code')
    parser.add_argument('-main_project', type=str, default='openstack')
    parser.add_argument('-project_list', type=list, default=['django', 'ansible','libcloud'])
    parser.add_argument('-test_rate', type=int, default=0.1)
    return parser


if __name__ == '__main__':
    params = read_args().parse_args()
    main(params)