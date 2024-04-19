import time
import sys
import json
import subprocess
from git import Repo
from gitdb.exc import BadName
import argparse
import pandas as pd
from utils import out_code_dict, out_txt
from lang_processors3.clean_comment import remove_c_com_and_doc
from lang_processors3.py_tokenizer2 import py_tokenize
from lang_processors3.cpp_processor import CppProcessor
from lang_processors3.java_processor import JavaProcessor
from logger import Logger



def is_auth_ext(file_path, auth_ext):
    try:
        splited_file = file_path.split('.')
        if len(splited_file) >= 2 and splited_file[-1].lower() in auth_ext:
            return True
        else:
            return False
    except AttributeError:
        return False
    

def diff_texts(codes_dict):
    command = 'diff -B -w -u -0 ../resource/preprocess_data/before.txt ../resource/preprocess_data/after.txt'
    process = subprocess.run(command.split(), stdout=subprocess.PIPE)
    output = process.stdout
    code_dict = out_code_dict(output.decode('utf-8','ignore'))
    codes_dict['added_code'].extend(code_dict['added_code'])
    codes_dict['deleted_code'].extend(code_dict['deleted_code'])
    return codes_dict
        
        
def print_code(repo, hexsha, filepath, ext, type):
    if type == 'before':
        output = repo.git.show(hexsha+"~1"+':'+filepath)
    else:
        output = repo.git.show(hexsha+':'+filepath)
    if ext == 'py':
        output = py_tokenize(output)
        if output == False:
            return False
    elif ext == 'java':
        output = remove_c_com_and_doc(output)
        processor = JavaProcessor('lang_processors2/tree-sitter/')
        output = processor.tokenize_code(output)
    else:
        output = remove_c_com_and_doc(output)
        processor = CppProcessor('lang_processors2/tree-sitter/')
        output = processor.tokenize_code(output)
    out_txt('../resource/preprocess_data/'+type+'.txt', output)


def pipe_process(repo, commit, hexsha, params, commit_dict, logger, type):
    auth_ext = params.auth_ext.split(',')
    if type == 'first':
        for filepath in commit.stats.files:
            if is_auth_ext(filepath, auth_ext) == False:
                continue
            ext = filepath.split('.')[-1].lower()
            codes_dict = {}
            codes_dict['filepath'] = filepath
            codes_dict['added_code'] = []
            codes_dict['deleted_code'] = []
            with open('../resource/preprocess_data/before.txt','w', encoding='utf-8') as f:
                f.truncate(0)
          
            flag = print_code(repo, hexsha, filepath, ext, type='after')
            if flag == False:
                logger.info(hexsha, filepath)
                continue
            codes_dict = diff_texts(codes_dict)
            if codes_dict['added_code'] == [] and codes_dict['deleted_code'] == []:
                continue
            commit_dict['codes'].append(codes_dict)
    
    else:
        diff = commit.diff(hexsha)
        for item in diff:
            if is_auth_ext(item.b_path, auth_ext) == False:
                continue
            ext = item.b_path.split('.')[-1].lower()
            codes_dict = {}
            codes_dict['filepath'] = item.b_path
            codes_dict['added_code'] = []
            codes_dict['deleted_code'] = []
            ch_type = item.change_type
            if ch_type == 'M' or ch_type == 'R':
                flag = print_code(repo, hexsha, item.a_path, ext, type='before')
                flag2 = print_code(repo, hexsha, item.b_path, ext, type='after')
                if flag == False:
                    logger.info(hexsha + '\t'+ item.a_path)
                    continue
                elif flag2 == False:
                    logger.info(hexsha + '\t'+ item.a_path)
                    continue
            elif ch_type == 'A' or ch_type == 'C':
                with open('../resource/preprocess_data/before.txt','w', encoding='utf-8') as f:
                    f.truncate(0)
                flag = print_code(repo, hexsha, item.b_path, ext, type='after')
                if flag == False:
                    logger.info(hexsha + '\t'+ item.a_path)
                    continue
            else:
                continue
            
            codes_dict = diff_texts(codes_dict)
            if codes_dict['added_code'] == [] and codes_dict['deleted_code'] == []:
                continue
            commit_dict['codes'].append(codes_dict)

    return commit_dict


def excute(params):
    commit_list = []
    logger = Logger(params.project)
    df = pd.read_csv(params.csv_filename, index_col=0)
    repo_list = list(df['repo_name'].unique())
    for repo_name in repo_list:  
        repo = Repo('../../../../sample_data/repo/'+params.project+'/'+repo_name)
        id_list = df.loc[df['repo_name'] == repo_name, 'commit_id']
        for i, hexsha in enumerate(id_list):
            commit_dict = {}
            commit_dict['commit_id'] = hexsha
            commit = repo.commit(hexsha)
            commit_dict['timestamp'] = commit.authored_date
            commit_dict['msg'] = commit.message
            commit_dict['codes'] = []
            print(repo_name, i, hexsha)
            try:
                commit = repo.commit(hexsha+'~1')
                commit_dict = pipe_process(repo, commit, hexsha, params, commit_dict, logger, type='normal')
            except (IndexError, BadName):   
                commit_dict = pipe_process(repo, commit, hexsha, params, commit_dict, logger, type='first')
            commit_list.append(commit_dict)
            
    with open(params.json_name, 'w') as f:
        json.dump(commit_list, f, indent=2)
    
  
def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-csv_filename', type=str)
    parser.add_argument('-project', type=str, default='openstack')
    parser.add_argument('-json_name', type=str, default='openstack.json')
    parser.add_argument('-auth_ext', type=str, default='java,c,h,cpp,hpp,cxx,hxx,py')
    return parser


if __name__ == '__main__':
    params = read_args().parse_args()
    sys.setrecursionlimit(3000)
    excute(params)
    
    sys.exit(0)