import sys
import json
from git import Repo
from gitdb.exc import BadName
import argparse
from diff_commit4 import pipe_process
from logger import Logger

def excute(params):
    logger = Logger(params.project)
    repo = Repo('../../../../sample_data/repo/'+params.project)
    head = repo.head
    
    if head.is_detached:
        pointer = head.commit.hexsha
    else:
        pointer = head.reference
        
    commits = list(repo.iter_commits(pointer))
    commits.reverse()
    commit_list = []
    for i, item in enumerate(commits):
        commit_dict = {}
        commit_dict['commit_id'] = item.hexsha
        commit = repo.commit(item.hexsha)
        commit_dict['timestamp'] = commit.authored_date
        commit_dict['msg'] = commit.message
        commit_dict['codes'] = []
        print(i, item.hexsha)
        try:
            commit = repo.commit(item.hexsha+'~1')
            pipe_process(repo, commit, item.hexsha, params, commit_dict, logger, type='normal')
        except (IndexError, BadName):   
            pipe_process(repo, commit, item.hexsha, params, commit_dict, logger, type='first')
        commit_list.append(commit_dict)
   
        
    with open(params.json_name, 'w') as f:
        json.dump(commit_list, f, indent=2)
        
def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-project', type=str, default='libcloud')
    parser.add_argument('-json_name', type=str, default='libcloud.json')
    parser.add_argument('-auth_ext', type=str, default='java,c,h,cpp,hpp,cxx,hxx,py')
    return parser

if __name__ == '__main__':
    params = read_args().parse_args()
    excute(params)
    sys.exit(0)