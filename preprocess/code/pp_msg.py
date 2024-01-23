import json
import sys
from clean_msg import CleanMsg

def excute(file, project):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    msg_data = {commit['commit_id']:commit['msg'] for commit in data}
    cm = CleanMsg(project)
    msg_list = []
    for id, msg in msg_data.items():
        msg_dict = {}
        msg_dict['commit_id'] = id
        msg_dict['text'] = cm.excute(msg)
        msg_list.append(msg_dict)
    with open(project+'3.json', 'w') as f:
        json.dump(msg_list, f, indent=2)

if __name__ == '__main__':
    file = sys.argv[1]
    project = sys.argv[2]
    excute(file, project)