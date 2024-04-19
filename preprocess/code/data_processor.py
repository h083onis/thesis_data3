from spm_processor import SpmProcessor
from clean_msg import CleanMsg
from clean_code import CleanCode
import pandas as pd

class DataProcessor():
    def __init__(self, project=None):
        self.code_processor = CleanCode()
        self.msg_processor = CleanMsg(project)
        
    def process_code(self, data, filepath=None):
        cleaned_data = self.code_processor.excute(data, filepath)
        return cleaned_data
    
    def process_msg(self, data):
        cleaned_data = self.msg_processor.excute(data)
        return cleaned_data
        
    def train_spm(self, data, args):
        self.spm = SpmProcessor(args)
        self.spm.train(data)
        
    def data_to_csv(self, data, repo_name):
        data_dict = {key:[] for key in data[0].keys()}
        [data_dict[key].append(item) for commit in data for key, item in commit.items()]
        df = pd.DataFrame(
            data = data_dict,
            columns=list(data_dict.keys())
        )
        df.to_csv(repo_name+'.csv')
        
        # commit_id_list = []
        # timestamp_list = []
        # msg_data_list = []
        # code_data_list = []
        # kind_list = []
        # for commit in data:
        #     commit_id_list.append(commit['commit_id'])
        #     timestamp_list.append(commit['timestamp'])
        #     msg_data_list.append(commit['msg'])
        #     code_data_list.append(commit['codes'])
        #     kind_list.append(commit['kinds'])
        # df = pd.DataFrame(
        #     data={
        #         'commit_id':commit_id_list,
        #         'timestamp':timestamp_list,
        #         'msg_data':msg_data_list,
        #         'code_data':code_data_list,
        #         'kinds':kind_list
        #         },
        # columns=['commit_id','timestamp','msg_data','code_data','kinds']
        # )
        # df.to_csv(repo_name+'.csv')
        
        
    
        
    
            