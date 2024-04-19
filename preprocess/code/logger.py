import datetime
import logging

class Logger:
    def __init__(self, log_file='general'):
        self.general_logger = logging.getLogger('general')
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler(str(log_file)+'.log')
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)

    def info(self, message):
        # 時刻をつけてコンソールとログに出力
        self.general_logger.info('[{}] \t {}'.format(self.now_string(), message))

    def now_string(self):
        return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    