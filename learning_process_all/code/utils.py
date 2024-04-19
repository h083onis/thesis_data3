import torch
import numpy as np

import datetime
import logging
import os
import pandas as pd


class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=5, verbose=False, path='checkpoint_model.pth'):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""
        self.logger = Logger()
        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.Inf   #前回の損失記憶用
        self.auc_max = 0   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path

    def __call__(self, val_loss, auc_score, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = auc_score
        loss = -val_loss
        
        if self.best_score is None:  #1Epoch目の処理
            self.best_score = auc_score   #1Epoch目はそのままベストスコアとして記録する
            self.val_loss_min = loss
            self.checkpoint(auc_score, model)  #記録後にモデルを保存してスコア表示する
        elif loss < self.val_loss_min:  # lossを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            
            if self.verbose:  #表示を有効にした場合は経過を表示
                self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:
            self.val_loss_min = loss  #lossを上書き
            self.counter = 0  #ストップカウンタリセット
            
        if self.best_score < score:
            #ベストスコアを更新した場合
            self.best_score = auc_score  #ベストスコアを上書き
            self.checkpoint(auc_score, model)  #モデルを保存してスコア表示

    def checkpoint(self, auc_score, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            self.logger.info(f'AUC score increased ({self.auc_max:.6f} --> {auc_score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.auc_max = auc_score  #その時のlossを記録する

class AUCEarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=5, verbose=False, path='checkpoint_model.pth'):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""
        self.logger = Logger()
        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        # self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.auc_max = 0   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path

    def __call__(self, auc_score, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = auc_score

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = auc_score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(auc_score, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = auc_score  #ベストスコアを上書き
            self.checkpoint(auc_score, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, auc_score, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            self.logger.info(f'AUC score increased ({self.auc_max:.6f} --> {auc_score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.auc_max = auc_score  #その時のlossを記録する
        
class Logger:

    def __init__(self):
        self.general_logger = logging.getLogger('general')
        self.result_logger = logging.getLogger('result')
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler('../log/general.log')
        file_result_handler = logging.FileHandler('../log/result.log')
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)
            self.result_logger.addHandler(stream_handler)
            self.result_logger.addHandler(file_result_handler)
            self.result_logger.setLevel(logging.INFO)

    def info(self, message):
        # 時刻をつけてコンソールとログに出力
        self.general_logger.info('[{}] - {}'.format(self.now_string(), message))

    def result(self, message):
        self.result_logger.info(message)

    def result_ltsv(self, dic):
        self.result(self.to_ltsv(dic))

    def result_scores(self, run_name, scores):
        # 計算結果をコンソールと計算結果用ログに出力
        dic = dict()
        dic['name'] = run_name
        dic['score'] = np.mean(scores)
        for i, score in enumerate(scores):
            dic[f'score{i}'] = score
        self.result(self.to_ltsv(dic))

    def now_string(self):
        return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def to_ltsv(self, dic):
        return '\t'.join(['{}:{}'.format(key, value) for key, value in dic.items()])