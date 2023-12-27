import os

import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from model import Model

class ModelXGB(Model):

    def train(self, tr_dataset, va_dataset=None):
        # データのセット
        validation = va_dataset is not None

        dtrain = xgb.DMatrix(tr_dataset[0], label=tr_dataset[1])
        if validation:
            dvalid = xgb.DMatrix(va_dataset[0], label=va_dataset[1])

        # ハイパーパラメータの設定
        params = dict(self.params)
        num_round = params.pop('num_round')
        early_stopping_rounds = params.pop('early_stopping_rounds')
        # 学習
        if validation:
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
            self.model = xgb.train(params, dtrain, num_round, evals=watchlist,
                                   early_stopping_rounds=early_stopping_rounds)
        else:
            watchlist = [(dtrain, 'train')]
            self.model = xgb.train(params, dtrain, num_round, evals=watchlist,
                                   early_stopping_rounds=early_stopping_rounds)
        
        self.save_model()

    def predict(self, te_dataset):
        dtest = xgb.DMatrix(np.array(te_dataset[0]))
        return self.model.predict(dtest, iteration_range=(0, self.model.best_iteration))

    def save_model(self):
        model_path = os.path.join('../best_model', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # best_ntree_limitが消えるのを防ぐため、pickleで保存することとした
        pickle.dump(self.model, open(model_path, 'wb'))

    def load_model(self):
        model_path = os.path.join('../best_model', f'{self.run_fold_name}.model')
        self.model = pickle.load(open(model_path, 'rb'))