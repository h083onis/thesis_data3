import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from model import Model
import pickle
from sklearn.utils.class_weight import compute_sample_weight

class ModelLGB(Model):

    def train(self, tr_dataset, va_dataset):
        # データのセット  
        sample_weights = compute_sample_weight(class_weight='balanced', y=tr_dataset[1])
        dtrain = lgb.Dataset(np.array(tr_dataset[0]), np.array(tr_dataset[1]), weight=sample_weights)
        dvalid = lgb.Dataset(np.array(va_dataset[0]), np.array(va_dataset[1]))

        # ハイパーパラメータの設定
        params = dict(self.params)
        num_round = params.pop('num_iterations')
        early_stopping_rounds = params.pop('early_stopping_rounds')
        # 学習
        self.model = lgb.train(params, dtrain, num_boost_round = num_round, 
                                valid_names=['train','valid'], valid_sets=[dtrain, dvalid],
                                callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds)])
        df = pd.read_csv('../resource/feature_importance2.csv', header=0, index_col=0)
        df.loc[self.run_fold_name] = self.model.feature_importance(importance_type='gain')
        df.to_csv('../resource/feature_importance2.csv')
        self.save_model()

    def predict(self, te_dataset):
        return self.model.predict(np.array(te_dataset[0]), num_iteration=self.model.best_iteration)

    def save_model(self):
        model_path = os.path.join('../best_model', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # best_ntree_limitが消えるのを防ぐため、pickleで保存することとした
        pickle.dump(self.model, open(model_path, 'wb'))

    def load_model(self):
        model_path = os.path.join('../best_model', f'{self.run_fold_name}.model')
        self.model = pickle.load(open(model_path, 'rb'))