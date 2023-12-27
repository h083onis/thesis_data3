import os

import numpy as np
from model import Model
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

class ModelEnsemble(Model):
    def train(self, tr_dataset, va_dataset):
        weights = compute_class_weight(class_weight='balanced', classes=np.unique(tr_dataset[1]), y=tr_dataset[1])
        self.model = LogisticRegression(penalty='l2',          # 正則化項(L1正則化 or L2正則化が選択可能)
                           dual=False,            # Dual or primal
                           tol=0.0001,            # 計算を停止するための基準値
                           C=1.0,                 # 正則化の強さ
                           fit_intercept=True,    # バイアス項の計算要否
                           intercept_scaling=1,   # solver=‘liblinear’の際に有効なスケーリング基準値
                           class_weight=weights,     # クラスに付与された重み
                           random_state=None,     # 乱数シード
                           solver='lbfgs',        # ハイパーパラメータ探索アルゴリズム
                           max_iter=100,          # 最大イテレーション数
                           multi_class='auto',    # クラスラベルの分類問題（2値問題の場合'auto'を指定）
                           verbose=0,             # liblinearおよびlbfgsがsolverに指定されている場合、冗長性のためにverboseを任意の正の数に設定
                           warm_start=False,      # Trueの場合、モデル学習の初期化に前の呼出情報を利用
                           n_jobs=None,           # 学習時に並列して動かすスレッドの数
                           l1_ratio=None          # L1/L2正則化比率(penaltyでElastic Netを指定した場合のみ)
                          )
        self.model.fit(np.array(tr_dataset[0]), np.array(tr_dataset[1]))
        self.save_model()

    def predict(self, te_dataset):
        return self.model.predict(np.array(te_dataset[0]))
    
    def save_model(self):
        model_path = os.path.join('../best_model', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # best_ntree_limitが消えるのを防ぐため、pickleで保存することとした
        pickle.dump(self.model, open(model_path, 'wb'))
        
    def load_model(self):
        model_path = os.path.join('../best_model', f'{self.run_fold_name}.model')
        self.model = pickle.load(open(model_path, 'rb'))
        