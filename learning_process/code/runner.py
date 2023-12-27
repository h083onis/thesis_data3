import numpy as np
import pandas as pd
from model import Model
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from typing import Callable, List, Optional, Tuple, Union
import torch
from utils import Logger
import pickle

logger = Logger()

class Runner:
    def __init__(self, run_name: str, model_cls: Callable[[str, dict], Model], tr_data: List, te_data: List, params: dict, device:torch.device):
        """コンストラクタ

        :param run_name: ランの名前
        :param model_cls: モデルのクラス
        :param features: 特徴量のリスト
        :param params: ハイパーパラメータ
        """
        self.run_name = run_name
        self.model_cls = model_cls
        self.tr_data = tr_data
        self.te_data = te_data
        self.params = params
        self.device = device
        # self.n_fold = list(set(tr_data[0]))

    def train_fold(self, i_fold: int, tr_idx:List[int], va_idx:List[int]) -> Tuple[
        Optional[np.array], Optional[float]]:
        """クロスバリデーションでのfoldを指定して学習・評価を行う

        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる

        :param i_fold: foldの番号
        :return: （モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア）のタプル
        """
        # 学習データの読込
 
        # 学習データ・バリデーションデータをセットする
        tr_dataset = [[self.tr_data[i][j] for j in tr_idx] for i in range(1, len(self.tr_data))]
        va_dataset = [[self.tr_data[i][j] for j in va_idx] for i in range(1, len(self.tr_data))]
        
        # 学習を行う
        model = self.build_model(i_fold)
        model.train(tr_dataset, va_dataset)

        # バリデーションデータへの予測・評価を行う
        va_pred = model.predict(va_dataset)
        loss_score = log_loss(va_dataset[-1], va_pred, eps=1e-15, normalize=True)
        auc_score = roc_auc_score(va_dataset[-1], va_pred)

        # 予測値、評価を返す
        return va_pred, [loss_score, auc_score]
            
    def run_train_cv(self, cv_type='random') -> None:
        """クロスバリデーションでの学習・評価を行う

        学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
        """
        logger.info(f'{self.run_name} - start training cv')

        loss_scores = []
        auc_scores = []
        va_idxes = []
        preds = []
        
        self.cv_type = cv_type
        
        if cv_type == 'random':
            n_fold = 5
            self.n_fold = [i for i in range(5)]
            kf = StratifiedKFold(n_fold, shuffle=True, random_state=100)
            idx_list = [[tr_idx.tolist(),va_idx.tolist()] for tr_idx, va_idx in kf.split(self.tr_data[1], self.tr_data[-1])]
        elif cv_type == 'time':
            self.n_fold = list(set(self.tr_data[0]))
        
        # 各foldで学習を行う
        for i_fold in self.n_fold:
            # 学習を行う
            if cv_type == 'random':
                tr_idx = idx_list[i_fold][0]
                va_idx = idx_list[i_fold][1]
            elif cv_type == 'time':
                tr_idx = [index for index, value in enumerate(self.tr_data[0]) if value != i_fold]
                va_idx = [index for index, value in enumerate(self.tr_data[0]) if value == i_fold]
            logger.info(f'{self.run_name} fold {i_fold} - start training')
            va_pred, score = self.train_fold(i_fold, tr_idx, va_idx)
            logger.info(f'{self.run_name} fold {i_fold} - end training - score {score}')
            
            # 結果を保持する
            va_idxes.append(va_idx)
            loss_scores.append(score[0])
            auc_scores.append(score[1])
            preds.append(va_pred)

        # 各foldの結果をまとめる
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]

        logger.info(f'{self.run_name} - end training cv - loss_score {np.mean(loss_scores)}')
        logger.info(f'{self.run_name} - end training cv - auc_score {np.mean(auc_scores)}')

        # 予測結果の保存
        pickle.dump(preds, open(f'../pred/{self.run_name}-{self.cv_type}-train.pkl', 'wb'))
        pickle.dump(loss_scores, open(f'../score/{self.run_name}-{self.cv_type}-train_loss.pkl', 'wb'))
        pickle.dump(auc_scores, open(f'../score/{self.run_name}-{self.cv_type}-train_auc.pkl', 'wb'))

        # 評価結果の保存
        logger.result_scores(self.run_name, loss_scores)
        logger.result_scores(self.run_name, auc_scores)

    def run_predict_cv(self) -> None:
        """クロスバリデーションで学習した各foldのモデルの平均により、テストデータの予測を行う

        あらかじめrun_train_cvを実行しておく必要がある
        """
        logger.info(f'{self.run_name} - start prediction cv')

        preds = []
        
        # 各foldのモデルで予測を行う
        for i_fold in self.n_fold:
            logger.info(f'{self.run_name} - start prediction fold:{i_fold}')
            model = self.build_model(i_fold)
            model.load_model()
            pred = model.predict(self.te_data)
            preds.append(pred)
            logger.info(f'{self.run_name} - end prediction fold:{i_fold}')

        # 予測の平均値を出力する
        pred_avg = np.mean(preds, axis=0)

        # 予測結果の保存
        pickle.dump(pred_avg, open(f'../pred/{self.run_name}-{self.cv_type}-test.pkl', 'wb'))

        logger.info(f'{self.run_name} - end prediction cv')

    def build_model(self, i_fold: Union[int, str]) -> Model:
        """クロスバリデーションでのfoldを指定して、モデルの作成を行う

        :param i_fold: foldの番号
        :return: モデルのインスタンス
        """
        # ラン名、fold、モデルのクラスからモデルを作成する
        run_fold_name = f'{self.run_name}-{self.cv_type}-{i_fold}'
        return self.model_cls(run_fold_name, self.params, self.device)

