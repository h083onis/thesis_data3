import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Optional, List
from torch.utils.data.dataset import Subset


class Model(metaclass=ABCMeta):

    def __init__(self, run_fold_name: str, params: dict, device) -> None:
        """コンストラクタ

        :param run_fold_name: ランの名前とfoldの番号を組み合わせた名前
        :param params: ハイパーパラメータ
        """
        self.run_fold_name = run_fold_name
        self.params = params
        self.device = device
        self.model = None

    @abstractmethod
    def train(self, tr_dataset: List, va_dataset: List) -> None:
        """モデルの学習を行い、学習済のモデルを保存する

        :param tr_dataset: 学習データ
        :param va_dataset: バリデーションデータ
        """
        pass

    @abstractmethod
    def predict(self, te_dataset: List) -> np.array:
        """学習済のモデルでの予測値を返す

        :param te_dataset: バリデーション or テストデータ
        :return: 予測値
        """
        pass

    @abstractmethod
    def load_model(self) -> None:
        """モデルの読み込みを行う"""
        pass