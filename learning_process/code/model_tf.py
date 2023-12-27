import os

import numpy as np
import torch
from torch import nn, Tensor
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Model
from transformer import TransformerModel
from dataset import TFDataset
from utils import AUCEarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
from utils import Logger
import gc

logger = Logger()

class ModelTF(Model):
    
    def train(self, tr_dataset, va_dataset):
        # データのセット
        # ハイパーパラメータの設定
        tr_dataset2 = TFDataset(tr_dataset)
        tr_loader = DataLoader(tr_dataset2, batch_size=self.params['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
        va_dataset2 = TFDataset(va_dataset)
        va_loader = DataLoader(va_dataset2, batch_size=self.params['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
        self.va_dataset = [torch.LongTensor(va_dataset[0]).to(self.device), torch.BoolTensor(va_dataset[1]).to(self.device), va_dataset[2]]
        print(self.va_dataset[0].shape, self.va_dataset[1].shape)
        
        self.model = TransformerModel(self.params, self.device).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.params['lr'], weight_decay=self.params['weight_decay'])
        weights = compute_class_weight(class_weight='balanced', classes=np.unique(tr_dataset[2]), y=tr_dataset[2])
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(self.device))
        earlystopping = AUCEarlyStopping(patience=self.params['patience'], verbose=True, path='../best_model/'+self.run_fold_name+'.pth')
        
        self.fit(tr_loader, va_loader, optimizer, criterion, earlystopping)
        model_path = os.path.join('../best_model', f'{self.run_fold_name}.pth')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model = TransformerModel(self.params, self.device).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        
    def predict(self, te_dataset):
        self.model.eval()
        with torch.no_grad():
            te_dataset2 = [torch.LongTensor(te_dataset[0]).to(self.device), torch.BoolTensor(te_dataset[1]).to(self.device)]
            outputs_test = self.model(te_dataset2)
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        return nn.Softmax(dim = 1)(outputs_test).cpu().numpy()[:,1]
    
    def load_model(self):
        model_path = os.path.join('../best_model', f'{self.run_fold_name}.pth')
        self.model = TransformerModel(self.params, self.device).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        
    def fit(self, tr_loader, va_loader, optimizer, criterion, earlystopping):
        from tqdm.notebook import tqdm
        for epoch in range(self.params['epochs']):
            # 1エポックあたりの正解数(精度計算用)
            n_train_acc, n_val_acc = 0, 0
            # 1エポックあたりの累積損失(平均化前)
            train_loss, val_loss = 0, 0
            # 1エポックあたりのデータ累積件数
            n_train, n_valid = 0, 0

            auc_score = 0

            #訓練フェーズ
            self.model.train()
            for features, masks, labels in tqdm(tr_loader):
                # 1バッチあたりのデータ件数
                train_batch_size = len(labels)
                # 1エポックあたりのデータ累積件数
                n_train += train_batch_size
                # GPUヘ転送
                features = features.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)

                # 勾配の初期化
                optimizer.zero_grad()

                # 予測計算
                outputs = self.model([features, masks])

                # 損失計算
                loss = criterion(outputs, labels)

                # 勾配計算
                loss.backward()

                # パラメータ修正
                optimizer.step()

                # 予測ラベル導出
                predicted = torch.max(outputs, 1)[1]

                # 平均前の損失と正解数の計算
                # lossは平均計算が行われているので平均前の損失に戻して加算
                train_loss += loss.item() * train_batch_size 
                n_train_acc += (predicted == labels).sum().item()

            self.model.eval()
            with torch.no_grad():
                for features_valid, masks_valid, labels_valid in va_loader:
                    # 1バッチあたりのデータ件数
                    valid_batch_size = len(labels_valid)
                    # 1エポックあたりのデータ累積件数
                    n_valid += valid_batch_size

                    # GPUヘ転送
                    features_valid = features_valid.to(self.device)
                    masks_valid = masks_valid.to(self.device)
                    labels_valid = labels_valid.to(self.device)

                    # 予測計算
                    outputs_valid = self.model([features_valid, masks_valid])

                    # 損失計算
                    loss_valid = criterion(outputs_valid, labels_valid)

                    # 予測ラベル導出
                    predicted_valid = torch.max(outputs_valid, 1)[1]

                    #  平均前の損失と正解数の計算
                    # lossは平均計算が行われているので平均前の損失に戻して加算
                    val_loss +=  loss_valid.item() * valid_batch_size
                    n_val_acc +=  (predicted_valid == labels_valid).sum().item()

            # 精度計算
            train_acc = n_train_acc / n_train
            val_acc = n_val_acc / n_valid
            # 損失計算
            avg_train_loss = train_loss / n_train
            avg_val_loss = val_loss / n_valid
            auc_score = self.cal_auc()

            logger.info(f'{self.run_fold_name} - epoch:{epoch}, train_loss:{avg_train_loss}, train_acc:{train_acc}, valid_loss:{avg_val_loss}, valid_acc:{val_acc}, auc_score:{auc_score}')
            earlystopping(auc_score, self.model) #callメソッド呼び出し
            if earlystopping.early_stop: 
                logger.info("Early Stopping")
                break
        del self.model
        torch.cuda.empty_cache()
        gc.collect()

    def cal_auc(self):
        self.model.eval()
        with torch.no_grad():
            outputs_test = self.model(self.va_dataset[:-1])
            pred = nn.Softmax(dim = 1)(outputs_test).cpu().numpy()[:,1]
        auc = roc_auc_score(self.va_dataset[-1], pred)
        return auc
