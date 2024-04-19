import os

import numpy as np
import torch
from torch import nn, Tensor
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from model import Model
from deepjit2 import DeepJIT2
from dataset import CodeAndMsgDataset2
from utils import AUCEarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from utils import Logger
from custom_loss import FocalLoss
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, log_loss
from timm.scheduler import CosineLRScheduler
import gc
logger = Logger()

class ModelDeepJIT2(Model):
    def train(self, filepath, train_idx, valid_idx, kind='train'):
        # ハイパーパラメータの設定
        dataset = CodeAndMsgDataset2(
            filepath,
            kind=kind,
            params=self.params
        )
        self.params['msg_n_tokens'] = len(dataset.msg_corpus)
        self.params['code_n_tokens'] = len(dataset.code_corpus)
        self.params['msg_pad_id'] = dataset.msg_corpus['<pad>']
        train_dataset = Subset(dataset, train_idx)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.params['batch_size'],
            shuffle=True,
            drop_last = True,
            num_workers=2,
            pin_memory=True
        )
        valid_dataset = Subset(dataset, valid_idx)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.params['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        self.model = DeepJIT2(self.params).to(self.device)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.params['lr'],
            weight_decay=self.params['weight_decay']
        )
        train_labels = [dataset.labels[idx] for idx in train_idx]
        if self.params['is_weights'] == True:
            weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
        else:
            weights = [1,1]
        if self.params['criterion'] == 'fl':
            criterion = FocalLoss(alpha=torch.FloatTensor(weights).to(self.device), gamma=self.params['gamma'])
        else:
            criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(self.device))
        earlystopping = AUCEarlyStopping(
            patience=self.params['patience'],
            verbose=True,
            path='../best_model/'+self.run_fold_name+'.pth'
        )
        if self.params['scheduler'] == True:
            # scheduler = CosineLRScheduler(optimizer, t_initial=self.params['epochs'], lr_min=1e-6, 
            #                   warmup_t=5, warmup_lr_init=1e-6, warmup_prefix=True)
            scheduler = CosineLRScheduler(optimizer, t_initial=self.params['epochs'], lr_min=1e-6, 
                              warmup_t=5, warmup_lr_init=self.params['lr'], warmup_prefix=False)
        else:
            scheduler = None
        
        self.fit(train_loader, valid_loader, optimizer, criterion, earlystopping, scheduler)
    
    def predict(self, filepath, kind='test'):
        dataset = CodeAndMsgDataset2(
            filepath,
            kind=kind,
            params=self.params
        )
        test_loader = DataLoader(
            dataset,
            batch_size=self.params['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        self.params['msg_n_tokens'] = len(dataset.msg_corpus)
        self.params['code_n_tokens'] = len(dataset.code_corpus)
        self.params['msg_pad_id'] = dataset.msg_corpus['<pad>']
        model_path = os.path.join('../best_model', f'{self.run_fold_name}.pth')
        self.model = DeepJIT2(self.params).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        pred = np.zeros(0)
        labels_array = np.zeros(0)
        self.model.eval()
        with torch.no_grad():
            for code_features, msg_features, labels in test_loader:
                code_features = code_features.to(self.device)
                msg_features = msg_features.to(self.device)
                outputs = self.model([code_features, msg_features])
                pred = np.concatenate([pred, nn.Softmax(dim = 1)(outputs).cpu().numpy()[:,1]])
                labels_array = np.concatenate([labels_array, labels.cpu().numpy()])
        print(f'auc: {roc_auc_score(labels_array, pred):.3f}')
        print(f'log_loss: {log_loss(labels_array, pred):.3f}')
        answer = [1 if prob >= 0.5 else 0 for prob in pred]
        print(f'recall: {recall_score(labels_array, answer):.3f}')
        print(f'precision: {precision_score(labels_array, answer):.3f}')
        print(f'f1: {f1_score(labels_array, answer):.3f}')
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        return pred
        
    def load_model(self):
        model_path = os.path.join('../best_model', f'{self.run_fold_name}.pth')
        self.model = DeepJIT2(self.params).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        
    def fit(self, train_loader, valid_loader, optimizer, criterion, earlystopping, scheduler=None):
        self.writer = SummaryWriter(log_dir='../logs/'+self.run_fold_name)
     
        from tqdm.notebook import tqdm
        for epoch in range(self.params['epochs']):
            n_train_acc, n_val_acc = 0, 0
            train_loss, val_loss = 0, 0
            n_train, n_val = 0, 0
            all_labels = []
            all_predictions = []
            if scheduler != None:
                scheduler.step(epoch)
            self.model.train()
            for code_features, msg_features, metrics_features, labels in tqdm(train_loader):
                train_batch_size = len(labels)
                n_train += train_batch_size
                code_features = code_features.to(self.device)
                msg_features = msg_features.to(self.device)
                metrics_features = metrics_features.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model([code_features, msg_features, metrics_features])
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                predicted = torch.max(outputs, 1)[1]
                train_loss += loss.item() * train_batch_size
                n_train_acc += (predicted == labels).sum().item()
            
            self.model.eval()
            with torch.no_grad():
                for val_code_features, val_msg_features, val_metrics_features, val_labels in valid_loader:
                    val_batch_size = len(val_labels)
                    n_val += val_batch_size
                    val_code_features = val_code_features.to(self.device)
                    val_msg_features = val_msg_features.to(self.device)
                    val_metrics_features = val_metrics_features.to(self.device)
                    val_labels = val_labels.to(self.device)
                    val_outputs = self.model([val_code_features, val_msg_features, val_metrics_features])
                    loss_valid = criterion(val_outputs, val_labels)
                    predicted_val = torch.max(val_outputs, 1)[1]
                    val_loss += loss_valid.item() * val_batch_size
                    n_val_acc += (predicted_val == val_labels).sum().item()
                    all_labels.extend(val_labels.cpu().numpy())
                    all_predictions.extend(torch.nn.Softmax(dim = 1)(val_outputs).cpu().numpy()[:,1])
                    
            train_acc = n_train_acc / n_train
            val_acc = n_val_acc / n_val
            avg_train_loss = train_loss / n_train
            avg_val_loss = val_loss / n_val
            auc = roc_auc_score(all_labels, all_predictions)
            answer = [1 if prob >= 0.5 else 0 for prob in all_predictions]
            recall = recall_score(all_labels, answer)
            precision = precision_score(all_labels, answer)
            f1 = f1_score(all_labels, answer)
            print (
                f'Epoch [{(epoch+1)}/{self.params["epochs"]}],\
                lr: {optimizer.param_groups[0]["lr"]:.8f},\
                train_loss: {avg_train_loss:.5f},\
                train_acc: {train_acc:.5f},\
                val_loss: {avg_val_loss:.5f},\
                val_acc: {val_acc:.5f},\
                auc: {auc:.5f},\
                recall: {recall:.5f},\
                precision: {precision:.5f},\
                f1: {f1:.5f}'
            )
            self.writer.add_scalar('lr',optimizer.param_groups[0]["lr"], epoch)
            self.writer.add_scalar('train/train_loss', avg_train_loss, epoch)
            self.writer.add_scalar('train/train_acc', train_acc, epoch)
            self.writer.add_scalar('val/val_loss', avg_val_loss, epoch)
            self.writer.add_scalar('val/val_acc', val_acc, epoch)
            self.writer.add_scalar('val/auc', auc, epoch)
            self.writer.add_scalar('val/recall', recall, epoch)
            self.writer.add_scalar('val/precision', precision, epoch)
            self.writer.add_scalar('val/f1', f1, epoch)
            earlystopping(auc, self.model)
            if earlystopping.early_stop:
                print("Early Stopping")
                break
        
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
                