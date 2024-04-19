import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import torch.optim as optim
from model import Model
from code_transformer import CodeChangeTransformer
from dataset import CodeChangeDataset
from utils import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from utils import Logger
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
import gc

# logger = Logger()


class ModelCCT(Model):

    # def __init__(self):
    #     super.__init__()

    def train(self, filepath, train_idx, valid_idx, kind='train'):
        dataset = CodeChangeDataset(
            filepath,
            spm_modelpath=self.params['spm_modelpath'],
            kind=kind,
            max_file=self.params['max_file'],
            max_line_len=self.params['max_line_len']
        )
        self.params['n_tokens'] = len(dataset.corpus)
        self.params['pad_id'] = dataset.corpus['<pad>']
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
        
        self.model = CodeChangeTransformer(self.params).to(self.device)
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params['lr'],
            weight_decay=self.params['weight_decay']
        )
        criterion = nn.CrossEntropyLoss()
        earlystopping = EarlyStopping(
            patience=self.params['patience'],
            verbose=True,
            path='../best_model/'+self.run_fold_name+'.pth'
        )
        self.fit(train_loader, valid_loader, optimizer, criterion, earlystopping)
        
        model_path = os.path.join(
            '../best_model', f'{self.run_fold_name}.path'
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        self.model = CodeChangeTransformer(self.params).to(self.device)
        self.model.load_state_dict(torch.load(model_path))

    def predict(self, filepath, test_idx):
        dataset = CodeChangeDataset(
            filepath,
            spm_modelpath=self.params['spm_modelpath'],
            max_file=self.params['max_file'],
            max_line_len=self.params['max_line_len']
        )
        test_loader = DataLoader(
            dataset,
            batch_size=self.params['batch_size'],
            shuffle=False,
            
        )
        pred = np.zeros(0)
        self.model.eval()
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                outputs = self.model(features)
                pred = np.concatenate(pred, nn.Softmax(dim = 1)(outputs).cpu().numpy()[:,1])
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        return pred

    def load_model(self):
        model_path = os.path.join('../best_model', f'{self.run_fold_name}.pth')
        self.model = CodeChangeTransformer(self.params).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        
    def fit(self, train_loader, valid_loader, optimizer, criterion, earlystopping):
        writer = SummaryWriter(log_dir='logs/'+self.run_fold_name)
        from tqdm import tqdm
        for epoch in range(self.params['epochs']):
            n_train_acc, n_val_acc = 0, 0
            train_loss, val_loss = 0, 0
            n_train, n_val = 0, 0
            all_labels = []
            all_predictions = []
            
            self.model.train()
            for features, segment_label, file_mask, labels in tqdm(train_loader):
                train_batch_size = len(labels)
                n_train += train_batch_size
                features = features.to(self.device)
                segment_label = segment_label.to(self.device)
                file_mask = file_mask.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model([features, segment_label, file_mask])
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                predicted = torch.max(outputs, 1)[1]
                train_loss += loss.item() * train_batch_size
                n_train_acc += (predicted == labels).sum().item()
            
            self.model.eval()
            with torch.no_grad():
                for val_features, val_segment_label, val_file_mask, val_labels in valid_loader:
                    val_batch_size = len(val_labels)
                    n_val += val_batch_size
                    val_features = val_features.to(self.device)
                    val_segment_label = val_segment_label.to(self.device)
                    val_file_mask = val_file_mask.to(self.device)
                    val_labels = val_labels.to(self.device)
                    val_outputs = self.model([val_features, val_segment_label, val_file_mask])
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
            recall = recall_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions)
            f1 = f1_score(all_labels, all_predictions)
            print (
                f'Epoch [{(epoch+1)}/{self.params["epochs"]}],\
                train_loss: {avg_train_loss:.5f},\
                train_acc: {train_acc:.5f},\
                val_loss: {avg_val_loss:.5f},\
                val_acc: {val_acc:.5f},\
                auc: {auc:.5f},\
                recall: {recall:.5f},\
                precision: {precision:.5f},\
                f1: {f1:.5f}'
            )
            writer.add_scalar('train_loss', avg_train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)
            writer.add_scalar('val_loss', avg_val_loss, epoch)
            writer.add_scalar('val_acc', val_acc, epoch)
            writer.add_scalar('auc', auc, epoch)
            writer.add_scalar('recall', recall, epoch)
            writer.add_scalar('precision', precision, epoch)
            writer.add_scalar('f1', f1, epoch)
            earlystopping(avg_val_loss, self.model)
            if earlystopping.early_stop:
                print("Early Stopping")
                break
        writer.close()
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
            
            
                    
                    
                