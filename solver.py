import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torch.nn.init as init

import torch
import torch.nn as nn
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

from utils import to_gpu, time_desc_decorator, DiffLoss, MSE, CMD
import models

def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.ConvTranspose2d):
            init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)
        # 全连接层
        elif isinstance(module, nn.Linear):
            init.xavier_normal_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)
        # 卷积层
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)
        # BatchNorm 层
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)
        # LSTM 和 GRU 层
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    init.xavier_normal_(param.data)
                elif 'weight_hh' in name:
                    init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
        # LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            init.constant_(module.bias, 0)
            init.constant_(module.weight, 1.0)


class Solver(object):
    def __init__(self, train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True, model=None):

        self.train_config = train_config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model
    
    @time_desc_decorator('Build Graph')
    def build(self, cuda=True):

        if self.model is None:
            self.model = models.MISA(self.train_config)
        
        if torch.cuda.is_available() and cuda:
            self.model.cuda()
            for layer in self.model.subject_private:
                layer.cuda()

        if self.is_train:
            self.optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate,
                weight_decay=self.train_config.weight_decay)

        initialize_weights(self.model)

    @time_desc_decorator('Training Start!')
    def train(self):
        curr_patience = patience = self.train_config.patience
        num_trials = 1

        self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
        self.domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.sp_loss_criterion = nn.CrossEntropyLoss(reduction="mean")

        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()   

        best_valid_loss = float('inf')
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        

        train_losses = []
        train_losses_cls = []
        train_losses_sim = []
        train_losses_diff = []
        train_losses_recon = []
        train_losses_subject_recon = []
        train_losses_subject_diff = []
        valid_losses = []
        val_losses_cls = []
        val_losses_sim = []
        val_losses_diff = []
        val_losses_recon = []
        val_losses_subject_recon = []
        val_losses_subject_diff = []

        train_accuracies = []
        valid_accuracies = []

        for e in range(self.train_config.n_epoch):
            self.model.train()


            train_loss_cls, train_loss_sim, train_loss_diff = [], [], []
            train_loss_recon = []
            train_loss_subject_recon = []
            train_loss_subject_diff = []
            train_loss = []
            cnt = 0
            train_acc = 0
            train_cnt = 0
            for batch in self.train_data_loader:
                cnt = cnt + 1
                self.model.zero_grad()
                eeg, eog, emg, y, l, d = batch

                eeg = to_gpu(eeg)
                eog = to_gpu(eog)
                emg = to_gpu(emg)
                y = to_gpu(y)
                l = to_gpu(l)
                d = to_gpu(d)

                y_tilde = self.model(eeg, eog, emg, l, d)

                y = y.squeeze()

                _, predicted = torch.max(y_tilde, 1)
                train_acc = train_acc + predicted.eq(y).sum().item()
                train_cnt = train_cnt + y.shape[0]

                cls_loss = criterion(y_tilde, y)
                diff_loss = self.get_diff_loss()
                recon_loss = self.get_recon_loss()
                cmd_loss = self.get_cmd_loss()
                similarity_loss = cmd_loss

                subject_recon_loss = self.get_subject_recon_loss()
                subject_diff_loss = self.get_subject_diff_loss()
                
                loss = self.train_config.cls_weight * cls_loss + \
                       self.train_config.diff_weight * diff_loss + \
                       self.train_config.sim_weight * similarity_loss + \
                       self.train_config.recon_weight * recon_loss + \
                       self.train_config.subject_recon_weight * subject_recon_loss + \
                       self.train_config.subject_diff_weight * subject_diff_loss

                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())
                train_loss_cls.append(cls_loss.item()*self.train_config.cls_weight)
                train_loss_diff.append(diff_loss.item()*self.train_config.diff_weight)
                train_loss_recon.append(recon_loss.item()*self.train_config.recon_weight)
                train_loss_sim.append(similarity_loss.item()*self.train_config.sim_weight)
                train_loss_subject_recon.append(subject_recon_loss.item()*self.train_config.subject_recon_weight)
                train_loss_subject_diff.append(subject_diff_loss.item()*self.train_config.subject_diff_weight)

            valid_loss, valid_cls_loss, valid_sim_loss, valid_diff_loss, valid_recon_loss, valid_subject_recon_loss, valid_subject_diff_loss, valid_acc = self.eval(mode="dev")

            train_losses.append(np.mean(train_loss))
            train_losses_cls.append(np.mean(train_loss_cls))
            train_losses_sim.append(np.mean(train_loss_sim))
            train_losses_diff.append(np.mean(train_loss_diff))
            train_losses_recon.append(np.mean(train_loss_recon))
            train_losses_subject_recon.append(np.mean(train_loss_subject_recon))
            train_losses_subject_diff.append(np.mean(train_loss_subject_diff))
            train_accuracies.append(train_acc/train_cnt)
            valid_losses.append(valid_loss)
            val_losses_cls.append(np.mean(valid_cls_loss))
            val_losses_sim.append(np.mean(valid_sim_loss))
            val_losses_diff.append(np.mean(valid_diff_loss))
            val_losses_recon.append(np.mean(valid_recon_loss))
            val_losses_subject_recon.append(np.mean(valid_subject_recon_loss))
            val_losses_subject_diff.append(np.mean(valid_subject_diff_loss))
            valid_accuracies.append(valid_acc)

            print(f"Epoch {e+1}/{self.train_config.n_epoch}, train loss: {np.mean(train_loss)}, train acc: {train_acc/train_cnt},valid loss: {valid_loss}, valid acc: {valid_acc}")
            print(f"Current patience: {curr_patience}, current trial: {num_trials}, current acc: {valid_acc}.")
            if valid_loss <= best_valid_loss:
                best_valid_loss = valid_loss
                print("Found new best model on dev set!")
                if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
                torch.save(self.model.state_dict(), f'checkpoints/model_{self.train_config.name}.std')
                torch.save(self.optimizer.state_dict(), f'checkpoints/optim_{self.train_config.name}.std')
                curr_patience = patience
            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    print("Running out of patience, loading previous best model.")
                    num_trials -= 1
                    curr_patience = patience
                    self.model.load_state_dict(torch.load(f'checkpoints/model_{self.train_config.name}.std'))
                    self.optimizer.load_state_dict(torch.load(f'checkpoints/optim_{self.train_config.name}.std'))
                    lr_scheduler.step()
                    print(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")
            
            if num_trials <= 0:
                print("Running out of patience, early stopping.")
                break

        _, _, _, _, _, _, _, test_acc = self.eval(mode="test",to_print=True)
        print(f"Test acc: {test_acc}")

        plt.figure(figsize=(15, 12))
        # Plotting loss
        plt.subplot(3, 3, 1)
        plt.plot(range(1, len(train_losses_sim)+1), train_losses_sim, label='Train sim')
        plt.plot(range(1, len(train_losses_diff)+1), train_losses_diff, label='Train diff')
        plt.plot(range(1, len(train_losses_recon)+1), train_losses_recon, label='Train recon')
        plt.plot(range(1, len(train_losses_subject_recon)+1), train_losses_subject_recon, label='Train subject recon')
        plt.plot(range(1, len(train_losses_subject_diff)+1), train_losses_subject_diff, label='Train subject diff')
        plt.title('Loss over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plotting accuracy
        plt.subplot(3, 3, 2)
        plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label='Train')
        plt.plot(range(1, len(valid_accuracies)+1), valid_accuracies, label='Validation')
        plt.title('Accuracy over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(3, 3, 3)
        plt.plot(range(1, len(train_losses)+1), train_losses, label='Train')
        plt.plot(range(1, len(valid_losses)+1), valid_losses, label='Validation')
        plt.title('Loss over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(3, 3, 4)
        plt.plot(range(1, len(train_losses_sim)+1), train_losses_sim, label='Train sim')
        plt.plot(range(1, len(val_losses_sim)+1), val_losses_sim, label='Val sim')
        plt.title('Sim loss over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(3, 3, 5)
        plt.plot(range(1, len(train_losses_diff)+1), train_losses_diff, label='Train diff')
        plt.plot(range(1, len(val_losses_diff)+1), val_losses_diff, label='Val diff')
        plt.title('Diff loss over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(3, 3, 6)
        plt.plot(range(1, len(train_losses_recon)+1), train_losses_recon, label='Train recon')
        plt.plot(range(1, len(val_losses_recon)+1), val_losses_recon, label='Val recon')
        plt.title('Recon loss over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(3, 3, 7)
        plt.plot(range(1, len(train_losses_subject_recon)+1), train_losses_subject_recon, label='Train subject recon')
        plt.plot(range(1, len(val_losses_subject_recon)+1), val_losses_subject_recon, label='Val subject recon')
        plt.title('Subject recon loss over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(3, 3, 8)
        plt.plot(range(1, len(train_losses_subject_diff)+1), train_losses_subject_diff, label='Train subject diff')
        plt.plot(range(1, len(val_losses_subject_diff)+1), val_losses_subject_diff, label='Val subject diff')
        plt.title('Subject diff loss over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(3, 3, 9)
        plt.plot(range(1, len(train_losses_cls)+1), train_losses_cls, label='Train cls')
        plt.plot(range(1, len(val_losses_cls)+1), val_losses_cls, label='Val cls')
        plt.title('Cls loss over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        ymin, ymax = plt.ylim()
        xmin, xmax = plt.xlim()
        plt.text((xmin+xmax)/2, ymin, f"cls_w_{self.train_config.cls_weight}_diff_w_{self.train_config.diff_weight}_sim_w_{self.train_config.sim_weight}_recon_w_{self.train_config.recon_weight}_sub_recon_w_{self.train_config.subject_recon_weight}_sub_diff_w_{self.train_config.subject_diff_weight}/lr_{self.train_config.learning_rate}_batch_{self.train_config.batch_size}_test_acc_{test_acc}", verticalalignment='bottom', horizontalalignment='center', fontsize=8)


        plt.tight_layout()
        plt.savefig(f'results_time_{self.train_config.name}_cls_w_{self.train_config.cls_weight}_diff_w_{self.train_config.diff_weight}_sim_w_{self.train_config.sim_weight}_recon_w_{self.train_config.recon_weight}_sub_recon_w_{self.train_config.subject_recon_weight}_sub_diff_w_{self.train_config.subject_diff_weight}_lr_{self.train_config.learning_rate}_batch_{self.train_config.batch_size}_test_acc_{test_acc}.png')

    def eval(self, mode='dev', to_print=False):
        assert(mode is not None)
        self.model.eval()

        y_true, y_pred = [], []
        eval_loss, eval_loss_diff = [], []
        eval_loss_sim, eval_loss_recon = [], []
        eval_loss_subject_recon, eval_loss_subject_diff = [], []
        eval_loss_cls = []

        if mode == "dev":
            dataloader = self.dev_data_loader
        elif mode == "test":
            dataloader = self.test_data_loader

            if to_print:
                self.model.load_state_dict(torch.load(
                    f'checkpoints/model_{self.train_config.name}.std'))


        with torch.no_grad():

            for batch in dataloader:
                self.model.zero_grad()
                eeg, eog, emg, y, l, d = batch

                eeg = to_gpu(eeg)
                eog = to_gpu(eog)
                emg = to_gpu(emg)
                y = to_gpu(y)
                l = to_gpu(l)
                d = to_gpu(d)

                y_tilde = self.model(eeg, eog, emg, l, d)

                y = y.squeeze()
                
                cls_loss = self.criterion(y_tilde, y)
                diff_loss = self.get_diff_loss()
                similarity_loss = self.get_cmd_loss()
                recon_loss = self.get_recon_loss()
                subject_recon_loss = self.get_subject_recon_loss()
                subject_diff_loss = self.get_subject_diff_loss()
                loss = self.train_config.cls_weight * cls_loss + \
                          self.train_config.diff_weight * diff_loss + \
                            self.train_config.sim_weight * similarity_loss + \
                            self.train_config.recon_weight * recon_loss + \
                            self.train_config.subject_recon_weight * subject_recon_loss + \
                            self.train_config.subject_diff_weight * subject_diff_loss

                eval_loss.append(loss.item())
                eval_loss_cls.append(cls_loss.item()*self.train_config.cls_weight)
                eval_loss_diff.append(diff_loss.item()*self.train_config.diff_weight)
                eval_loss_sim.append(similarity_loss.item()*self.train_config.sim_weight)
                eval_loss_recon.append(recon_loss.item()*self.train_config.recon_weight)
                eval_loss_subject_recon.append(subject_recon_loss.item()*self.train_config.subject_recon_weight)
                eval_loss_subject_diff.append(subject_diff_loss.item()*self.train_config.subject_diff_weight)

                y_pred.append(y_tilde.detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())

        eval_loss = np.mean(eval_loss_cls)
        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()

        accuracy = accuracy_score(y_true, np.argmax(y_pred, axis=1))

        test_preds = np.argmax(y_pred, 1)
        test_truth = y_true

        if to_print:
            cm = confusion_matrix(test_truth, test_preds)
            print("Confusion Matrix (pos/neg) :")
            print(cm)
            print("Classification Report (pos/neg) :")
            print(classification_report(test_truth, test_preds, digits=5))
            print("Accuracy (pos/neg) ", accuracy_score(test_truth, test_preds))
            fig, ax = plt.subplots()
            ax.imshow(cm, cmap='Blues')

            # 添加颜色条
            cbar = ax.figure.colorbar(ax.imshow(cm, cmap='Blues'))
            cbar.ax.set_ylabel('quantity', rotation=-90, va="bottom")

            # 添加文本
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j],
                            ha="center", va="center",
                            color="white" if cm[i, j] > np.max(cm) / 2 else "black")

            ax.set_xlabel('predicted label')
            ax.set_ylabel('true label')
            ax.set_title('Confusion Matrix')
            plt.savefig(f'confusion_matrix{self.train_config.name}.png')

        return eval_loss, eval_loss_cls, eval_loss_sim, eval_loss_diff, eval_loss_recon, eval_loss_subject_recon, eval_loss_subject_diff, accuracy

    def get_cmd_loss(self,):

        # losses between shared states
        loss = self.loss_cmd(self.model.utt_shared_eeg, self.model.utt_shared_eog, 5)
        loss += self.loss_cmd(self.model.utt_shared_eeg, self.model.utt_shared_emg, 5)
        loss += self.loss_cmd(self.model.utt_shared_emg, self.model.utt_shared_eog, 5)
        loss = loss/3.0

        return loss

    def get_diff_loss(self):

        shared_eeg = self.model.utt_shared_eeg
        shared_eog = self.model.utt_shared_eog
        shared_emg = self.model.utt_shared_emg
        private_eeg = self.model.utt_private_eeg
        private_eog = self.model.utt_private_eog
        private_emg = self.model.utt_private_emg

        # Between private and shared
        loss = self.loss_diff(private_eeg, shared_eeg)
        loss += self.loss_diff(private_eog, shared_eog)
        loss += self.loss_diff(private_emg, shared_emg)

        # Across privates
        loss += self.loss_diff(private_emg, private_eeg)
        loss += self.loss_diff(private_eog, private_emg)
        loss += self.loss_diff(private_eeg, private_eog)

        return loss
    
    def get_recon_loss(self, ):

        loss = self.loss_recon(self.model.utt_eeg_recon, self.model.utt_eeg_orig)
        loss += self.loss_recon(self.model.utt_eog_recon, self.model.utt_eog_orig)
        loss += self.loss_recon(self.model.utt_emg_recon, self.model.utt_emg_orig)
        loss = loss / 3.0

        return loss

    def get_subject_diff_loss(self):
        loss = self.loss_diff(self.model.utt_private_subject, self.model.utt_shared_subject)
        return loss

    def get_subject_recon_loss(self, ):
        loss = self.loss_recon(self.model.utt_subject_recon, self.model.utt_subject)
        return loss





