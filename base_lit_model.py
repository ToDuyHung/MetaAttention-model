import random

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

import pytorch_lightning as pl
import torchmetrics


class BaseLitModel(pl.LightningModule):
    def __init__(self, model, loss, get_optim, lr, get_lr_scheduler=None):
        super().__init__()
        self.model = model
        self.loss = loss
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.val_top2_acc = torchmetrics.Accuracy(top_k=2)
        self.get_optim = get_optim
        self.lr = lr
        self.get_lr_scheduler = get_lr_scheduler
        self.save_hyperparameters(ignore=['model', 'loss', 'get_optim', 'get_lr_scheduler'])

    def forward(self, X):
        return self.model(X)

    def configure_optimizers(self):
        optim = self.get_optim(self.parameters(), self.lr)
        if not self.get_lr_scheduler:
            return optim
        
        return {'optimizer': optim, 'lr_scheduler': self.get_lr_scheduler(optim), "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        l = self.loss(y_pred, y)
        self.train_acc(y_pred, y)
        self.log('train_loss', l)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return l
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        l = self.loss(y_pred, y)
        self.val_acc(y_pred, y)
        self.val_top2_acc(y_pred, y)
        self.log('val_loss', l, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_top2_acc', self.val_top2_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def predict_step(self, batch, batch_idx):
        X, _ = batch
        return self(X)
    