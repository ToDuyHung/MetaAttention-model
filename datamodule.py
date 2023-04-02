import os
import pickle

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


class HAM10000_ABCD_Dataset(Dataset):
    def __init__(self, X, y, encoder, transform=None, p_dropout=0, mode='ABCD', use_meta=True):
        self.use_meta = use_meta
        self.encoder = encoder
        self.mode = mode
        self.X = X
        self.data = self._split_and_encode_data(X)
        self.y = self.encoder['dx'].transform(y['dx'])
        self.transform = transform
        self.p_dropout=p_dropout
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        y = self.y[index]
        # img = Image.open(self.data['path'][index])
        img = cv2.imread(self.data['path'][index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            # img = self.transform(img)
            img = self.transform(image=img)['image']
        
        if not self.use_meta:
            return img, torch.tensor(y, dtype=torch.long)

        if self.mode == '':
            row = {k: self.data[k][index] for k in ('age', 'sex', 'localization')}
        else:
            row = {k: self.data[k][index] for k in ('age', 'sex', 'localization', 'abcd')}

        if self.p_dropout > 0:
            for attr in ('age', 'sex', 'localization'):
                if np.random.rand() < self.p_dropout:
                    row[attr] = np.zeros_like(row[attr])

        metadata = torch.tensor(np.concatenate(list(row.values())), dtype=torch.float32)
        data = (img, metadata)

        return data, torch.tensor(y, dtype=torch.long)

    def _split_and_encode_data(self, X):
        data = {
            'path': X['path']
        }

        abcd = []
        if 'D' not in self.mode:
            X = X.drop('eq_diameters', axis=1)
        if 'C' not in self.mode:
            X = X.drop(['C_rs', 'C_gs', 'C_bs'], axis=1)
        if 'B' not in self.mode:
            X = X.drop('compact_indexs', axis=1)

        for attr in X.keys():
            if attr == 'path':
                continue
            elif attr == 'age':
                data['age'] = self.encoder[attr].transform(X[attr].values.reshape(-1, 1))
            elif attr in ('sex', 'localization'):
                data[attr] = self.encoder[attr].transform(X[attr].values.reshape(-1, 1)).toarray()
            else:
                abcd.append(self.encoder[attr].transform(X[attr].values.reshape(-1, 1)))
        data['abcd'] = np.concatenate(abcd, axis=1)
        return data

    def label_counts(self):
        return np.bincount(self.y)
class HAM10000_ABCD_DataModule(pl.LightningDataModule):
    def __init__(self, data_path, img_size, batch_size=128, sampler_func=None, mode='ABCD', use_meta=True):
        super().__init__()
        self.use_meta=use_meta
        self.data_path = data_path
        self.batch_size = batch_size
        self.sampler_func = sampler_func
        self.sampler = None
        self.train_trans = A.Compose([
            A.SmallestMaxSize(max_size=img_size + 20),
            # A.Resize(width=int((img_size+20) / 0.75), height=(img_size+20)),
            A.ShiftScaleRotate(0.05, 0.2, 20, border_mode=cv2.BORDER_REPLICATE),
            A.RandomCrop(width=img_size, height=img_size),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.05),
            # A.Affine(scale=(.8, 1.2), rotate=(-20, 20), mode=cv2.BORDER_REPLICATE, always_apply=True),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        self.val_trans = A.Compose([
            A.SmallestMaxSize(max_size=img_size + 20),
            # A.Resize(width=int(img_size / 0.75), height=img_size),
            A.CenterCrop(width=img_size, height=img_size),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        self.dropout=0.3
        self.train_ds, self.val_ds = None, None
        self.encoder = None
        self.mode = mode

    def setup(self, stage=None):
        if not self.train_ds:
            X_train = pd.read_csv(os.path.join(self.data_path, 'X_train_abcd.csv'), index_col=0)
            y_train = pd.read_csv(os.path.join(self.data_path, 'y_train.csv'), index_col=0)
            X_val = pd.read_csv(os.path.join(self.data_path, 'X_val_abcd.csv'), index_col=0)
            y_val = pd.read_csv(os.path.join(self.data_path, 'y_val.csv'), index_col=0)
            X_test = pd.read_csv(os.path.join(self.data_path, 'X_test_abcd.csv'), index_col=0)
            y_test = pd.read_csv(os.path.join(self.data_path, 'y_test.csv'), index_col=0)
            
            for x in (X_train, X_val, X_test):
                for attr in ('sex', 'localization'):
                    x[attr].replace({'unknown': None}, inplace=True)
            
                for attr in ('age', 'asymm_idxs', 'eccs', 'compact_indexs', 'C_rs', 'C_gs', 'C_bs', 'eq_diameters'):
                    x[attr].fillna(X_train[attr].mean(), inplace=True)

            with open(os.path.join(self.data_path, 'HAM10000_abcd_encoder_norm.pkl'), 'rb') as f:
                self.encoder = pickle.load(f)
    
            # Define Dataset
            self.train_ds = HAM10000_ABCD_Dataset(X_train, y_train, self.encoder, transform=self.train_trans, p_dropout=self.dropout, mode=self.mode, use_meta=self.use_meta)
            self.val_ds = HAM10000_ABCD_Dataset(X_val, y_val, self.encoder, transform=self.val_trans, mode=self.mode, use_meta=self.use_meta)
            self.test_ds = HAM10000_ABCD_Dataset(X_test, y_test, self.encoder, transform=self.val_trans, mode=self.mode, use_meta=self.use_meta)
    
            # Define Batch Sampler
            if self.sampler_func:
                self.sampler = self.sampler_func(self.train_ds)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=2, sampler = self.sampler, pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=2, pin_memory=True)
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=2, pin_memory=True)