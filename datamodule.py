import os
import pickle

import cv2
import pytorch_lightning as pl
import albumentations as A
import pandas as pd

from albumentations.pytorch import ToTensorV2

class HAM10000_ABCD_DataModule(pl.LightningDataModule):
    def __init__(self, data_path, img_size, batch_size=128, sampler_func=None, mode='ABCD', use_meta=True):
        super().__init__()
        self.use_meta=use_meta
        self.data_path = data_path
        self.batch_size = batch_size
        self.sampler_func = sampler_func
        self.sampler = None
        # self.train_trans = transforms.Compose([
        #     transforms.RandomAffine(degrees=20, scale=(.8, 1.2)),
        #     transforms.Resize(img_size + 20),
        #     transforms.RandomCrop(img_size),
        #     transforms.RandomVerticalFlip(),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
        #     transforms.GaussianBlur(kernel_size=(5, 5)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        # self.val_trans = transforms.Compose([
        #     transforms.Resize(img_size),
        #     transforms.CenterCrop(img_size),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
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