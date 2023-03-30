import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import os
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data.sampler import WeightedRandomSampler, Sampler, SubsetRandomSampler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import cv2
import albumentations as A
from Block import MetaBlock, WrappedMetaBlock, CombineModel

import pickle
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import loggers as pl_loggers

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

PATH = '/content/drive/MyDrive/LVTN/'
# DATAPATH = os.path.join(PATH, 'HAM10000', 'official_train_test_data')
# DATAPATH = '/AIHCM/ComputerVision/hungtd/norm_data_csv'
DATAPATH = '/AIHCM/FileShare/Public/AI_Member/hungtd/data_sv23/hungtd/norm_data_csv'
# MODELPATH = os.path.join(PATH, 'pytorch-model', 'ham10000')
# MODELPATH = '/AIHCM/ComputerVision/hungtd/all_model_path/EffNoABAdditive_OldTransform'
MODELPATH = '/AIHCM/FileShare/Public/AI_Member/hungtd/data_sv23/hungtd/all_model_path/EffNoABAdditive_OldTransform'

def get_weighted_random_sampler(dataset):
    sampler_weights = torch.zeros(len(dataset))
    class_sampler = dataset.label_counts()
    for idx, label in enumerate(dataset.y):
        sampler_weights[idx] = class_sampler[label]

    sampler_weights = 1000. / sampler_weights
    return WeightedRandomSampler(sampler_weights.type('torch.DoubleTensor'), len(sampler_weights))

def get_oversampler(dataset):
    ros = RandomOverSampler(random_state=0)
    indices = np.arange(len(dataset))
    indices, _ = ros.fit_resample(indices.reshape(-1, 1), dataset.y)
    indices = indices.reshape(-1, )
    np.random.shuffle(indices)
    return SubsetRandomSampler(indices)

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
        name = self.data['path'][index].split('/')[-1]
        folder = self.data['path'][index].split('/')[-2]
        # name = os.path.join('/AIHCM/ComputerVision/hungtd/ham10000_hugvision_augment', name)
        name = os.path.join('/AIHCM/FileShare/Public/AI_Member/hungtd/data_sv23/hungtd/ham10000_hugvision_augment', folder, name)
        
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
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
    
class EffNetMetaImageAttentionV3(nn.Module):
    def __init__(self, base_model, meta_attn_blk_cls, attention_indices, num_classes, d_meta, dropout=0.1, embed_dim=512):
        super().__init__()
        self.meta_attn_blk_cls = meta_attn_blk_cls
        self.d_meta = d_meta
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.num_classes = num_classes
 
        self.ftr_extractors = nn.ModuleList()
        old_idx = 0
        for idx in attention_indices:
            self.ftr_extractors.append(base_model[old_idx:idx+1])
            old_idx = idx+1
        if old_idx < len(base_model):
            self.ftr_extractors.append(base_model[old_idx:])
        self.num_attn = len(attention_indices)
        self.attn_blks, self.out_channels = self._attn_blocks()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.out_channels, num_classes)
        )
 
        self._init_xavierUniform(self.attn_blks)
        self._init_xavierUniform(self.classifier)
    
    def forward(self, X, get_attn_maps=False):
        img, meta = X
        y = img
        out_ctx = []
        out_attn = []
        for i in range(self.num_attn):
            y = self.ftr_extractors[i](y)
            out_ctx.append(y)
        if self.num_attn < len(self.ftr_extractors):
            y = self.ftr_extractors[-1](y)
        g = self.avgpool(y).flatten(1)
        for i in range(self.num_attn):
            ctx, attn = self.attn_blks[i](out_ctx[i], meta, g)
            out_ctx[i] = ctx
            out_attn.append(attn)
        g = torch.cat(out_ctx, dim=1)
        y = self.classifier(g)
        if get_attn_maps:
            return y, out_attn
        return y
 
    def _attn_blocks(self):
        attn_blks = nn.ModuleList()
        out = torch.zeros(1, 3, 1, 1)
        out_channels = []
        self.eval()
        with torch.no_grad():
            for i in range(self.num_attn):
                out = self.ftr_extractors[i](out)
                out_channels.append(out.shape[1])
            if self.num_attn < len(self.ftr_extractors):
                out = self.ftr_extractors[-1](out)
        self.train()
        for ch in out_channels:
            attn_blks.append(
                # self.MetaImageAttentionBlock(self.d_meta,
                #                              ch,
                #                              out_channels[-1],
                #                              self.embed_dim,
                #                              self.dropout)
                self.meta_attn_blk_cls(self.d_meta,
                                       ch,
                                       out.shape[1],
                                       self.embed_dim,
                                       self.dropout)
            )
        return attn_blks, sum(out_channels)
    
    def unfreeze_img_conv(self):
        for param in self.ftr_extractors.parameters():
            param.requires_grad = True
    
    def freeze_img_conv(self):
        for param in self.ftr_extractors.parameters():
            param.requires_grad = False
 
    def _init_xavierUniform(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain = np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, a = 0, b = 1)
                nn.init.constant_(m.bias, val = 0.)
    
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain = np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, val = 0.)

class AdditiveMetaAttentionBlock(nn.Module):
    def __init__(self, d_meta, c_img, d_glob, embed_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim if embed_dim else c_img
        self.d_meta = d_meta
        self.meta_embed = nn.Sequential(
            nn.Linear(d_meta, self.embed_dim, bias=False),
            # nn.GELU()
        )

        self.img_embed = nn.Sequential(
            nn.Conv2d(in_channels=c_img, out_channels=c_img, kernel_size=3, padding='same', groups=c_img, bias=False),
            nn.Conv2d(in_channels=c_img, out_channels=embed_dim, kernel_size=1, bias=False)
            # nn.Conv2d(c_img, embed_dim, kernel_size=3, padding='same', bias=False),
            # nn.Conv2d(c_img, embed_dim, kernel_size=7, padding='same'),
            # nn.GELU()
        ) if embed_dim else nn.Identity()

        self.glob_embed = nn.Linear(d_glob, self.embed_dim, bias=False)

        # self.score = nn.Conv2d(embed_dim, 1, kernel_size=3, padding='same', bias=False)
        self.score = nn.Linear(embed_dim, 1, bias=False)

        self.norm = nn.LayerNorm(c_img)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, X_img, X_meta, global_ftr):
        # X_img: N x c x h x w
        # X_meta: N x d_meta
        N, _, h, w = X_img.size()

        meta_embed = self.dropout(self.meta_embed(X_meta)) # N x embed_dim
        img_embed = self.img_embed(X_img) # N x embed_dim x h x w
        g_embed = self.dropout(self.glob_embed(global_ftr)) # N x embed_dim

        img_embed = img_embed.flatten(2) # N x embed_dim x h*w
        # Scaled Dot Production
        c = img_embed + meta_embed.unsqueeze(-1) + g_embed.unsqueeze(-1) # N x embed_dim x h*w
        # score = self.score(torch.tanh(c.view(N, -1, h, w))) # N x 1 x h x w
        score = self.score(torch.tanh(c.transpose(1, 2))) # N x h*w x 1
        attn_map = F.softmax(score.view(N, 1, -1), dim=2) # N x 1 x h*w
        
        # N x 1 x c
        context = torch.matmul(attn_map, X_img.flatten(-2).transpose(-2, -1))
        context = self.norm(context.squeeze(1))
        attn_map = attn_map.view(N, 1, h, w)
        # context = (1 + attn_map) * X_img
        # (N x c x h x w, N x h x w)
        return context, attn_map.squeeze(1)

class ScaledDotMetaAttentionBlock(nn.Module):
    def __init__(self, d_meta, c_img, d_glob, embed_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim if embed_dim else c_img
        self.d_meta = d_meta
        self.meta_embed = nn.Sequential(
            nn.Linear(d_meta, self.embed_dim, bias=False),
            # nn.GELU()
        )

        self.img_embed = nn.Sequential(
            nn.Conv2d(in_channels=c_img, out_channels=c_img, kernel_size=3, padding='same', groups=c_img, bias=False),
            nn.Conv2d(in_channels=c_img, out_channels=embed_dim, kernel_size=1, bias=False)
            # nn.Conv2d(c_img, embed_dim, kernel_size=3, padding='same', bias=False),
            # nn.Conv2d(c_img, embed_dim, kernel_size=7, padding='same'),
            # nn.GELU()
        ) if embed_dim else nn.Identity()

        self.glob_embed = nn.Linear(d_glob, self.embed_dim, bias=False)

        # self.score = nn.Conv2d(embed_dim, 1, kernel_size=3, padding='same', bias=False)

        self.norm = nn.LayerNorm(c_img)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, X_img, X_meta, global_ftr):
        # X_img: N x c x h x w
        # X_meta: N x d_meta
        # global_ftr: N x d_glob
        N, _, h, w = X_img.size()

        meta_embed = self.dropout(self.meta_embed(X_meta)) # N x embed_dim
        img_embed = self.img_embed(X_img) # N x embed_dim x h x w
        g_embed = self.dropout(self.glob_embed(global_ftr)) # N x embed_dim

        x = meta_embed.unsqueeze(1) + g_embed.unsqueeze(1) # N x 1 x embed_dim
        # g_embed = g_embed.unsqueeze(1)
        # x = torch.cat([meta_embed, g_embed], dim=1) # N x 2 x embed_dim
        img_embed = img_embed.flatten(2) # N x embed_dim x h*w
        # Scaled Dot Production
        c = torch.matmul(x, img_embed) / self.embed_dim ** 0.5 # N x 2 x h*w
        # c = c.mean(dim=1) # N x h*w
        # attn_map = F.softmax(c.unsqueeze(1), dim=2) # N x 1 x h*w
        attn_map = F.softmax(c, dim=2) # N x 1 x h*w
        
        # N x 1 x c
        context = torch.matmul(attn_map, X_img.flatten(-2).transpose(-2, -1))
        context = self.norm(context.squeeze(1))
        attn_map = attn_map.view(N, 1, h, w)
        # context = (1 + attn_map) * X_img
        # (N x c x h x w, N x h x w)
        return context, attn_map.squeeze(1)
    
data = HAM10000_ABCD_DataModule(data_path=DATAPATH,
                                img_size = 224,
                                batch_size = 32,
                                sampler_func=get_weighted_random_sampler,
                                mode='', use_meta=True)
data.setup()

effnet_version = 'efficientnet_b4'
base_model = getattr(models, effnet_version)(pretrained=True)
# base_model = models.resnet50(pretrained=True)

num_classes=7
num_hidden = 1024
d_meta = data.val_ds[0][0][1].shape[0]
print(d_meta)

# base_model.classifier[1] = nn.Linear(base_model.classifier[1].in_features, num_classes)
# model = base_model

# # ---------------ResNet only -----------------------
# base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
# model = base_model

#  --------------AdditiveMetaAttention ResNet----------------
# base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
# img_block = torch.nn.Sequential(*(list(base_model.children())[:-1]))
# model = EffNetMetaImageAttentionV3(img_block, ScaledDotMetaAttentionBlock, [4, 5, 6], num_classes, d_meta, dropout=0.3, embed_dim=512)
# -----------------------------------------------------

#  --------------AdditiveMetaAttention ----------------
img_block = base_model.features
model = EffNetMetaImageAttentionV3(img_block, AdditiveMetaAttentionBlock, [3, 5, 8], num_classes, d_meta, dropout=0.3, embed_dim=512)
# -----------------------------------------------------

# img_block = nn.Sequential(*list(base_model.children())[:-2])
# model = EffNetMetaImageAttentionV3(img_block, ScaledDotMetaAttentionBlock, [4, 5, 6], num_classes, d_meta, dropout=0.3, embed_dim=512)
# ---------------ResNet Meta -----------------------
# base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
# img_block = torch.nn.Sequential(*(list(base_model.children())[:-1]))
# num_hidden = 1024
# metadata_block = nn.Sequential(
#           nn.LazyLinear(1024),
#           nn.BatchNorm1d(1024),
#           nn.ReLU()
# )
# metablock = WrappedMetaBlock(MetaBlock(64, 1024))
# classifier = nn.Sequential(
#     nn.LazyLinear(num_hidden),
#     nn.BatchNorm1d(num_hidden),
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(num_hidden, num_classes)
#     # NormalizedLinear(num_hidden, num_classes)
# ) 
# # model = base_model
# model = CombineModel(img_block, metadata_block, metablock, classifier, freeze_conv=False)

optim = lambda param, lr: torch.optim.AdamW(param, lr)
lr = 1e-3
lr_sched = lambda optim: torch.optim.lr_scheduler.ReduceLROnPlateau(optim, min_lr=1e-6)
scheduler_cosine = lambda optim: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, 60)
ce_loss = F.cross_entropy
# focal_loss = losses.FocalLoss(alpha=0.25, gamma=2, reduction='mean')
# cosface_loss = AngularPenaltySMLoss('cosface')
# arcface_loss = AngularPenaltySMLoss('arcface')
weighted_ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(1000. / data.train_ds.label_counts(), dtype=torch.float))
lit_model = BaseLitModel(model, ce_loss, optim, lr, get_lr_scheduler=lr_sched)

version_model = 'v4-res-net-scaled-dot'
model_name = 'effnetmetaimgattention-abcd'
model_path = os.path.join(MODELPATH, model_name, version_model)
# checkpt_callback = ModelCheckpoint(
#     monitor='val_loss',
#     dirpath=os.path.join(MODELPATH, 'effnetmetaimgattention', version_model),
#     mode='min',
#     save_top_k=5,
#     save_on_train_epoch_end=True,
#     filename="{epoch:02d}-{val_loss:.4f}",
#     save_last=True
# )
checkpt_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=model_path,
    mode='min',
    save_top_k=5,
    save_on_train_epoch_end=True,
    filename="{epoch:02d}-{val_acc:.4f}",
    save_last=True
)
logger = pl_loggers.TensorBoardLogger(MODELPATH, name=model_name, version=version_model)
trainer = pl.Trainer(gpus=0, max_epochs=50,
                     callbacks=[checkpt_callback],
                     logger=logger,
                     auto_lr_find=True,
                     num_sanity_val_steps=0,
                     strategy='dp')
# trainer.tune(model, train_loader, val_loader)
# trainer.fit(lit_model, datamodule=data)

# trainer.tune(model, train_loader, val_loader)
# trainer.fit(lit_model, datamodule=data)

# trainer.fit(lit_model, datamodule=data)

lit_model = BaseLitModel.load_from_checkpoint(
    # os.path.join(model_path, 'epoch=83-val_acc=0.8596.ckpt'),
    os.path.join(model_path, 'epoch=105-val_acc=0.8634.ckpt'),
    model=model,
    loss=ce_loss,
    get_optim=optim,
    lr=lr,
    get_lr_scheduler=lr_sched)
test_loader = data.test_dataloader()
preds = trainer.predict(lit_model, dataloaders=test_loader)
preds = torch.cat(preds)
y_pred = preds.argmax(dim=1)
y = data.test_ds.y
print(classification_report(y, y_pred, target_names=data.test_ds.encoder['dx'].classes_, digits=4))