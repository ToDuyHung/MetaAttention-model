import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import models

from base_lit_model import BaseLitModel
from datamodule import HAM10000_ABCD_DataModule
from metablock import CombineModel, MetaBlock, WrappedMetaBlock
from models import (AdditiveMetaAttentionBlock, EffNetMetaImageAttentionV3,
                    ScaledDotMetaAttentionBlock)
from samplers import get_oversampler, get_weighted_random_sampler

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

PATH = '/content/drive/MyDrive/LVTN/'
DATAPATH = os.path.join(PATH, 'HAM10000', 'official_train_test_data')
MODELPATH = os.path.join(PATH, 'pytorch-model', 'ham10000')


def train(lit_model, epochs, datamodule, model_path, model_name, version_model):
    checkpt_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath=model_path,
        mode='max',
        save_top_k=5,
        save_on_train_epoch_end=True,
        filename="{epoch:02d}-{val_acc:.4f}-{val_loss:.4f}",
        save_last=True
    )
    logger = pl_loggers.TensorBoardLogger(MODELPATH, name=model_name, version=version_model)
    trainer = pl.Trainer(gpus=1, max_epochs=epochs,
                        callbacks=[checkpt_callback],
                        logger=logger,
                        auto_lr_find=True,
                        num_sanity_val_steps=0,
                        strategy='dp')
        
    trainer.fit(lit_model, datamodule=datamodule)

if __name__ == "__main__":
    data = HAM10000_ABCD_DataModule(data_path=DATAPATH,
                                    img_size = 224,
                                    batch_size = 32,
                                    sampler_func=get_weighted_random_sampler,
                                    mode='', use_meta=True)
    data.setup()

    d_meta = data.val_ds[0][0][1].shape[0]
    num_classes=7
    num_hidden = 1024
    optim = lambda param, lr: torch.optim.AdamW(param, lr)
    lr = 1e-3
    lr_sched = lambda optim: torch.optim.lr_scheduler.ReduceLROnPlateau(optim, min_lr=1e-6)
    scheduler_cosine = lambda optim: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, 60)
    ce_loss = F.cross_entropy
    trainer = pl.Trainer(gpus=1)

    MODE = "AdditiveEFF"

    if MODE == "RESNETONLY":
        base_model = models.resnet50(pretrained=True)
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
        model = base_model
        model_name = 'resnet'
    elif MODE == "EFFNETONLY":
        effnet_version = 'efficientnet_b4'
        base_model = getattr(models, effnet_version)(pretrained=True)
        base_model.classifier[1] = nn.Linear(base_model.classifier[1].in_features, num_classes)
        model = base_model
        model_name = 'effnet'
    elif MODE == "AdditiveRES":
        base_model = models.resnet50(pretrained=True)
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
        img_block = torch.nn.Sequential(*(list(base_model.children())[:-1]))
        model = EffNetMetaImageAttentionV3(img_block, AdditiveMetaAttentionBlock, [4, 5, 6], num_classes, d_meta, dropout=0.3, embed_dim=512)
        model_name = 'resnetadditive'
    elif MODE == "ScaledDotRES":
        base_model = models.resnet50(pretrained=True)
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
        img_block = torch.nn.Sequential(*(list(base_model.children())[:-1]))
        model = EffNetMetaImageAttentionV3(img_block, ScaledDotMetaAttentionBlock, [4, 5, 6], num_classes, d_meta, dropout=0.3, embed_dim=512)
        model_name = 'resnetscaleddot'
    elif MODE == "AdditiveEFF":
        effnet_version = 'efficientnet_b4'
        base_model = getattr(models, effnet_version)(pretrained=True)
        img_block = base_model.features
        model = EffNetMetaImageAttentionV3(img_block, AdditiveMetaAttentionBlock, [3, 5, 8], num_classes, d_meta, dropout=0.3, embed_dim=512)
        model_name = 'effnetadditive'
    elif MODE == "ScaledDotEFF":
        effnet_version = 'efficientnet_b4'
        base_model = getattr(models, effnet_version)(pretrained=True)
        img_block = base_model.features
        model = EffNetMetaImageAttentionV3(img_block, ScaledDotMetaAttentionBlock, [3, 5, 8], num_classes, d_meta, dropout=0.3, embed_dim=512)
        model_name = 'effnetscaleddot'
    elif MODE == "ResMetablock":
        base_model = models.resnet50(pretrained=True)
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
        img_block = torch.nn.Sequential(*(list(base_model.children())[:-1]))
        num_hidden = 1024
        metadata_block = nn.Sequential(
                  nn.LazyLinear(1024),
                  nn.BatchNorm1d(1024),
                  nn.ReLU()
        )
        metablock = WrappedMetaBlock(MetaBlock(64, 1024))
        classifier = nn.Sequential(
            nn.LazyLinear(num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, num_classes)
            # NormalizedLinear(num_hidden, num_classes)
        ) 
        model = CombineModel(img_block, metadata_block, metablock, classifier, freeze_conv=False)
        model_name = 'resnetmetablock'

    version_model = 'v1'
    model_path = os.path.join(MODELPATH, model_name, version_model)

    warmup_epochs = 50
    total_epochs = 150

    lit_model = BaseLitModel(
        model=model,
        loss=ce_loss,
        get_optim=optim,
        lr=lr,
        get_lr_scheduler=lr_sched
    )
    
    # Warm up model by freezing Conv layers
    model.freeze_img_conv()
    train(lit_model, warmup_epochs, data, model_path, model_name, version_model)

    # Unfreeze for full train
    model.unfreeze_img_conv()
    train(lit_model, total_epochs - warmup_epochs, data, model_path, model_name, version_model)

