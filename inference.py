import os
import torch
from torchvision import models
import torch.nn as nn
from datamodule import HAM10000_ABCD_DataModule

if __name__ == "__main__":
    data = HAM10000_ABCD_DataModule(data_path=DATAPATH,
                                img_size = 224,
                                batch_size = 32,
                                sampler_func=get_weighted_random_sampler,
                                mode='', use_meta=True)
    data.setup()
    num_classes=7
    num_hidden = 1024
    d_meta = data.val_ds[0][0][1].shape[0]
    print(d_meta)

    MODE = "AdditiveEFF"

    if MODE == "RESNETONLY":
        base_model = models.resnet50(pretrained=True)
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
        model = base_model
    elif MODE == "EFFNETONLY":
        effnet_version = 'efficientnet_b4'
        base_model = getattr(models, effnet_version)(pretrained=True)
        base_model.classifier[1] = nn.Linear(base_model.classifier[1].in_features, num_classes)
        model = base_model
    elif MODE == "AdditiveRES":
        base_model = models.resnet50(pretrained=True)
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
        img_block = torch.nn.Sequential(*(list(base_model.children())[:-1]))
        model = EffNetMetaImageAttentionV3(img_block, AdditiveMetaAttentionBlock, [4, 5, 6], num_classes, d_meta, dropout=0.3, embed_dim=512)
    elif MODE == "ScaledDotRES":
        base_model = models.resnet50(pretrained=True)
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
        img_block = torch.nn.Sequential(*(list(base_model.children())[:-1]))
        model = EffNetMetaImageAttentionV3(img_block, ScaledDotMetaAttentionBlock, [4, 5, 6], num_classes, d_meta, dropout=0.3, embed_dim=512)
    elif MODE == "AdditiveEFF":
        effnet_version = 'efficientnet_b4'
        base_model = getattr(models, effnet_version)(pretrained=True)
        img_block = base_model.features
        model = EffNetMetaImageAttentionV3(img_block, AdditiveMetaAttentionBlock, [3, 5, 8], num_classes, d_meta, dropout=0.3, embed_dim=512)
    elif MODE == "ScaledDotEFF":
        effnet_version = 'efficientnet_b4'
        base_model = getattr(models, effnet_version)(pretrained=True)
        img_block = base_model.features
        model = EffNetMetaImageAttentionV3(img_block, ScaledDotMetaAttentionBlock, [3, 5, 8], num_classes, d_meta, dropout=0.3, embed_dim=512)
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