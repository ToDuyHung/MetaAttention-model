import torch
import torch.nn as nn
from SoftAttention import SoftAttention
class MetaBlock(nn.Module):
    def __init__(self, V, U):
        super(MetaBlock, self).__init__()
        self.fb = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))
        self.gb = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))

    def forward(self, V, U):
        t1 = self.fb(U)
        t2 = self.gb(U)
        V = torch.sigmoid(torch.tanh(V * t1.unsqueeze(-1)) + t2.unsqueeze(-1))
        return V
    
class ConcatBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X_img, X_meta):
        return torch.cat([X_img, X_meta], dim=1)

class MetaNet(nn.Module):
    """
    Implementing the MetaNet approach
    Fusing Metadata and Dermoscopy Images for Skin Disease Diagnosis - https://ieeexplore.ieee.org/document/9098645
    """
    def __init__(self, in_channels, middle_channels, out_channels):
        super(MetaNet, self).__init__()
        self.metanet = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 1), # N * mid * 1 * 1 
            nn.ReLU(),
            nn.Conv2d(middle_channels, out_channels, 1), # N * out * 1 * 1
            nn.Sigmoid()
        )

    def forward(self, feat_maps, metadata):
        metadata = torch.unsqueeze(metadata, -1) # N * d * 1
        metadata = torch.unsqueeze(metadata, -1) # N * d * 1 * 1
        x = self.metanet(metadata)
        x = x * feat_maps
        return x

class WrappedMetaBlock(nn.Module):
    def __init__(self, metablock):
        super().__init__()
        self.metablock = metablock
        self.feat_maps = self.metablock.fb[0].out_features
    
    def forward(self, X_img, X_meta):
        X_img = X_img.view(X_img.size(0), self.feat_maps, -1)
        return self.metablock(X_img, X_meta).flatten(1)

class ConcatBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X_img, X_meta):
        return torch.cat([X_img, X_meta], dim=1)

class CombineModel(nn.Module):
    def __init__(self, img_block, metadata_block, combine_block, classifier, freeze_conv=False):
        assert img_block and combine_block and classifier
        super().__init__()
        self.img_block = img_block
        if freeze_conv:
            for param in self.img_block.parameters():
                param.requires_grad = False
        self.metadata_block = metadata_block
        self.combine_block = combine_block
        self.classifier = classifier
    
    def forward(self, X, get_attention=False):
        X_img, X_meta = X
        X_img = self.img_block(X_img)

        if self.metadata_block:
            X_meta = self.metadata_block(X_meta)
        if isinstance(self.combine_block, MetaNet) or isinstance(self.combine_block, SoftAttention):
            y_tmp = self.combine_block(X_img, X_meta)
            y = y_tmp[0].flatten(1)
            if not get_attention:
                return self.classifier(y)
            else:
                return self.classifier(y), y_tmp[1]
        X_img = X_img.flatten(1)
        y = self.combine_block(X_img, X_meta)
    
        return self.classifier(y)
    def unfreeze_img_conv(self):
        for param in self.img_block.parameters():
            param.requires_grad = True
    
    def freeze_img_conv(self):
        for param in self.img_block.parameters():
            param.requires_grad = False