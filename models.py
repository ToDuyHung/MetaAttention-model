import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    