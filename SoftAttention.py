import torch.nn as nn
import torch
import torch.nn.functional as F
class SoftAttention(nn.Module):
    def __init__(self, out_channels, ch, in_channels, middle_channels, aggregate=False):
        super().__init__()
        self.channels = ch
        self.conv3d = nn.Conv3d(1, out_channels, kernel_size=(ch, 3, 3), stride=(ch, 1, 1), padding=(0, 1, 1))

        self.conv5d = nn.Conv3d(1, out_channels, kernel_size=(ch, 5, 5), stride=(ch, 1, 1), padding=(0, 2, 2))
        self.conv7d = nn.Conv3d(1, out_channels, kernel_size=(ch, 7, 7), stride=(ch, 1, 1), padding=(0, 3, 3))

        self.aggregate_channels = aggregate
        self.metanet = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 1), # N * mid * 1 * 1 
            nn.ReLU(),
            nn.Conv2d(middle_channels, ch, 1), # N * out * 1 * 1
            nn.Sigmoid()
        )
        
    def forward(self, X, metadata):
        # X shape: N x ch x H x W
        # X_exp shape: N x 1 x ch x H x W
        X_exp = X.unsqueeze(1)
        conv3d = F.relu(self.conv3d(X_exp)) # N x out_channels x 1 x H x W
        conv3d = conv3d.squeeze(2) # N x out_channels x H x W
        old_shape = conv3d.shape
        conv3d = conv3d.view(conv3d.shape[0], conv3d.shape[1], -1) # N x out_channels x H*W
        softmax_alpha = F.softmax(conv3d, dim=-1)
        softmax_alpha_3 = softmax_alpha.view(old_shape) # N x out_channels x H x W

        # ----------------- Add kernel size 5x5 and 7x7 -------------------------
        conv5d = F.relu(self.conv5d(X_exp)) # N x out_channels x 1 x H x W
        conv5d = conv5d.squeeze(2) # N x out_channels x H x W
        old_shape = conv5d.shape
        conv5d = conv5d.view(conv5d.shape[0], conv5d.shape[1], -1) # N x out_channels x H*W
        softmax_alpha = F.softmax(conv5d, dim=-1)
        softmax_alpha_5 = softmax_alpha.view(old_shape) # N x out_channels x H x W

        conv7d = F.relu(self.conv7d(X_exp)) # N x out_channels x 1 x H x W
        conv7d = conv7d.squeeze(2) # N x out_channels x H x W
        old_shape = conv7d.shape
        conv7d = conv7d.view(conv7d.shape[0], conv7d.shape[1], -1) # N x out_channels x H*W
        softmax_alpha = F.softmax(conv7d, dim=-1)
        softmax_alpha_7 = softmax_alpha.view(old_shape) # N x out_channels x H x W
        # -------------------------------------------------------------------------

        if self.aggregate_channels:
            # Back up
            # softmax_alpha = softmax_alpha.sum(dim=1, keepdim=True) # N x 1 x H x W

            softmax_alpha_3 = softmax_alpha_3.sum(dim=1, keepdim=True) # N x 1 x H x W
            softmax_alpha_5 = softmax_alpha_5.sum(dim=1, keepdim=True) # N x 1 x H x W
            softmax_alpha_7 = softmax_alpha_7.sum(dim=1, keepdim=True) # N x 1 x H x W

            softmax_alpha = softmax_alpha_3 + softmax_alpha_5 + softmax_alpha_7

            metadata = torch.unsqueeze(metadata, -1)
            metadata = torch.unsqueeze(metadata, -1)
            x = self.metanet(metadata)
            softmax_alpha = x + softmax_alpha

            y = softmax_alpha * X # y shape: N x ch x H x W
        else:
            softmax_alpha = softmax_alpha.unsqueeze(2) # N x out_channels x 1 x H x W
            y = softmax_alpha * X_exp # y shape: N x out_channels x ch x H x W
            y = y.view(y.shape[0], -1, y.shape[3], y.shape[4]) # N x out_channels * ch x H x W
        return [y, softmax_alpha]


def init_softattention_weights(m):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.zeros_(m.bias.data)

class AttentionModel(nn.Module):
    def __init__(self, img_conv, multiheads=16, aggregate=True, freeze_img_conv=False):
        super().__init__()
        self.img_conv = img_conv
        self.attention = SoftAttention(multiheads, ch=img_conv[-1].out_channels, aggregate=True)
        self.attention.apply(init_softattention_weights)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout = nn.Dropout(0.5)
        if freeze_img_conv:
            self.freeze_img_conv()

    def unfreeze_img_conv(self):
        for param in self.img_conv.parameters():
            param.requires_grad = True
    
    def freeze_img_conv(self):
        for param in self.img_conv.parameters():
            param.requires_grad = False

    def forward(self, X, get_attn_maps=False):
        X_img = self.img_conv(X)
        context, map = self.attention(X_img)
        X_img = self.maxpool(X_img)
        context = self.maxpool(context)
        output = torch.cat([X_img, context], dim=1)
        output = self.dropout(F.relu(output))
        if get_attn_maps:
            return (output, map)
        return output