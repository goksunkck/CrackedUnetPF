"""
Attention U-Net with GroupNorm instead of BatchNorm.
GroupNorm is more stable for small batches and domain shift scenarios.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_num_groups(num_channels, min_groups=1, max_groups=32):
    """
    Calculate number of groups for GroupNorm.
    Groups must divide num_channels evenly.
    """
    for g in [32, 16, 8, 4, 2, 1]:
        if num_channels >= g and num_channels % g == 0:
            return min(g, max_groups)
    return min_groups


class DoubleConvGN(nn.Module):
    """(convolution => [GN] => LeakyReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        num_groups = get_num_groups(out_channels)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownGN(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvGN(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class AttentionBlockGN(nn.Module):
    """Attention Gate with GroupNorm"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlockGN, self).__init__()
        num_groups_int = get_num_groups(F_int)
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups_int, F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups_int, F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, 1),  # 1 group for 1 channel
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UpAttentionGN(nn.Module):
    """Upscaling with Attention Gate then double conv (GroupNorm)"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvGN(in_channels, out_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvGN(in_channels, out_channels)
            
        self.att = AttentionBlockGN(F_g=in_channels // 2, F_l=in_channels // 2, F_int=in_channels // 4)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x2_att = self.att(g=x1, x=x2)
        x = torch.cat([x2_att, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AttentionUNetGN(nn.Module):
    """
    Attention U-Net with GroupNorm instead of BatchNorm.
    More stable for domain shift and small batch scenarios.
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(AttentionUNetGN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConvGN(n_channels, 64)
        self.down1 = DownGN(64, 128)
        self.down2 = DownGN(128, 256)
        self.down3 = DownGN(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownGN(512, 1024 // factor)
        self.dropout = nn.Dropout(p=0.2)
        
        # Attention Up Blocks
        self.up1 = UpAttentionGN(1024, 1024 // factor, bilinear)
        self.up2 = UpAttentionGN(512, 512 // factor, bilinear)
        self.up3 = UpAttentionGN(256, 256 // factor, bilinear)
        self.up4 = UpAttentionGN(128, 128, bilinear)
        
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x5 = self.dropout(x5)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits
