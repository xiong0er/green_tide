"""
模型基础模块和工具函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


class ConvBlock(nn.Module):
    """基础卷积块: Conv + BN + ReLU"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 padding: int = 1, stride: int = 1, bias: bool = False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    """双卷积块"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_ch, out_ch),
            ConvBlock(out_ch, out_ch)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """下采样: MaxPool + DoubleConv"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """上采样: Upsample/ConvTranspose + DoubleConv"""
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 处理尺寸不匹配
        diff_h = x2.size()[2] - x1.size()[2]
        diff_w = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    """注意力门控"""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ASPP(nn.Module):
    """空洞空间金字塔池化 (Atrous Spatial Pyramid Pooling)"""
    def __init__(self, in_ch: int, out_ch: int, rates: List[int] = [6, 12, 18]):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ) for rate in rates
        ])

        self.branch_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 2), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]

        feat1 = self.branch1(x)
        feats = [branch(x) for branch in self.branches]

        feat_pool = self.branch_pool(x)
        feat_pool = F.interpolate(feat_pool, size=size, mode='bilinear', align_corners=False)

        out = torch.cat([feat1] + feats + [feat_pool], dim=1)
        return self.conv_cat(out)


class PyramidPooling(nn.Module):
    """金字塔池化模块 (PSPNet使用)"""
    def __init__(self, in_ch: int, out_ch: int, sizes: List[int] = [1, 2, 3, 6]):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_ch, out_ch // len(sizes), 1, bias=False),
                nn.BatchNorm2d(out_ch // len(sizes)),
                nn.ReLU(inplace=True)
            ) for size in sizes
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch + out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]
        feats = [x]
        for stage in self.stages:
            feats.append(F.interpolate(stage(x), size=size, mode='bilinear', align_corners=False))
        return self.bottleneck(torch.cat(feats, dim=1))


class ChannelAttention(nn.Module):
    """通道注意力 (SE-Net / ECA-Net风格)"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """空间注意力"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention


class CBAM(nn.Module):
    """卷积块注意力模块 (Channel + Spatial)"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class DropBlock(nn.Module):
    """DropBlock正则化"""
    def __init__(self, block_size: int = 7, keep_prob: float = 0.9):
        super().__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob

    def forward(self, x):
        if not self.training or self.keep_prob == 1.0:
            return x

        gamma = (1 - self.keep_prob) / (self.block_size ** 2)
        mask = torch.bernoulli(torch.ones_like(x) * gamma)
        mask = 1 - F.max_pool2d(mask, self.block_size, 1, self.block_size//2)
        return x * mask * mask.numel() / mask.sum()


def initialize_weights(module):
    """初始化模型权重"""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0, 0.01)
        nn.init.constant_(module.bias, 0)


class SegmentationHead(nn.Module):
    """分割头"""
    def __init__(self, in_ch: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv = nn.Conv2d(in_ch, num_classes, 1)

    def forward(self, x):
        x = self.dropout(x)
        return self.conv(x)


if __name__ == "__main__":
    # 测试模块
    x = torch.randn(2, 64, 32, 32)

    # 测试ASPP
    aspp = ASPP(64, 128)
    out = aspp(x)
    print(f"ASPP输入: {x.shape}, 输出: {out.shape}")

    # 测试CBAM
    cbam = CBAM(64)
    out = cbam(x)
    print(f"CBAM输入: {x.shape}, 输出: {out.shape}")
