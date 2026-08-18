"""
U-Net模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.blocks import DoubleConv, Down, Up, AttentionGate


class UNet(nn.Module):
    """
    U-Net模型
    论文: U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 base_channels: int = 64, bilinear: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # 编码器
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)

        # 解码器
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)

        # 输出层
        self.outc = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, x):
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码器
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 输出
        logits = self.outc(x)
        return logits


class AttentionUNet(nn.Module):
    """
    Attention U-Net
    在跳跃连接中加入注意力门控
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 base_channels: int = 64, bilinear: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # 编码器
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)

        # 注意力门
        self.att1 = AttentionGate(base_channels * 16 // factor, base_channels * 8, base_channels * 8 // 2)
        self.att2 = AttentionGate(base_channels * 8 // factor, base_channels * 4, base_channels * 4 // 2)
        self.att3 = AttentionGate(base_channels * 4 // factor, base_channels * 2, base_channels * 2 // 2)
        self.att4 = AttentionGate(base_channels * 2 // factor, base_channels, base_channels // 2)

        # 解码器
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)

        self.outc = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, x):
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 带注意力的解码器（先上采样g再应用注意力）
        x = self.up1(x5, self.att1(F.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=False), x4))
        x = self.up2(x, self.att2(F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=False), x3))
        x = self.up3(x, self.att3(F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False), x2))
        x = self.up4(x, self.att4(F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False), x1))

        return self.outc(x)


class NestedUNet(nn.Module):
    """
    U-Net++ (Nested U-Net)
    论文: UNet++: A Nested U-Net Architecture for Medical Image Segmentation
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 base_channels: int = 32, deep_supervision: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        filters = [base_channels, base_channels * 2, base_channels * 4,
                  base_channels * 8, base_channels * 16]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 编码器
        self.conv0_0 = DoubleConv(in_channels, filters[0])
        self.conv1_0 = DoubleConv(filters[0], filters[1])
        self.conv2_0 = DoubleConv(filters[1], filters[2])
        self.conv3_0 = DoubleConv(filters[2], filters[3])
        self.conv4_0 = DoubleConv(filters[3], filters[4])

        # 解码器 - 嵌套结构
        self.conv0_1 = DoubleConv(filters[0] + filters[1], filters[0])
        self.conv1_1 = DoubleConv(filters[1] + filters[2], filters[1])
        self.conv2_1 = DoubleConv(filters[2] + filters[3], filters[2])
        self.conv3_1 = DoubleConv(filters[3] + filters[4], filters[3])

        self.conv0_2 = DoubleConv(filters[0] * 2 + filters[1], filters[0])
        self.conv1_2 = DoubleConv(filters[1] * 2 + filters[2], filters[1])
        self.conv2_2 = DoubleConv(filters[2] * 2 + filters[3], filters[2])

        self.conv0_3 = DoubleConv(filters[0] * 3 + filters[1], filters[0])
        self.conv1_3 = DoubleConv(filters[1] * 3 + filters[2], filters[1])

        self.conv0_4 = DoubleConv(filters[0] * 4 + filters[1], filters[0])

        self.final = nn.Conv2d(filters[0], num_classes, 1)

    def forward(self, x):
        # 编码器
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            return [self.final(x0_1), self.final(x0_2), self.final(x0_3), self.final(x0_4)]
        else:
            return self.final(x0_4)


if __name__ == "__main__":
    x = torch.randn(2, 3, 512, 512)

    # 测试UNet
    model = UNet(3, 2)
    out = model(x)
    print(f"UNet输出: {out.shape}")

    # 测试AttentionUNet
    model = AttentionUNet(3, 2)
    out = model(x)
    print(f"AttentionUNet输出: {out.shape}")

    # 测试NestedUNet
    model = NestedUNet(3, 2)
    out = model(x)
    print(f"NestedUNet输出: {out.shape}")
