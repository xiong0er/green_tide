"""
CCNet模型 - Criss-Cross Attention Network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones.resnet import ResNetBackbone


class CrissCrossAttention(nn.Module):
    """Criss-Cross注意力"""
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()

        # Q, K, V
        query = self.query_conv(x)  # B, C', H, W
        key = self.key_conv(x)      # B, C', H, W
        value = self.value_conv(x)  # B, C, H, W

        # 水平注意力 (Q @ K^T)
        query_h = query.permute(0, 3, 1, 2).contiguous().view(B * W, C // 8, H)  # BW, C', H
        key_h = key.permute(0, 3, 1, 2).contiguous().view(B * W, C // 8, H)  # BW, C', H
        value_h = value.permute(0, 3, 1, 2).contiguous().view(B * W, C, H)  # BW, C, H

        energy_h = torch.bmm(query_h, key_h.permute(0, 2, 1))  # BW, H, H
        attention_h = self.softmax(energy_h)
        out_h = torch.bmm(value_h, attention_h.permute(0, 2, 1))  # BW, C, H
        out_h = out_h.view(B, W, C, H).permute(0, 2, 3, 1)  # B, C, H, W

        # 垂直注意力
        query_v = query.permute(0, 2, 1, 3).contiguous().view(B * H, C // 8, W)  # BH, C', W
        key_v = key.permute(0, 2, 1, 3).contiguous().view(B * H, C // 8, W)  # BH, C', W
        value_v = value.permute(0, 2, 1, 3).contiguous().view(B * H, C, W)  # BH, C, W

        energy_v = torch.bmm(query_v, key_v.permute(0, 2, 1))  # BH, W, W
        attention_v = self.softmax(energy_v)
        out_v = torch.bmm(value_v, attention_v.permute(0, 2, 1))  # BH, C, W
        out_v = out_v.view(B, H, C, W).permute(0, 2, 1, 3)  # B, C, H, W

        # 融合
        out = out_h + out_v
        out = self.gamma * out + x

        return out


class RCCAModule(nn.Module):
    """RCCA模块 - 递归Criss-Cross注意力"""
    def __init__(self, in_channels, num_classes, recurrence=2):
        super().__init__()
        self.recurrence = recurrence
        inter_channels = in_channels // 4

        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        self.cca = CrissCrossAttention(inter_channels)

        self.conv_out = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(inter_channels, num_classes, 1)
        )

    def forward(self, x):
        x = self.conv_in(x)

        for _ in range(self.recurrence):
            x = self.cca(x)

        x = self.conv_out(x)
        return x


class CCNet(nn.Module):
    """
    CCNet模型
    论文: CCNet: Criss-Cross Attention for Semantic Segmentation
    """
    def __init__(self, in_channels=3, num_classes=2, backbone='resnet50',
                 pretrained=False, recurrence=2):
        super().__init__()
        self.num_classes = num_classes
        self.recurrence = recurrence

        # 编码器
        self.encoder = ResNetBackbone(
            in_channels=in_channels,
            arch=backbone,
            pretrained=pretrained
        )

        # RCCA模块
        self.head = RCCAModule(self.encoder.out_channels[-1], num_classes, recurrence)

        # 辅助分类头
        self.aux_head = nn.Sequential(
            nn.Conv2d(self.encoder.out_channels[2], 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, 1)
        )

    def forward(self, x):
        input_size = x.size()[2:]

        # 编码器
        features = self.encoder(x)
        x = features['c4']
        aux_feat = features['c3']

        # 主分支
        out = self.head(x)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)

        # 训练时使用辅助分支
        if self.training:
            aux_out = self.aux_head(aux_feat)
            aux_out = F.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=False)
            return out, aux_out
        else:
            return out


if __name__ == "__main__":
    x = torch.randn(2, 3, 512, 512)

    model = CCNet(3, 2, 'resnet50', recurrence=2)
    model.train()
    out, aux = model(x)
    print(f"CCNet训练输出: main={out.shape}, aux={aux.shape}")

    model.eval()
    out = model(x)
    print(f"CCNet推理输出: {out.shape}")
