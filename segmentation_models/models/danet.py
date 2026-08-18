"""
DANet模型 - 双重注意力网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones.resnet import ResNetBackbone


class PAM(nn.Module):
    """位置注意力模块 (Position Attention Module)"""
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()

        # 生成Q, K, V
        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # B, HW, C'
        key = self.key_conv(x).view(B, -1, H * W)  # B, C', HW
        value = self.value_conv(x).view(B, -1, H * W)  # B, C, HW

        # 注意力
        attention = self.softmax(torch.bmm(query, key))  # B, HW, HW

        # 加权
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B, C, HW
        out = out.view(B, C, H, W)

        out = self.gamma * out + x
        return out


class CAM(nn.Module):
    """通道注意力模块 (Channel Attention Module)"""
    def __init__(self, in_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()

        # 生成特征
        query = x.view(B, C, -1)  # B, C, HW
        key = x.view(B, C, -1).permute(0, 2, 1)  # B, HW, C
        value = x.view(B, C, -1)  # B, C, HW

        # 注意力
        attention = self.softmax(torch.bmm(query, key))  # B, C, C

        # 加权
        out = torch.bmm(attention, value)  # B, C, HW
        out = out.view(B, C, H, W)

        out = self.gamma * out + x
        return out


class DANetHead(nn.Module):
    """DANet头"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        inter_channels = in_channels // 4

        # 位置注意力分支
        self.pam_conv = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.pam = PAM(inter_channels)
        self.pam_out = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(inter_channels, num_classes, 1)
        )

        # 通道注意力分支
        self.cam_conv = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.cam = CAM(inter_channels)
        self.cam_out = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(inter_channels, num_classes, 1)
        )

        # 融合
        self.fusion = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels, num_classes, 1)
        )

    def forward(self, x):
        # 位置注意力
        pam_feat = self.pam_conv(x)
        pam_feat = self.pam(pam_feat)
        pam_out = self.pam_out(pam_feat)

        # 通道注意力
        cam_feat = self.cam_conv(x)
        cam_feat = self.cam(cam_feat)
        cam_out = self.cam_out(cam_feat)

        # 融合
        feat_sum = pam_feat + cam_feat
        fusion_out = self.fusion(feat_sum)

        if self.training:
            return [pam_out, cam_out, fusion_out]
        else:
            return fusion_out


class DANet(nn.Module):
    """
    DANet模型
    论文: Dual Attention Network for Scene Segmentation
    """
    def __init__(self, in_channels=3, num_classes=2, backbone='resnet50',
                 pretrained=False):
        super().__init__()
        self.num_classes = num_classes

        # 编码器
        self.encoder = ResNetBackbone(
            in_channels=in_channels,
            arch=backbone,
            pretrained=pretrained
        )

        # 解码器
        self.head = DANetHead(self.encoder.out_channels[-1], num_classes)

    def forward(self, x):
        input_size = x.size()[2:]

        # 编码器
        features = self.encoder(x)
        x = features['c4']

        # 解码器
        outputs = self.head(x)

        # 上采样
        if self.training:
            return [F.interpolate(out, size=input_size, mode='bilinear',
                                 align_corners=False) for out in outputs]
        else:
            return F.interpolate(outputs, size=input_size, mode='bilinear',
                                align_corners=False)


if __name__ == "__main__":
    x = torch.randn(2, 3, 512, 512)

    model = DANet(3, 2, 'resnet50')
    model.train()
    out = model(x)
    print(f"DANet训练输出数量: {len(out)}")
    for i, o in enumerate(out):
        print(f"  输出{i}: {o.shape}")

    model.eval()
    out = model(x)
    print(f"DANet推理输出: {out.shape}")
