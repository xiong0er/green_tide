"""
DeepLabV3+模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones.resnet import ResNetDilatedBackbone
from models.utils.blocks import ASPP, ConvBlock


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+模型
    论文: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (2018)
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 backbone: str = 'resnet50', output_stride: int = 16,
                 pretrained: bool = False):
        super().__init__()
        self.num_classes = num_classes

        # 编码器 (带空洞卷积的ResNet)
        self.encoder = ResNetDilatedBackbone(
            in_channels=in_channels,
            arch=backbone,
            output_stride=output_stride,
            pretrained=pretrained
        )

        encoder_channels = self.encoder.out_channels

        # ASPP模块
        self.aspp = ASPP(encoder_channels, 256, rates=self._get_rates(output_stride))

        # 低层特征处理
        self.low_level_conv = nn.Sequential(
            ConvBlock(256 if 'resnet50' in backbone else 64, 48, 1, padding=0),
            nn.Dropout2d(0.5)
        )

        # 解码器
        self.decoder = nn.Sequential(
            ConvBlock(256 + 48, 256, 3),
            nn.Dropout2d(0.5),
            ConvBlock(256, 256, 3)
        )

        # 分类头
        self.classifier = nn.Conv2d(256, num_classes, 1)

        self._init_weights()

    def _get_rates(self, output_stride):
        """根据输出步长计算空洞率"""
        if output_stride == 16:
            return [6, 12, 18]
        elif output_stride == 8:
            return [12, 24, 36]
        else:
            return [6, 12, 18]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_size = x.size()[2:]

        # 编码器
        x = self.encoder(x)

        # ASPP
        x = self.aspp(x)

        # 解码器上采样
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        # 低层特征（这里简化处理，使用固定的1/4特征）
        low_level_feat = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        low_level_feat = self.low_level_conv(low_level_feat)
        low_level_feat = F.interpolate(low_level_feat, size=x.size()[2:], mode='bilinear', align_corners=False)

        # 融合
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.decoder(x)

        # 分类
        x = self.classifier(x)

        # 上采样到原始尺寸
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        return x


if __name__ == "__main__":
    x = torch.randn(2, 3, 512, 512)
    model = DeepLabV3Plus(3, 2, 'resnet50')
    out = model(x)
    print(f"DeepLabV3+输出: {out.shape}")
