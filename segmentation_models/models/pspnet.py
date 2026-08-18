"""
PSPNet模型 - 金字塔场景解析网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones.resnet import ResNetBackbone
from models.utils.blocks import PyramidPooling


class PSPNet(nn.Module):
    """
    PSPNet模型
    论文: Pyramid Scene Parsing Network (2017)
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 backbone: str = 'resnet50', pretrained: bool = False,
                 sizes: list = [1, 2, 3, 6]):
        super().__init__()
        self.num_classes = num_classes

        # 编码器
        self.encoder = ResNetBackbone(
            in_channels=in_channels,
            arch=backbone,
            pretrained=pretrained
        )

        # 金字塔池化
        encoder_out = self.encoder.out_channels[-1]
        self.psp = PyramidPooling(encoder_out, 512, sizes=sizes)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

        # 辅助分类头（用于训练）
        self.aux_classifier = nn.Sequential(
            nn.Conv2d(self.encoder.out_channels[2], 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

        self._init_weights()

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
        features = self.encoder(x)
        c4 = features['c4']  # 1/32
        c3 = features['c3']  # 1/16

        # 金字塔池化
        x = self.psp(c4)

        # 主分类
        main_out = self.classifier(x)
        main_out = F.interpolate(main_out, size=input_size, mode='bilinear', align_corners=False)

        # 训练时使用辅助分类头
        if self.training:
            aux_out = self.aux_classifier(c3)
            aux_out = F.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=False)
            return main_out, aux_out
        else:
            return main_out


if __name__ == "__main__":
    x = torch.randn(2, 3, 512, 512)
    model = PSPNet(3, 2, 'resnet50')

    model.train()
    out, aux = model(x)
    print(f"PSPNet训练输出: main={out.shape}, aux={aux.shape}")

    model.eval()
    out = model(x)
    print(f"PSPNet推理输出: {out.shape}")
