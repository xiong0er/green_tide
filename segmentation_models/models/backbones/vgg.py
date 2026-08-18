"""
VGG骨干网络 - 用于SegNet等模型
"""
import torch
import torch.nn as nn


class VGGBackbone(nn.Module):
    """VGG骨干网络"""
    def __init__(self, in_channels: int = 3, pretrained: bool = False):
        super().__init__()
        self.in_channels = in_channels

        # VGG配置
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                   512, 512, 512, 'M', 512, 512, 512, 'M']

        self.features = self._make_layers(in_channels)

        # 编码器输出通道
        self.out_channels = 512

    def _make_layers(self, in_channels):
        layers = []
        for v in self.cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(2, 2, return_indices=True))
            else:
                conv = nn.Conv2d(in_channels, v, 3, padding=1)
                layers.extend([conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
                in_channels = v
        return nn.ModuleList(layers)

    def forward(self, x):
        indices = []
        sizes = []

        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                sizes.append(x.size())
                x, idx = layer(x)
                indices.append(idx)
            else:
                x = layer(x)

        return x, indices, sizes


class VGGDecoder(nn.Module):
    """VGG解码器 - 用于SegNet"""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.cfg = [512, 512, 512, 256, 256, 256, 128, 128, 64, 64]

        layers = []
        in_channels = 512
        for v in self.cfg:
            layers.append(nn.Conv2d(in_channels, v, 3, padding=1))
            layers.append(nn.BatchNorm2d(v))
            layers.append(nn.ReLU(inplace=True))
            in_channels = v

        self.features = nn.ModuleList(layers)
        self.classifier = nn.Conv2d(64, num_classes, 1)

    def forward(self, x, indices, sizes):
        idx = len(indices) - 1

        for layer in self.features:
            if isinstance(layer, nn.Conv2d) and x.size() != sizes[idx]:
                # 上采样
                x = nn.MaxUnpool2d(2, 2)(x, indices[idx], output_size=sizes[idx])
                idx = max(0, idx - 1)
            x = layer(x)

        return self.classifier(x)


if __name__ == "__main__":
    x = torch.randn(2, 3, 512, 512)
    encoder = VGGBackbone(3)
    decoder = VGGDecoder(2)

    feat, indices, sizes = encoder(x)
    print(f"Encoder输出: {feat.shape}")

    out = decoder(feat, indices, sizes)
    print(f"Decoder输出: {out.shape}")
