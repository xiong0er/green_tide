"""
SegNet模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegNet(nn.Module):
    """
    SegNet模型
    论文: SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation (2015)
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # 编码器
        self.encoder_conv = nn.ModuleList([
            self._make_conv_block(in_channels, 64, 2),
            self._make_conv_block(64, 128, 2),
            self._make_conv_block(128, 256, 3),
            self._make_conv_block(256, 512, 3),
            self._make_conv_block(512, 512, 3)
        ])

        self.encoder_pools = nn.ModuleList([
            nn.MaxPool2d(2, 2, return_indices=True) for _ in range(5)
        ])

        # 解码器
        self.decoder_unpools = nn.ModuleList([
            nn.MaxUnpool2d(2, 2) for _ in range(5)
        ])

        self.decoder_conv = nn.ModuleList([
            self._make_conv_block(512, 512, 3),
            self._make_conv_block(512, 256, 3),
            self._make_conv_block(256, 128, 3),
            self._make_conv_block(128, 64, 2),
            self._make_conv_block(64, 64, 2)
        ])

        # 分类层
        self.classifier = nn.Conv2d(64, num_classes, 1)

    def _make_conv_block(self, in_ch, out_ch, num_layers):
        """创建卷积块"""
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*layers)

    def forward(self, x):
        # 编码器
        indices = []
        sizes = []

        for conv, pool in zip(self.encoder_conv, self.encoder_pools):
            x = conv(x)
            sizes.append(x.size())
            x, idx = pool(x)
            indices.append(idx)

        # 解码器
        for unpool, conv in zip(self.decoder_unpools, self.decoder_conv):
            idx = indices.pop()
            size = sizes.pop()
            x = unpool(x, idx, output_size=size)
            x = conv(x)

        # 分类
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 3, 512, 512)
    model = SegNet(3, 2)
    out = model(x)
    print(f"SegNet输出: {out.shape}")
