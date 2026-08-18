"""
ResNet骨干网络
"""
import torch
import torch.nn as nn
import torchvision.models as models


class ResNetBackbone(nn.Module):
    """ResNet骨干网络"""
    def __init__(self, in_channels: int = 3, arch: str = 'resnet50', pretrained: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.arch = arch

        # 加载预训练模型
        if arch == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            self.out_channels = [64, 128, 256, 512]
        elif arch == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            self.out_channels = [64, 128, 256, 512]
        elif arch == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.out_channels = [256, 512, 1024, 2048]
        elif arch == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
            self.out_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"不支持的架构: {arch}")

        # 修改第一层以适应不同输入通道
        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        else:
            self.conv1 = resnet.conv1

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)   # 1/4
        c2 = self.layer2(c1)  # 1/8
        c3 = self.layer3(c2)  # 1/16
        c4 = self.layer4(c3)  # 1/32

        return {
            'c1': c1,
            'c2': c2,
            'c3': c3,
            'c4': c4
        }


class ResNetDilatedBackbone(nn.Module):
    """带空洞卷积的ResNet - 用于DeepLab"""
    def __init__(self, in_channels: int = 3, arch: str = 'resnet50',
                 output_stride: int = 16, pretrained: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.arch = arch
        self.output_stride = output_stride

        # 加载预训练模型
        if arch == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.out_channels = 2048
        elif arch == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
            self.out_channels = 2048
        else:
            raise ValueError(f"不支持的架构: {arch}")

        # 修改第一层
        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        else:
            self.conv1 = resnet.conv1

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2

        # 根据output_stride设置空洞率
        if output_stride == 16:
            self.layer3 = self._make_dilated_layer(resnet.layer3, stride=2, dilation=2)
            self.layer4 = self._make_dilated_layer(resnet.layer4, stride=1, dilation=4)
        elif output_stride == 8:
            self.layer3 = self._make_dilated_layer(resnet.layer3, stride=1, dilation=2)
            self.layer4 = self._make_dilated_layer(resnet.layer4, stride=1, dilation=4)
        else:
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4

    def _make_dilated_layer(self, layer, stride, dilation):
        """修改层为空洞卷积"""
        # 修改第一个block的步幅
        if hasattr(layer[0], 'conv2'):
            layer[0].conv2.stride = (stride, stride)
            layer[0].conv2.dilation = (dilation, dilation)
            layer[0].conv2.padding = (dilation, dilation)
        elif hasattr(layer[0], 'conv1'):
            layer[0].conv1.stride = (stride, stride)
            layer[0].conv1.dilation = (dilation, dilation)
            layer[0].conv1.padding = (dilation, dilation)

        return layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


if __name__ == "__main__":
    x = torch.randn(2, 3, 512, 512)

    # 测试ResNetBackbone
    backbone = ResNetBackbone(3, 'resnet50')
    features = backbone(x)
    for k, v in features.items():
        print(f"{k}: {v.shape}")

    # 测试ResNetDilatedBackbone
    dilated = ResNetDilatedBackbone(3, 'resnet50', output_stride=16)
    out = dilated(x)
    print(f"Dilated输出: {out.shape}")
