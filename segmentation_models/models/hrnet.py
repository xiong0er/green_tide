"""
HRNet模型 - 高分辨率网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """HRNet基础块"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HRModule(nn.Module):
    """HRNet高分辨率模块"""
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method='SUM'):
        super().__init__()
        self.num_branches = num_branches
        self.fuse_method = fuse_method

        self.branches = self._make_branches(num_branches, blocks, num_blocks,
                                           num_channels, num_inchannels)
        self.fuse_layers = self._make_fuse_layers(num_branches, num_inchannels)
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        layers = []
        downsample = None
        inplanes = num_channels[branch_index] * block.expansion
        planes = num_channels[branch_index]

        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels, num_inchannels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self, num_branches, num_inchannels):
        fuse_layers = []
        for i in range(num_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_inchannels[i], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_inchannels[i])
                            ))
                        else:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_inchannels[j], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_inchannels[j]),
                                nn.ReLU(inplace=True)
                            ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(self.num_branches):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class HRNet(nn.Module):
    """
    HRNet模型 - 简化版
    论文: Deep High-Resolution Representation Learning for Visual Recognition
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 base_channels: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Stem
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # Stage 1
        self.stage1 = self._make_stage(64, base_channels, 4)

        # Stage 2 - 2个分支
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 2, 3, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU(inplace=True)
            )
        ])
        self.stage2 = HRModule(2, BasicBlock, [4, 4],
                              [base_channels, base_channels * 2],
                              [base_channels, base_channels * 2])

        # Stage 3 - 3个分支
        self.transition2 = nn.ModuleList([
            None,
            None,
            nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True)
            )
        ])
        self.stage3 = HRModule(3, BasicBlock, [4, 4, 4],
                              [base_channels, base_channels * 2, base_channels * 4],
                              [base_channels, base_channels * 2, base_channels * 4])

        # Stage 4 - 4个分支
        self.transition3 = nn.ModuleList([
            None, None, None,
            nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels * 8, 3, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels * 8),
                nn.ReLU(inplace=True)
            )
        ])
        self.stage4 = HRModule(4, BasicBlock, [4, 4, 4, 4],
                              [base_channels, base_channels * 2, base_channels * 4, base_channels * 8],
                              [base_channels, base_channels * 2, base_channels * 4, base_channels * 8])

        # 分类头
        self.classifier = nn.Sequential(
            nn.Conv2d(sum([base_channels, base_channels * 2, base_channels * 4, base_channels * 8]),
                     num_classes, 1)
        )

    def _make_stage(self, in_channels, out_channels, num_blocks):
        """创建Stage"""
        layers = []
        for i in range(num_blocks):
            layers.append(BasicBlock(in_channels if i == 0 else out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Stage 1
        x = self.stage1(x)

        # Stage 2
        x = [trans(x) for trans in self.transition1]
        x = self.stage2(x)

        # Stage 3
        x = [x[i] if trans is None else trans(x[i]) for i, trans in enumerate(self.transition2)]
        x = self.stage3(x)

        # Stage 4
        x = [x[i] if trans is None else trans(x[i]) for i, trans in enumerate(self.transition3)]
        x = self.stage4(x)

        # 融合所有分支
        size = x[0].size()[2:]
        x0 = x[0]
        x1 = F.interpolate(x[1], size=size, mode='bilinear', align_corners=False)
        x2 = F.interpolate(x[2], size=size, mode='bilinear', align_corners=False)
        x3 = F.interpolate(x[3], size=size, mode='bilinear', align_corners=False)

        x = torch.cat([x0, x1, x2, x3], dim=1)

        # 分类
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    x = torch.randn(2, 3, 512, 512)
    model = HRNet(3, 2, base_channels=32)
    out = model(x)
    print(f"HRNet输出: {out.shape}")
