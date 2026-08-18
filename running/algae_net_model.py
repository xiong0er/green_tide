import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        

    def forward(self, x):
        return self.conv(x)


class ResNet18_UNet(nn.Module):
    def __init__(self, in_channels=7, out_channels=1):
        super().__init__()
        
        self.first_convs = nn.ModuleDict({
            'c6': nn.Conv2d(6, 64, kernel_size=3, padding=1),
            'c7': nn.Conv2d(7, 64, kernel_size=3, padding=1),
            'c11': nn.Conv2d(11, 64, kernel_size=3, padding=1),
            'c12': nn.Conv2d(12, 64, kernel_size=3, padding=1),
        })
        self.enc1 = ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(256, 512)

        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec1 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = ConvBlock(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def _forward_impl(self, x):
        n_channels = x.shape[1]
        
        # 根据通道数选择正确的分支
        conv_key = f'c{n_channels}'
        if conv_key in self.first_convs:
            x = self.first_convs[conv_key](x)
        else:
            raise ValueError(f"不支持的通道数: {n_channels}. 请在 UNet_Attention 的 self.first_convs 中添加 'c{n_channels}'")
        
        # 拼接并调整通道数
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        d1 = self.up1(e4)
        d1 = torch.cat([d1, e3], dim=1)
        d1 = self.dec1(d1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d3 = self.up3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.dec3(d3)
        out = torch.sigmoid(self.final(d3))
        return out
    
    def forward(self, x):
        if isinstance(x, list):
            outputs = [self._forward_impl(img.unsqueeze(0)) for img in x]
            return torch.cat(outputs, dim=0)
        else:
            # 如果输入是单个张量 (用于测试或标准 dataloader)
            return self._forward_impl(x)
