"""
Swin Transformer + UperNet模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class Mlp(nn.Module):
    """MLP"""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """窗口注意力"""
    def __init__(self, dim, window_size=7, num_heads=8, qkv_bias=True,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer块"""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias,
                                   attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, drop=drop)

    def forward(self, x, H, W):
        B, N, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 窗口划分
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # 分窗口
        x = rearrange(x, 'b (h ws1) (w ws2) c -> b h w ws1 ws2 c',
                     ws1=self.window_size, ws2=self.window_size)
        x = x.reshape(-1, self.window_size * self.window_size, C)

        # 窗口注意力
        x = self.attn(x)

        # 反窗口划分
        x = x.reshape(-1, Hp // self.window_size, Wp // self.window_size,
                     self.window_size, self.window_size, C)
        x = rearrange(x, 'b h w ws1 ws2 c -> b (h ws1) (w ws2) c')

        # 裁剪
        x = x[:, :H, :W, :].reshape(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class PatchMerging(nn.Module):
    """Patch合并（下采样）"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        # 填充
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """Swin Transformer层"""
    def __init__(self, dim, depth, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size

        # 构建块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop)
            for i in range(depth)])

        # 下采样层
        if downsample is not None:
            self.downsample = downsample(dim=dim)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class SwinTransformer(nn.Module):
    """Swin Transformer骨干"""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_layers = len(depths)

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)

        # 层
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)

    def forward(self, x):
        x = self.patch_embed(x)  # B, embed_dim, H//4, W//4
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')

        features = []
        for layer in self.layers:
            x_pre, H_old, W_old, x_out, Wh, Ww = layer(x, H, W)
            # 保留下采样前的特征作为 skip
            features.append(rearrange(x_pre, 'b (h w) c -> b c h w', h=H_old, w=W_old))
            # 更新：下一层的输入是下采样后的特征
            x, H, W = x_out, Wh, Ww

        return features


class SwinUperNet(nn.Module):
    """
    Swin Transformer + UperNet
    论文: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=512,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()

        # Swin Transformer骨干
        self.backbone = SwinTransformer(
            img_size=img_size, patch_size=4, in_chans=in_channels,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads)

        # UperNet特征融合
        self.fpn_in = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim * (2 ** i), 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for i in range(4)
        ])

        self.fpn_out = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])

        # PPM模块
        self.ppm_pooling = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(embed_dim * 8, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ) for scale in [1, 2, 3, 6]
        ])
        self.ppm_last_conv = nn.Sequential(
            nn.Conv2d(embed_dim * 8 + 256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 分类头
        self.conv_last = nn.Sequential(
            nn.Conv2d(256 * 4, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        input_size = x.size()[2:]

        # 骨干网络
        features = self.backbone(x)

        # FPN
        fpn_features = []
        for i, (feat, fpn_in) in enumerate(zip(features, self.fpn_in)):
            f = fpn_in(feat)
            if i > 0:
                f = f + F.interpolate(fpn_features[-1], size=f.size()[2:],
                                     mode='bilinear', align_corners=False)
            fpn_features.append(f)

        # PPM (使用最后一个特征)
        ppm_out = [features[-1]]
        for ppm in self.ppm_pooling:
            ppm_feat = ppm(features[-1])
            ppm_feat = F.interpolate(ppm_feat, size=features[-1].size()[2:],
                                    mode='bilinear', align_corners=False)
            ppm_out.append(ppm_feat)
        ppm_out = self.ppm_last_conv(torch.cat(ppm_out, dim=1))

        # 融合所有FPN特征
        fpn_outs = []
        for i, (feat, fpn_out) in enumerate(zip(fpn_features, self.fpn_out)):
            if i == len(fpn_features) - 1:
                f = ppm_out
            else:
                f = fpn_out(feat)
            fpn_outs.append(F.interpolate(f, size=input_size,
                                         mode='bilinear', align_corners=False))

        output = self.conv_last(torch.cat(fpn_outs, dim=1))

        return output


if __name__ == "__main__":
    x = torch.randn(2, 3, 512, 512)
    model = SwinUperNet(3, 2, 512)
    out = model(x)
    print(f"SwinUperNet输出: {out.shape}")
