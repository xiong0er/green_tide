"""
SegFormer模型 - 基于Transformer的高效分割
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class PatchEmbed(nn.Module):
    """图像块嵌入"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # B, C, H//P, W//P
        return x


class Attention(nn.Module):
    """多头注意力"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
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

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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


class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                             attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity() if drop_path == 0 else nn.Identity()  # 简化版
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                      act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, embed_dim=256, num_heads=8, depth=4, mlp_ratio=4.):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: B, C, H, W -> B, H*W, C
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # B, H*W, C -> B, C, H, W
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class SegFormer(nn.Module):
    """
    SegFormer模型 - 简化版
    论文: SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=512,
                 embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                 depths=[2, 2, 6, 3], mlp_ratios=[4, 4, 4, 4]):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(embed_dims)

        # 分阶段处理
        self.patch_embeds = nn.ModuleList()
        self.transformer_encoders = nn.ModuleList()

        in_ch = in_channels
        for i in range(self.num_stages):
            # Patch embedding
            patch_embed = nn.Sequential(
                nn.Conv2d(in_ch if i == 0 else embed_dims[i-1],
                         embed_dims[i], 3, stride=2 if i > 0 else 4, padding=1),
                nn.BatchNorm2d(embed_dims[i]),
                nn.ReLU(inplace=True)
            )
            self.patch_embeds.append(patch_embed)

            # Transformer
            transformer = TransformerEncoder(
                embed_dim=embed_dims[i],
                num_heads=num_heads[i],
                depth=depths[i],
                mlp_ratio=mlp_ratios[i]
            )
            self.transformer_encoders.append(transformer)

        # MLP解码器
        self.decoder = SegFormerHead(embed_dims, num_classes)

    def forward(self, x):
        B = x.shape[0]
        features = []

        for i in range(self.num_stages):
            x = self.patch_embeds[i](x)
            x = self.transformer_encoders[i](x)
            features.append(x)

        # 解码
        x = self.decoder(features)

        return x


class SegFormerHead(nn.Module):
    """SegFormer MLP解码器"""
    def __init__(self, embed_dims, num_classes, dropout=0.1):
        super().__init__()
        c1_in, c2_in, c3_in, c4_in = embed_dims

        # 每个阶段的MLP
        self.linear_c4 = nn.Sequential(
            nn.Conv2d(c4_in, 256, 1),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)
        )
        self.linear_c3 = nn.Sequential(
            nn.Conv2d(c3_in, 256, 1),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        )
        self.linear_c2 = nn.Sequential(
            nn.Conv2d(c2_in, 256, 1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        )
        self.linear_c1 = nn.Sequential(
            nn.Conv2d(c1_in, 256, 1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )

        # 融合和预测
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(256 * 4, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

        self.linear_pred = nn.Conv2d(256, num_classes, 1)

    def forward(self, features):
        c1, c2, c3, c4 = features

        _c4 = self.linear_c4(c4)
        _c3 = self.linear_c3(c3)
        _c2 = self.linear_c2(c2)
        _c1 = self.linear_c1(c1)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.linear_pred(_c)

        return x


if __name__ == "__main__":
    x = torch.randn(2, 3, 512, 512)
    model = SegFormer(3, 2, 512)
    out = model(x)
    print(f"SegFormer输出: {out.shape}")
