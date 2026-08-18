"""
RoutedSingleUNet: 单支路路由模型
- 每个样本根据 sensor_type 路由到 MSI 或 OLI 编码器
- MSI (S2): 11ch → ECA → PixelUnshuffle encoder
- OLI (L8): 7ch → ECA → StridedConv encoder
- 共享 PixelShuffle 解码器 + 分割头
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.eca import ECALayer
from model.vertical_mamba import WindowedVisionMamba2Block
from model.pixel_unshuffle import FastPixelUnshuffle


class TopKAttentionBlock(nn.Module):
    """窗口 Top-K 注意力块"""
    def __init__(self, dim, num_heads=8, window_size=8, topk_rate=0.4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        window_size_sq = window_size * window_size
        self.topk = max(1, int(window_size_sq * topk_rate))
        self.scale = (dim // num_heads) ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, h, w):
        b, l, c = x.shape
        ws = self.window_size
        x_n = self.norm(x)
        qkv = self.qkv(x_n).reshape(b, l, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.reshape(b, self.num_heads, h // ws, ws, w // ws, ws, c // self.num_heads)
        q = q.permute(0, 1, 2, 4, 3, 5, 6)
        k = k.reshape(b, self.num_heads, h // ws, ws, w // ws, ws, c // self.num_heads)
        k = k.permute(0, 1, 2, 4, 3, 5, 6)
        v = v.reshape(b, self.num_heads, h // ws, ws, w // ws, ws, c // self.num_heads)

        hw, ww = h // ws, w // ws
        q = q.reshape(b, self.num_heads, hw, ww, ws * ws, c // self.num_heads)
        k = k.reshape(b, self.num_heads, hw, ww, ws * ws, c // self.num_heads)
        v = v.reshape(b, self.num_heads, hw, ww, ws * ws, c // self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.topk < ws * ws:
            topk_vals, topk_idx = torch.topk(attn, k=min(self.topk, ws * ws), dim=-1)
            mask = torch.full_like(attn, float('-inf'))
            mask.scatter_(-1, topk_idx, topk_vals)
            attn = mask
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.reshape(b, self.num_heads, hw, ws, ww, ws, c // self.num_heads)
        out = out.permute(0, 1, 2, 4, 3, 5, 6).reshape(b, self.num_heads, l, c // self.num_heads)
        out = out.transpose(1, 2).reshape(b, l, c)
        return self.proj(out)


class RoutedEncoder(nn.Module):
    """路由编码器：根据 use_pixel_unshuffle 选择下采样方式"""
    def __init__(self, in_channels, hidden_dims, num_mamba, num_topk, mamba_dim, topk_rate, use_pixel_unshuffle):
        super().__init__()
        self.stages = nn.ModuleList()
        self.feature_extractors = nn.ModuleList()

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, mamba_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, mamba_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        current_dim = mamba_dim
        for i, hidden_dim in enumerate(hidden_dims[:4]):
            if i < num_mamba:
                self.feature_extractors.append(
                    WindowedVisionMamba2Block(d_model=current_dim, window_size=8)
                )
            else:
                self.feature_extractors.append(
                    TopKAttentionBlock(dim=current_dim, topk_rate=topk_rate)
                )

            if use_pixel_unshuffle:
                self.stages.append(nn.Sequential(
                    FastPixelUnshuffle(2),
                    nn.Conv2d(current_dim * 4, hidden_dim, 3, padding=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
            else:
                self.stages.append(nn.Sequential(
                    nn.Conv2d(current_dim, hidden_dim, 3, stride=2, padding=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
            current_dim = hidden_dim

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.input_proj(x)
        features = []
        for feat_ext, down_stage in zip(self.feature_extractors, self.stages):
            seq = rearrange(x, 'b c h w -> b (h w) c')
            seq = feat_ext(seq, h, w)
            x = rearrange(seq, 'b (h w) c -> b c h w', h=h, w=w)
            features.append(x)
            x = down_stage(x)
            h, w = x.shape[2], x.shape[3]
        return x, features


class SimpleResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return self.act(out + residual)


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, seg_channels=1):
        super().__init__()
        self.seg_branch = nn.Sequential(
            SimpleResBlock(in_channels),
            SimpleResBlock(in_channels),
            nn.Conv2d(in_channels, seg_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.seg_branch(x)


class RoutedSingleUNet(nn.Module):
    """
    单支路路由 U-Net: 每个样本根据 sensor_type 选择编码器，共享解码器。
    - S2 (sensor_type='s2'): 11ch → MSI ECA → PixelUnshuffle encoder
    - L8 (sensor_type='l8'): 7ch → OLI ECA → StridedConv encoder
    - 共享 PixelShuffle decoder + SegmentationHead
    """
    def __init__(self,
                 msi_in_channels=11,
                 oli_in_channels=7,
                 hidden_dim=1024,
                 bottleneck_size=16,
                 num_mamba_layers=2,
                 num_topk_layers=2,
                 eca_k_size=3,
                 eca_topk_msi=6,
                 eca_topk_oli=4,
                 mamba_dim=64,
                 topk_rate=0.4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eca_topk_msi = eca_topk_msi
        self.eca_topk_oli = eca_topk_oli
        self.oli_in_channels = oli_in_channels

        # ECA 层
        self.msi_eca = ECALayer(msi_in_channels, k_size=eca_k_size, topk=eca_topk_msi, noise_std_init=0.1)
        self.oli_eca = ECALayer(oli_in_channels, k_size=eca_k_size, topk=eca_topk_oli, noise_std_init=0.1)

        encoder_dims = [128, 256, 512, hidden_dim]

        # MSI 编码器 (PixelUnshuffle)
        self.msi_encoder = RoutedEncoder(
            msi_in_channels, encoder_dims,
            num_mamba=num_mamba_layers, num_topk=num_topk_layers,
            mamba_dim=mamba_dim, topk_rate=topk_rate,
            use_pixel_unshuffle=True
        )

        # OLI 编码器 (StridedConv)
        self.oli_encoder = RoutedEncoder(
            oli_in_channels, encoder_dims,
            num_mamba=num_mamba_layers, num_topk=num_topk_layers,
            mamba_dim=mamba_dim, topk_rate=topk_rate,
            use_pixel_unshuffle=False
        )

        # 共享解码器
        decoder_dims = encoder_dims[::-1]
        self.skip_align = nn.ModuleList()
        skip_channels = [mamba_dim] + encoder_dims[:-1]
        for i in range(len(decoder_dims) - 1):
            self.skip_align.append(
                nn.Conv2d(skip_channels[-(i+1)], decoder_dims[i], 1)
            )

        self.decoder_convs = nn.ModuleList()
        for i in range(len(decoder_dims) - 1):
            self.decoder_convs.append(nn.Sequential(
                nn.Conv2d(decoder_dims[i] * 2, decoder_dims[i + 1], 3, padding=1),
                nn.GroupNorm(8, decoder_dims[i + 1]),
                nn.LeakyReLU(0.2, inplace=True)
            ))

        self.pixel_shuffle_layers = nn.ModuleList()
        for dim in decoder_dims:
            self.pixel_shuffle_layers.append(nn.Sequential(
                nn.Conv2d(dim, dim * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.GroupNorm(8, dim),
                nn.LeakyReLU(0.2, inplace=True)
            ))

        # 共享分割头
        self.seg_head = SegmentationHead(in_channels=decoder_dims[-1])

    def set_epoch(self, epoch, total_epochs):
        self.msi_eca.set_epoch(epoch, total_epochs)
        self.oli_eca.set_epoch(epoch, total_epochs)

    def _encode_msi(self, x):
        """S2 编码: 11ch → ECA → PixelUnshuffle encoder"""
        b, c, h, w = x.shape
        filtered = self.msi_eca(x)
        topk_idx = self.msi_eca.last_topk_indices
        masked = torch.zeros_like(x)
        idx_exp = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(b, self.eca_topk_msi, h, w)
        masked.scatter_(1, idx_exp, filtered)
        return self.msi_encoder(masked)

    def _encode_oli(self, x):
        """L8 编码: 7ch → ECA → StridedConv encoder"""
        b, c, h, w = x.shape
        # x 是 padded 的 11ch，取前 7ch
        x_7ch = x[:, :self.oli_in_channels]
        filtered = self.oli_eca(x_7ch)
        topk_idx = self.oli_eca.last_topk_indices
        masked = torch.zeros_like(x_7ch)
        idx_exp = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(b, self.eca_topk_oli, h, w)
        masked.scatter_(1, idx_exp, filtered)
        return self.oli_encoder(masked)

    def _decode(self, feat, skip_features):
        """共享解码器"""
        x = feat
        for i in range(len(self.pixel_shuffle_layers)):
            x = self.pixel_shuffle_layers[i](x)
            if i < len(self.decoder_convs):
                s = skip_features[-(i+1)]
                if x.shape[2:] != s.shape[2:]:
                    s = F.interpolate(s, size=x.shape[2:], mode='bilinear', align_corners=False)
                s = self.skip_align[i](s)
                x = torch.cat([x, s], dim=1)
                x = self.decoder_convs[i](x)
        return self.seg_head(x)

    def forward(self, x, sensor_types):
        """
        x: [B, 11, H, W] — L8 样本 padded 到 11ch
        sensor_types: list of 's2' or 'l8', length B
        """
        B = x.size(0)
        device = x.device
        H, W = x.shape[2], x.shape[3]

        s2_indices = [i for i, st in enumerate(sensor_types) if st == 's2']
        l8_indices = [i for i, st in enumerate(sensor_types) if st == 'l8']

        # 初始化输出
        seg_out = torch.zeros(B, 1, H, W, device=device)

        if s2_indices:
            s2_x = x[s2_indices]
            feat, skips = self._encode_msi(s2_x)
            seg_s2 = self._decode(feat, skips)
            seg_out[s2_indices] = seg_s2

        if l8_indices:
            l8_x = x[l8_indices]
            feat, skips = self._encode_oli(l8_x)
            seg_l8 = self._decode(feat, skips)
            seg_out[l8_indices] = seg_l8

        return {'msi_seg': seg_out}
