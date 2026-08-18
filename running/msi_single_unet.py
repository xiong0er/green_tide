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
        self.topk_rate = topk_rate  # 保存 topk_rate 用于打印
        window_size_sq = window_size * window_size
        self.topk = max(1, int(window_size_sq * topk_rate))
        self.scale = (dim // num_heads) ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, h, w):
        b, l, c = x.shape
        ws = self.window_size
        x = self.norm(x)
        qkv = self.qkv(x).reshape(b, l, 3, self.num_heads, c // self.num_heads)
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
        out = self.proj(out)
        return out


class MSIEncoder(nn.Module):
    """MSI 编码器：每层 = 特征提取 → 下采样"""
    def __init__(self, in_channels, hidden_dims=[256, 512, 768, 1024], num_mamba=2, num_topk=2, mamba_dim=64, topk_rate=0.4, use_pixel_unshuffle=True):
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
    """带 GroupNorm 的残差块"""
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
        out = out + residual
        out = self.act(out)
        return out


class SegmentationHead(nn.Module):
    """纯分割输出头"""
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


class MSISingleUNet(nn.Module):
    """
    MSI 单支路 U-Net（纯分割版本）
    - 移除 SFT 模块
    - 移除重构分支
    - 只输出分割图
    """
    def __init__(self,
                 msi_in_channels=11,
                 hidden_dim=1024,
                 bottleneck_size=32,
                 num_mamba_layers=2,
                 num_topk_layers=2,
                 eca_k_size=3,
                 eca_topk_msi=6,
                 mamba_dim=64,
                 topk_rate=0.4,
                 use_pixel_unshuffle=True):
        super().__init__()

        self.bottleneck_size = bottleneck_size
        self.hidden_dim = hidden_dim
        self.msi_in_channels = msi_in_channels
        self.eca_topk_msi = eca_topk_msi
        self.topk_rate = topk_rate

        # ECA 输入过滤
        self.msi_eca = ECALayer(msi_in_channels, k_size=eca_k_size, topk=eca_topk_msi, noise_std_init=0.1)

        encoder_dims = [128, 256, 512, hidden_dim]

        # MSI 编码器
        self.msi_encoder = MSIEncoder(
            msi_in_channels, encoder_dims,
            num_mamba=num_mamba_layers, num_topk=num_topk_layers,
            mamba_dim=mamba_dim, topk_rate=topk_rate,
            use_pixel_unshuffle=use_pixel_unshuffle
        )

        # 解码器
        self.num_upsample = len(encoder_dims)
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

        # 纯分割输出头（无 SFT）
        self.seg_head = SegmentationHead(
            in_channels=decoder_dims[-1],
            seg_channels=1
        )

    def set_epoch(self, epoch, total_epochs):
        self.msi_eca.set_epoch(epoch, total_epochs)

    def forward(self, msi_input):
        # ECA 输入过滤
        msi_filtered = self.msi_eca(msi_input)
        msi_topk_indices = self.msi_eca.last_topk_indices

        # 将选中的 K 个通道放回原始位置
        b = msi_input.size(0)
        h, w = msi_input.shape[2:]
        msi_masked = torch.zeros_like(msi_input)
        msi_idx_exp = msi_topk_indices.unsqueeze(-1).unsqueeze(-1).expand(b, self.eca_topk_msi, h, w)
        msi_masked.scatter_(1, msi_idx_exp, msi_filtered)

        # 编码器
        msi_feat, msi_skip = self.msi_encoder(msi_masked)

        # 解码器
        x = msi_feat
        for i in range(len(self.pixel_shuffle_layers)):
            x = self.pixel_shuffle_layers[i](x)
            if i < len(self.decoder_convs):
                skip = msi_skip[-(i+1)]
                if x.shape[2:] != skip.shape[2:]:
                    skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
                skip = self.skip_align[i](skip)
                x = torch.cat([x, skip], dim=1)
                x = self.decoder_convs[i](x)

        # 分割输出
        seg = self.seg_head(x)

        return {
            'msi_seg': seg,                    # [B, 1, H, W]
            'msi_topk_indices': msi_topk_indices,  # [B, eca_topk_msi]
        }


def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MSISingleUNet(
        msi_in_channels=11,
        hidden_dim=1024,
        eca_topk_msi=6,
    ).to(device)

    msi_input = torch.randn(2, 12, 512, 512).to(device)
    with torch.no_grad():
        outputs = model(msi_input)

    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
    return model, outputs


if __name__ == "__main__":
    model, outputs = test_model()
