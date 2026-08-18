"""
DualBranchSegUNet: 双分支纯分割模型（无融合 + 共享解码器）

设计动机：
- 去掉 PixelGate/任何融合层——融合的逐像素竞争会偏向高质量传感器 (S2)，压制 L8。
- 两个传感器各自专属编码器（MSI: PixelUnshuffle, OLI: StridedConv），bottleneck 均为 hidden_dim。
- 每个传感器独立走一遍 **共享解码器**（权重共享，但各用自己的 skip），互不干扰。
- shared_seg_head=True: 两传感器共用分割头（等同 RoutedSingleUNet）。
  shared_seg_head=False: S2/L8 各用独立分割头。

输出接口与 train_dual_branch_pure_seg.py 对齐：
  {'head1_msi_seg', 'head2_oli_seg', 'msi_topk_indices', 'oli_topk_indices'}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.eca import ECALayer
from running.unet_model import MSIBranchEncoder, OLIBranchEncoder, SimpleResBlock


class AlgaeClassifier(nn.Module):
    """二分类辅助头：预测 L8 tile 是否含藻类（解决空标签混淆）"""
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2)
        )

    def forward(self, bottleneck_feat):
        return self.classifier(bottleneck_feat)  # [B, 2]


class SegHead(nn.Module):
    """纯分割输出头：2×SimpleResBlock → Conv → Sigmoid"""
    def __init__(self, in_channels, seg_channels=1):
        super().__init__()
        self.branch = nn.Sequential(
            SimpleResBlock(in_channels),
            SimpleResBlock(in_channels),
            nn.Conv2d(in_channels, seg_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.branch(x)


class DualBranchSegUNet(nn.Module):
    def __init__(self,
                 msi_in_channels=11,
                 oli_in_channels=7,
                 hidden_dim=1024,
                 num_mamba_layers=2,
                 num_topk_layers=2,
                 eca_k_size=3,
                 eca_topk_msi=6,
                 eca_topk_oli=4,
                 mamba_dim=64,
                 topk_rate=0.4,
                 shared_seg_head=True,
                 use_algae_classifier=False,
                 drop_l8_shallow_skips=0,
                 cross_scan_layers=0,
                 horizontal_scan_layers_msi=0,
                 horizontal_scan_layers_oli=0,
                 transformer_layers_msi=0,
                 transformer_layers_oli=0,
                 deep_topk_rate=None,
                 eca_mode='topk', disable_eca=False):
        super().__init__()
        self.msi_in_channels = msi_in_channels
        self.oli_in_channels = oli_in_channels
        self.eca_topk_msi = eca_topk_msi
        self.eca_topk_oli = eca_topk_oli
        self.shared_seg_head = shared_seg_head
        self.use_algae_classifier = use_algae_classifier
        self.drop_l8_shallow_skips = drop_l8_shallow_skips
        self.eca_mode = eca_mode
        self.disable_eca = disable_eca

        # ECA 输入过滤
        if not disable_eca:
            self.msi_eca = ECALayer(msi_in_channels, k_size=eca_k_size, topk=eca_topk_msi,
                                    noise_std_init=0.1, mode=eca_mode)
            self.oli_eca = ECALayer(oli_in_channels, k_size=eca_k_size, topk=eca_topk_oli,
                                    noise_std_init=0.1, mode=eca_mode)
        else:
            self.msi_eca = None
            self.oli_eca = None

        # 两支专属编码器，bottleneck 均为 hidden_dim（支持不对称特征提取器）
        encoder_dims = [128, 256, 512, hidden_dim]
        self.msi_encoder = MSIBranchEncoder(
            msi_in_channels, encoder_dims,
            num_mamba=num_mamba_layers, num_topk=num_topk_layers,
            mamba_dim=mamba_dim, topk_rate=topk_rate,
            cross_scan_layers=cross_scan_layers,
            horizontal_scan_layers=horizontal_scan_layers_msi,
            transformer_layers=transformer_layers_msi,
            deep_topk_rate=deep_topk_rate
        )
        self.oli_encoder = OLIBranchEncoder(
            oli_in_channels, encoder_dims,
            num_mamba=num_mamba_layers, num_topk=num_topk_layers,
            mamba_dim=mamba_dim, topk_rate=topk_rate,
            cross_scan_layers=cross_scan_layers,
            horizontal_scan_layers=horizontal_scan_layers_oli,
            transformer_layers=transformer_layers_oli,
            deep_topk_rate=deep_topk_rate
        )

        # ========== 共享解码器（无融合）==========
        decoder_dims = encoder_dims[::-1]                   # [hidden_dim, 512, 256, 128]
        skip_channels = [mamba_dim] + encoder_dims[:-1]     # [mamba_dim, 128, 256, 512]

        # 两支编码器用相同 encoder_dims，skip 通道一致 → 共享一套 align（解码器的一部分）
        self.skip_align = nn.ModuleList()
        for i in range(len(decoder_dims) - 1):
            self.skip_align.append(
                nn.Conv2d(skip_channels[-(i + 1)], decoder_dims[i], 1)
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

        # ========== 分割头 ==========
        self.seg_head_msi = SegHead(decoder_dims[-1])
        if shared_seg_head:
            self.seg_head_oli = self.seg_head_msi
        else:
            self.seg_head_oli = SegHead(decoder_dims[-1])

        # ========== 二分类辅助头（预测 L8 是否有藻类）==========
        if use_algae_classifier:
            self.algae_cls = AlgaeClassifier(hidden_dim)
        else:
            self.algae_cls = None

    def set_epoch(self, epoch, total_epochs):
        if self.msi_eca is not None:
            self.msi_eca.set_epoch(epoch, total_epochs)
        if self.oli_eca is not None:
            self.oli_eca.set_epoch(epoch, total_epochs)

    def _eca_filter(self, x, eca, topk, in_channels):
        """ECA 波段选择/加权。返回 (output, indices)。"""
        b = x.size(0)
        h, w = x.shape[2:]
        if eca is None:
            # Full bands mode: pass through, use all channels
            idx = torch.arange(in_channels, device=x.device).unsqueeze(0).expand(b, -1)
            return x, idx
        filtered = eca(x)  # [B, K, H, W] (topk/random) or [B, C, H, W] (soft)
        idx = eca.last_topk_indices  # [B, K] or [B, in_channels] for soft
        if eca.mode == 'soft':
            # Soft mode returns all channels weighted, no scatter needed
            return filtered, idx
        # TopK / random_fixed: scatter back to original positions
        k = filtered.size(1)
        masked = torch.zeros_like(x)
        idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(b, k, h, w)
        masked.scatter_(1, idx_exp, filtered)
        return masked, idx

    def _decode(self, feat, skips, drop_n=0):
        """共享解码器：单传感器特征 + 自己的 skip。

        drop_n > 0: 丢弃前 drop_n 个最高分辨率的浅层 skip（对应 L8 低 SNR 优化）。
                    弃掉的 skip 用零填充替代，保持通道维度兼容。
        """
        x = feat
        for i in range(len(self.pixel_shuffle_layers)):
            x = self.pixel_shuffle_layers[i](x)
            if i < len(self.decoder_convs):
                if i < drop_n:
                    # 丢弃浅层 skip，用零替代
                    x = torch.cat([x, torch.zeros_like(x)], dim=1)
                else:
                    s = skips[-(i + 1)]
                    if x.shape[2:] != s.shape[2:]:
                        s = F.interpolate(s, size=x.shape[2:], mode='bilinear', align_corners=False)
                    s = self.skip_align[i](s)
                    x = torch.cat([x, s], dim=1)
                x = self.decoder_convs[i](x)
        return x

    def forward(self, msi_input, oli_input):
        # ECA 过滤
        msi_masked, msi_idx = self._eca_filter(msi_input, self.msi_eca, self.eca_topk_msi, self.msi_in_channels)
        oli_masked, oli_idx = self._eca_filter(oli_input, self.oli_eca, self.eca_topk_oli, self.oli_in_channels)

        # 各自编码
        msi_feat, msi_skip = self.msi_encoder(msi_masked)
        oli_feat, oli_skip = self.oli_encoder(oli_masked)

        # 各自走共享解码器（无融合；L8 可选丢弃浅层 skip）
        x_msi = self._decode(msi_feat, msi_skip, drop_n=0)
        x_oli = self._decode(oli_feat, oli_skip, drop_n=self.drop_l8_shallow_skips)

        # 分割头
        msi_seg = self.seg_head_msi(x_msi)
        oli_seg = self.seg_head_oli(x_oli)

        # 二分类辅助头：预测 L8 是否有藻类；高置信无藻时用掩码置零 seg
        if self.algae_cls is not None:
            logits = self.algae_cls(oli_feat)
            probs = F.softmax(logits, dim=1)
            # 置信度 > 0.95 判断为无藻类 → mask 置零（保持梯度通路）
            confident_no_algae = (logits.argmax(dim=1) == 0) & (probs[:, 0] > 0.95)
            keep_mask = (~confident_no_algae).float().view(-1, 1, 1, 1)
            oli_seg = oli_seg * keep_mask

        result = {
            'head1_msi_seg': msi_seg,
            'head2_oli_seg': oli_seg,
            'msi_topk_indices': msi_idx,
            'oli_topk_indices': oli_idx,
        }
        if self.algae_cls is not None:
            result['oli_algae_logits'] = logits
        return result
