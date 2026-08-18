import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 导入已有模块
from model.eca import ECANet, ECALayer
from model.vertical_mamba import WindowedVisionMamba2Block, CrossScanVisionMamba2Block, HorizontalScanVisionMamba2Block, WindowTransformerBlock
from model.dyt import DyT
from model.sft import SpatialFeatureTransform
from model.pixel_unshuffle import FastPixelUnshuffle


class TopKAttentionBlock(nn.Module):
    """窗口Top-K注意力块，使用LayerNorm

    topk计算方式: window_size**2 * topk_rate
    例如: window_size=8, topk_rate=0.4 -> topk = 64 * 0.4 = 25.6 ≈ 26
    """
    def __init__(self, dim, num_heads=8, window_size=8, topk_rate=0.4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.topk_rate = topk_rate

        # 根据 window_size 和 topk_rate 计算 topk
        window_size_sq = window_size * window_size
        self.topk = max(1, int(window_size_sq * topk_rate))

        self.scale = (dim // num_heads) ** -0.5

        # 使用LayerNorm代替DyT
        self.norm = nn.LayerNorm(dim)

        # QKV投影
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, h, w):
        """
        x: [B, L, C] where L = H * W
        """
        b, l, c = x.shape
        ws = self.window_size

        # LayerNorm归一化
        x = self.norm(x)

        # 生成QKV
        qkv = self.qkv(x).reshape(b, l, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, L, dim_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 重塑为窗口: [B, heads, L, dim] -> [B, heads, h//ws, w//ws, ws, ws, dim]
        assert h * w == l, f"Shape mismatch: h={h}, w={w}, l={l}"
        q = q.reshape(b, self.num_heads, h // ws, ws, w // ws, ws, c // self.num_heads)
        q = q.permute(0, 1, 2, 4, 3, 5, 6)  # [B, heads, h//ws, w//ws, ws, ws, dim]

        k = k.reshape(b, self.num_heads, h // ws, ws, w // ws, ws, c // self.num_heads)
        k = k.permute(0, 1, 2, 4, 3, 5, 6)

        v = v.reshape(b, self.num_heads, h // ws, ws, w // ws, ws, c // self.num_heads)
        v = v.permute(0, 1, 2, 4, 3, 5, 6)

        # 窗口内注意力: [B, heads, h//ws, w//ws, ws*ws, ws*ws]
        hw, ww = h // ws, w // ws
        q = q.reshape(b, self.num_heads, hw, ww, ws * ws, c // self.num_heads)
        k = k.reshape(b, self.num_heads, hw, ww, ws * ws, c // self.num_heads)
        v = v.reshape(b, self.num_heads, hw, ww, ws * ws, c // self.num_heads)

        # 计算窗口内注意力 [B, heads, hw, ww, ws^2, ws^2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 64x64 per window

        # 窗口内Top-K (topk = window_size**2 * topk_rate)
        if self.topk < ws * ws:
            topk_vals, topk_idx = torch.topk(attn, k=min(self.topk, ws * ws), dim=-1)
            mask = torch.full_like(attn, float('-inf'))
            mask.scatter_(-1, topk_idx, topk_vals)
            attn = mask

        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # [B, heads, hw, ww, ws^2, dim_head]

        # 重塑回原始形状
        out = out.reshape(b, self.num_heads, hw, ws, ww, ws, c // self.num_heads)
        out = out.permute(0, 1, 2, 4, 3, 5, 6).reshape(b, self.num_heads, l, c // self.num_heads)
        out = out.transpose(1, 2).reshape(b, l, c)
        out = self.proj(out)

        return out


class MSIBranchEncoder(nn.Module):
    """MSI支路编码器：每层 = 特征提取 → Pixel Unshuffle下采样"""
    def __init__(self, in_channels, hidden_dims=[256, 512, 768, 1024], num_mamba=2,
                 num_topk=2, mamba_dim=64, topk_rate=0.4, cross_scan_layers=0,
                 transformer_layers=0, horizontal_scan_layers=0, deep_topk_rate=None):
        super().__init__()
        self.stages = nn.ModuleList()
        self.feature_extractors = nn.ModuleList()

        # 投影层：3x3卷积替代1x1，增加空间感知
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, mamba_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, mamba_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        current_dim = mamba_dim  # 使用投影后的维度

        for i, hidden_dim in enumerate(hidden_dims[:4]):  # 只取前 4 层
            # 特征提取层
            if i < num_mamba:
                if i < transformer_layers:
                    # 前 transformer_layers 层使用窗口 Transformer（替代 WVM2B）
                    self.feature_extractors.append(
                        WindowTransformerBlock(d_model=current_dim, window_size=8)
                    )
                elif i < horizontal_scan_layers:
                    # 前 horizontal_scan_layers 层使用行扫描 Mamba（仅左右双向）
                    self.feature_extractors.append(
                        HorizontalScanVisionMamba2Block(d_model=current_dim, window_size=8, expand=1)
                    )
                elif i < cross_scan_layers:
                    # 前 cross_scan_layers 层使用交叉扫描 Mamba（列+行四方向）
                    self.feature_extractors.append(
                        CrossScanVisionMamba2Block(d_model=current_dim, window_size=8, expand=1)
                    )
                else:
                    # 其余 Mamba 层使用标准 WVM2B（列双向）
                    self.feature_extractors.append(
                        WindowedVisionMamba2Block(d_model=current_dim, window_size=8)
                    )
            else:
                # 后 num_topk 层使用 Top-K Attention
                rate = deep_topk_rate if deep_topk_rate is not None else topk_rate
                self.feature_extractors.append(
                    TopKAttentionBlock(dim=current_dim, topk_rate=rate)
                )

            # 下采样层：Pixel Unshuffle + 3x3 Conv + GroupNorm + LeakyReLU
            self.stages.append(nn.Sequential(
                FastPixelUnshuffle(2),
                nn.Conv2d(current_dim * 4, hidden_dim, 3, padding=1),  # 3x3替代1x1
                nn.GroupNorm(8, hidden_dim),  # 新增GroupNorm
                nn.LeakyReLU(0.2, inplace=True)  # 新增LeakyReLU
            ))
            current_dim = hidden_dim

    def forward(self, x):
        """返回所有阶段的特征用于skip connection"""
        b, c, h, w = x.shape

        # 投影到Mamba兼容的维度
        x = self.input_proj(x)

        features = []

        for feat_ext, down_stage in zip(self.feature_extractors, self.stages):
            # 1. 特征提取 (序列格式)
            seq = rearrange(x, 'b c h w -> b (h w) c')
            seq = feat_ext(seq, h, w)
            x = rearrange(seq, 'b (h w) c -> b c h w', h=h, w=w)
            features.append(x)

            # 2. 下采样
            x = down_stage(x)
            # 使用实际输出的空间尺寸
            h, w = x.shape[2], x.shape[3]

        return x, features


class OLIBranchEncoder(nn.Module):
    """OLI支路编码器：每层 = 特征提取 → Strided Conv下采样"""
    def __init__(self, in_channels, hidden_dims=[256, 512, 768, 1024], num_mamba=2,
                 num_topk=2, mamba_dim=64, topk_rate=0.4, cross_scan_layers=0,
                 transformer_layers=0, horizontal_scan_layers=0, deep_topk_rate=None):
        super().__init__()
        self.stages = nn.ModuleList()
        self.feature_extractors = nn.ModuleList()

        # 投影层：3x3卷积替代1x1，增加空间感知
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, mamba_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, mamba_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        current_dim = mamba_dim  # 使用投影后的维度

        for i, hidden_dim in enumerate(hidden_dims[:4]):  # 只取前 4 层
            # 特征提取层
            if i < num_mamba:
                if i < transformer_layers:
                    self.feature_extractors.append(
                        WindowTransformerBlock(d_model=current_dim, window_size=8)
                    )
                elif i < horizontal_scan_layers:
                    self.feature_extractors.append(
                        HorizontalScanVisionMamba2Block(d_model=current_dim, window_size=8, expand=1)
                    )
                elif i < cross_scan_layers:
                    self.feature_extractors.append(
                        CrossScanVisionMamba2Block(d_model=current_dim, window_size=8, expand=1)
                    )
                else:
                    self.feature_extractors.append(
                        WindowedVisionMamba2Block(d_model=current_dim, window_size=8)
                    )
            else:
                rate = deep_topk_rate if deep_topk_rate is not None else topk_rate
                self.feature_extractors.append(
                    TopKAttentionBlock(dim=current_dim, topk_rate=rate)
                )

            # 下采样层：3x3 Strided Conv + GroupNorm + LeakyReLU
            self.stages.append(nn.Sequential(
                nn.Conv2d(current_dim, hidden_dim, 3, stride=2, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            current_dim = hidden_dim

    def forward(self, x):
        """返回所有阶段的特征用于skip connection"""
        b, c, h, w = x.shape

        # 投影到Mamba兼容的维度
        x = self.input_proj(x)

        features = []

        for feat_ext, down_stage in zip(self.feature_extractors, self.stages):
            # 1. 特征提取 (序列格式)
            seq = rearrange(x, 'b c h w -> b (h w) c')
            seq = feat_ext(seq, h, w)
            x = rearrange(seq, 'b (h w) c -> b c h w', h=h, w=w)
            features.append(x)

            # 2. 下采样
            x = down_stage(x)
            # 使用实际输出的空间尺寸，而不是简单的整数除法
            h, w = x.shape[2], x.shape[3]

        return x, features


class ConcatFusion(nn.Module):
    """拼接融合：concat + 1x1 conv（对称，无门控竞争）

    替代 PixelGateFusion，避免逐像素 softmax 门控偏向高质量传感器 (S2)。
    两支特征拼接后由卷积学习融合，解码器可同时利用两个传感器的完整特征。
    """
    def __init__(self, dim):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GroupNorm(8, dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, feat1, feat2):
        """feat1, feat2: [B, C, H, W] -> [B, C, H, W]"""
        return self.fuse(torch.cat([feat1, feat2], dim=1))


class SimpleResBlock(nn.Module):
    """带GroupNorm的标准残差块，内部使用LeakyReLU避免梯度消失"""
    def __init__(self, dim):
        super().__init__()
        # 加入GroupNorm (针对Batch Size = 1的最佳实践)
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, dim)  # 将通道分为8组归一化
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

class OutputHead(nn.Module):
    """输出头：包含重构图和分割图分支"""
    def __init__(self, in_channels, recon_channels, seg_channels=1):
        super().__init__()

        # 重构图分支：残差块 -> 最终卷积 -> Sigmoid (将其限制在[0, 1])
        self.recon_branch = nn.Sequential(
            SimpleResBlock(in_channels),
            SimpleResBlock(in_channels),
            nn.Conv2d(in_channels, recon_channels, 3, padding=1),
            nn.Sigmoid()
        )

        # 分割图分支：残差块 -> 最终卷积 -> Sigmoid (将其限制在[0, 1])
        self.seg_branch = nn.Sequential(
            SimpleResBlock(in_channels),
            SimpleResBlock(in_channels),
            nn.Conv2d(in_channels, seg_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        recon = self.recon_branch(x)
        seg = self.seg_branch(x)
        return recon, seg


class DualBranchUNet(nn.Module):
    """
    双分支U-Net模型（ECA物理截断版本）
    - ECA输出 [B, K, H, W]，物理截断，非mask
    - 重建输出K个通道，loss只计算选中的K个通道
    """
    def __init__(self,
                 msi_in_channels=12,      # MSI输入通道 (默认12, 训练时用11)
                 oli_in_channels=7,       # OLI输入通道
                 hidden_dim=1024,
                 bottleneck_size=32,
                 num_mamba_layers=2,
                 num_topk_layers=2,
                 eca_k_size=3,
                 eca_topk_msi=3,          # MSI选择的Top-K通道数
                 eca_topk_oli=3,          # OLI选择的Top-K通道数
                 mamba_dim=64,
                 topk_rate=0.4,
                 use_sft=True,
                 single_branch=None):
        super().__init__()
        self.topk_rate = topk_rate
        self.use_sft = use_sft
        self.single_branch = single_branch  # None, 'msi', or 'oli'

        self.bottleneck_size = bottleneck_size
        self.hidden_dim = hidden_dim
        self.msi_in_channels = msi_in_channels
        self.oli_in_channels = oli_in_channels
        self.eca_topk_msi = eca_topk_msi
        self.eca_topk_oli = eca_topk_oli

        # ========== 输入ECA过滤（物理截断）==========
        self.msi_eca = ECALayer(msi_in_channels, k_size=eca_k_size, topk=eca_topk_msi, noise_std_init=0.1)
        self.oli_eca = ECALayer(oli_in_channels, k_size=eca_k_size, topk=eca_topk_oli, noise_std_init=0.1)

        # 编码器各层维度（4层下采样）
        encoder_dims = [128, 256, 512, hidden_dim]

        # 编码器：输入使用原始通道数，确保物理波段对应关系
        self.msi_encoder = MSIBranchEncoder(
            msi_in_channels, encoder_dims,
            num_mamba=num_mamba_layers, num_topk=num_topk_layers,
            mamba_dim=mamba_dim, topk_rate=topk_rate
        )
        self.oli_encoder = OLIBranchEncoder(
            oli_in_channels, encoder_dims,
            num_mamba=num_mamba_layers, num_topk=num_topk_layers,
            mamba_dim=mamba_dim, topk_rate=topk_rate
        )
        # ========== 拼接融合（无门控，避免偏向 S2）==========
        self.gate_fusion = ConcatFusion(hidden_dim)

        # ========== 解码器（Pixel Shuffle上采样） ==========
        # 计算上采样层数
        self.num_upsample = len(encoder_dims)

        # 解码器层（无Norm）
        decoder_dims = encoder_dims[::-1]

        # Skip connection对齐层（处理通道不匹配）
        self.skip_align_msi = nn.ModuleList()
        self.skip_align_oli = nn.ModuleList()
        self.skip_fusion = nn.ModuleList()  # 每层MSI+OLI skip融合模块
        skip_channels = [mamba_dim] + encoder_dims[:-1]  # [mamba_dim, 128, 256, 512] (4层编码器)
        for i in range(len(decoder_dims) - 1):  # 3 layers
            # 分别对齐MSI和OLI skip特征
            self.skip_align_msi.append(
                nn.Conv2d(skip_channels[-(i+1)], decoder_dims[i], 1)
            )
            self.skip_align_oli.append(
                nn.Conv2d(skip_channels[-(i+1)], decoder_dims[i], 1)
            )
            # 融合模块：将MSI和OLI skip特征拼接融合（无门控）
            self.skip_fusion.append(
                ConcatFusion(decoder_dims[i])
            )

        # 解码器卷积层（输入是cat后的通道，输出是下一层维度）
        self.decoder_convs = nn.ModuleList()
        for i in range(len(decoder_dims) - 1):  # 3 layers
            self.decoder_convs.append(nn.Sequential(
                nn.Conv2d(decoder_dims[i] * 2, decoder_dims[i + 1], 3, padding=1),
                nn.GroupNorm(8, decoder_dims[i + 1]),  # 新增：归一化层
                nn.LeakyReLU(0.2, inplace=True)
            ))

        # Pixel Shuffle上采样层（注入 GroupNorm）
        # 现在有4层上采样：16×16→32→64→128→256
        self.pixel_shuffle_layers = nn.ModuleList()
        for dim in decoder_dims:  # 4 layers for 4x upsampling
            self.pixel_shuffle_layers.append(nn.Sequential(
                nn.Conv2d(dim, dim * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.GroupNorm(8, dim),  # 新增：归一化层 (PixelShuffle后通道数恢复为dim)
                nn.LeakyReLU(0.2, inplace=True)
            ))

        # ========== SFT模块与粗分割头 ==========
        # 专门生成粗分割图的轻量级网络，防止和主输出头抢权重
        self.coarse_seg_head1 = nn.Sequential(
            nn.Conv2d(decoder_dims[-1], 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        self.coarse_seg_head2 = nn.Sequential(
            nn.Conv2d(decoder_dims[-1], 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

        self.sft_head1 = SpatialFeatureTransform(decoder_dims[-1], cond_channels=1)
        self.sft_head2 = SpatialFeatureTransform(decoder_dims[-1], cond_channels=1)

        # ========== 输出头 ==========
        # 输出头1：MSI重构图（输出eca_topk_msi通道，即选中的通道）, MSI分割图
        # 输出头 1：MSI 重构图（只输出 ECA 选择的 topk 通道）+ 分割图
        self.head1 = OutputHead(
            in_channels=decoder_dims[-1],
            recon_channels=eca_topk_msi,  # 只输出选中的 K 个通道
            seg_channels=1
        )
        # 输出头 2：OLI 重构图（只输出 ECA 选择的 topk 通道）+ 分割图
        self.head2 = OutputHead(
            in_channels=decoder_dims[-1],
            recon_channels=eca_topk_oli,  # 只输出选中的 K 个通道
            seg_channels=1
        )

    def set_epoch(self, epoch, total_epochs):
        """
        设置当前epoch，更新ECA层的噪声退火
        在训练循环的每个epoch开始时调用
        """
        self.msi_eca.set_epoch(epoch, total_epochs)
        self.oli_eca.set_epoch(epoch, total_epochs)

    def forward(self, msi_input, oli_input):

        b = msi_input.size(0)

        # ========== 单支路模式：仅处理对应传感器 ==========
        if self.single_branch == 'msi':
            msi_filtered = self.msi_eca(msi_input)
            msi_topk_indices = self.msi_eca.last_topk_indices
            msi_masked = torch.zeros_like(msi_input)
            h, w = msi_input.shape[2:]
            msi_idx_exp = msi_topk_indices.unsqueeze(-1).unsqueeze(-1).expand(b, self.eca_topk_msi, h, w)
            msi_masked.scatter_(1, msi_idx_exp, msi_filtered)

            bottleneck, skip = self.msi_encoder(msi_masked)
            x = bottleneck

            for i in range(len(self.pixel_shuffle_layers)):
                x = self.pixel_shuffle_layers[i](x)
                if i < len(self.decoder_convs):
                    s = skip[-(i+1)]  # skip[-1], skip[-2], skip[-3]
                    if x.shape[2:] != s.shape[2:]:
                        s = F.interpolate(s, size=x.shape[2:], mode='bilinear', align_corners=False)
                    s = self.skip_align_msi[i](s)
                    x = torch.cat([x, s], dim=1)
                    x = self.decoder_convs[i](x)

            recon, seg = self.head1(x)
            return {
                'head1_msi_recon': recon,
                'head1_msi_seg': seg,
                'msi_topk_indices': msi_topk_indices,
            }

        elif self.single_branch == 'oli':
            oli_filtered = self.oli_eca(oli_input)
            oli_topk_indices = self.oli_eca.last_topk_indices
            oli_masked = torch.zeros_like(oli_input)
            h, w = oli_input.shape[2:]
            oli_idx_exp = oli_topk_indices.unsqueeze(-1).unsqueeze(-1).expand(b, self.eca_topk_oli, h, w)
            oli_masked.scatter_(1, oli_idx_exp, oli_filtered)

            bottleneck, skip = self.oli_encoder(oli_masked)
            x = bottleneck

            for i in range(len(self.pixel_shuffle_layers)):
                x = self.pixel_shuffle_layers[i](x)
                if i < len(self.decoder_convs):
                    s = skip[-(i+1)]
                    if x.shape[2:] != s.shape[2:]:
                        s = F.interpolate(s, size=x.shape[2:], mode='bilinear', align_corners=False)
                    s = self.skip_align_oli[i](s)
                    x = torch.cat([x, s], dim=1)
                    x = self.decoder_convs[i](x)

            recon, seg = self.head2(x)
            return {
                'head2_oli_recon': recon,
                'head2_oli_seg': seg,
                'oli_topk_indices': oli_topk_indices,
            }

        # ========== 双支路模式（原有逻辑）==========
        # ========== ECA输入过滤 ==========
        msi_filtered = self.msi_eca(msi_input)  # [B, K1, H, W]
        oli_filtered = self.oli_eca(oli_input)  # [B, K2, H, W]

        # 保存ECA选择的通道索引
        msi_topk_indices = self.msi_eca.last_topk_indices  # [B, K1]
        oli_topk_indices = self.oli_eca.last_topk_indices  # [B, K2]

        # 将选中的K个通道放回原始位置，其余填0
        msi_masked = torch.zeros_like(msi_input)  # [B, C_msi, H, W]
        h, w = msi_input.shape[2:]
        msi_idx_exp = msi_topk_indices.unsqueeze(-1).unsqueeze(-1).expand(b, self.eca_topk_msi, h, w)
        msi_masked.scatter_(1, msi_idx_exp, msi_filtered)

        oli_masked = torch.zeros_like(oli_input)  # [B, C_oli, H, W]
        h, w = oli_input.shape[2:]
        oli_idx_exp = oli_topk_indices.unsqueeze(-1).unsqueeze(-1).expand(b, self.eca_topk_oli, h, w)
        oli_masked.scatter_(1, oli_idx_exp, oli_filtered)

        # ========== 编码器（MSI + OLI 双支路）==========
        msi_feat, msi_skip = self.msi_encoder(msi_masked)
        oli_feat, oli_skip = self.oli_encoder(oli_masked)

        # ========== Gate融合（MSI + OLI）==========
        fused = self.gate_fusion(msi_feat, oli_feat)

        # ========== 解码器（使用MSI+OLI skip融合）==========
        x = fused
        for i in range(len(self.pixel_shuffle_layers)):
            x = self.pixel_shuffle_layers[i](x)
            if i < len(self.decoder_convs):
                skip_msi = msi_skip[-(i+1)]
                skip_oli = oli_skip[-(i+1)]

                if x.shape[2:] != skip_msi.shape[2:]:
                    skip_msi = F.interpolate(skip_msi, size=x.shape[2:], mode='bilinear', align_corners=False)
                if x.shape[2:] != skip_oli.shape[2:]:
                    skip_oli = F.interpolate(skip_oli, size=x.shape[2:], mode='bilinear', align_corners=False)

                skip_msi = self.skip_align_msi[i](skip_msi)
                skip_oli = self.skip_align_oli[i](skip_oli)

                skip_fused = self.skip_fusion[i](skip_msi, skip_oli)

                x = torch.cat([x, skip_fused], dim=1)
                x = self.decoder_convs[i](x)

        # ========== 输出头 ==========
        head1_recon_init, head1_seg_init = self.head1(x)
        if self.use_sft:
            x_sft1 = self.sft_head1(x, head1_seg_init.detach())
            head1_recon, head1_seg = self.head1(x_sft1)
        else:
            head1_recon, head1_seg = head1_recon_init, head1_seg_init

        head2_recon_init, head2_seg_init = self.head2(x)
        if self.use_sft:
            x_sft2 = self.sft_head2(x, head2_seg_init.detach())
            head2_recon, head2_seg = self.head2(x_sft2)
        else:
            head2_recon, head2_seg = head2_recon_init, head2_seg_init

        result = {
            'head1_msi_recon': head1_recon,
            'head1_msi_seg': head1_seg,
            'head2_oli_recon': head2_recon,
            'head2_oli_seg': head2_seg,
            'msi_topk_indices': msi_topk_indices,
            'oli_topk_indices': oli_topk_indices,
        }
        if self.use_sft:
            result['head1_msi_seg_init'] = head1_seg_init
            result['head2_oli_seg_init'] = head2_seg_init
        return result


def test_model():
    """测试模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    model = DualBranchUNet(
        msi_in_channels=12,
        oli_in_channels=7,
        hidden_dim=1024,
        bottleneck_size=32,
        eca_topk_msi=3,    # 测试选择3个MSI通道
        eca_topk_oli=2     # 测试选择2个OLI通道
    ).to(device)

    # 测试输入
    msi_input = torch.randn(2, 12, 512, 512).to(device)
    oli_input = torch.randn(2, 7, 512, 512).to(device)

    print(f"MSI Input shape: {msi_input.shape}")
    print(f"OLI Input shape: {oli_input.shape}")

    # 前向传播
    with torch.no_grad():
        outputs = model(msi_input, oli_input)

    # 打印输出形状
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")

    # 验证重建输出通道数
    assert outputs['head1_msi_recon'].shape[1] == 3, f"Expected 3 MSI channels, got {outputs['head1_msi_recon'].shape[1]}"
    assert outputs['head2_oli_recon'].shape[1] == 2, f"Expected 2 OLI channels, got {outputs['head2_oli_recon'].shape[1]}"
    print("\n✓ Channel selection test passed!")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")

    return model, outputs


if __name__ == "__main__":
    model, outputs = test_model()
