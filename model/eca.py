import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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

class ECALayer(nn.Module):
    """
    Efficient Channel Attention (ECA) Layer with multiple operation modes.

    Modes:
    - 'topk': Hard Top-K channel selection (default). 物理截断，输出 [B, K, H, W]
    - 'soft': Soft attention weighting. 用 sigmoid 权重乘所有波段，输出 [B, C, H, W]
    - 'random_fixed': Randomly select K bands at init, fix for all samples. 输出 [B, K, H, W]

    训练时添加退火高斯噪声（初始σ，指数退火）。
    """
    def __init__(self, channel, k_size=3, topk=3, noise_std_init=0.1, mode='topk'):
        super(ECALayer, self).__init__()
        self.channel = channel
        self.topk = min(topk, channel)
        self.noise_std_init = noise_std_init
        self.mode = mode
        self.training_step = 0

        # 当前噪声标准差（会随训练退火）
        self.register_buffer('current_noise_std', torch.tensor(noise_std_init))

        # ECA注意力机制
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

        # Random fixed mode: pre-select bands
        if mode == 'random_fixed':
            rand_indices = torch.randperm(channel)[:self.topk]
            self.register_buffer('fixed_indices', rand_indices)
        else:
            self.register_buffer('fixed_indices', torch.arange(self.topk))

    def set_epoch(self, epoch, total_epochs):
        """
        设置当前epoch，更新噪声退火
        指数退火: noise_std = init * exp(-epoch / (total_epochs/5))
        """
        # 指数退火，约5个epoch衰减到1/e
        decay_factor = math.exp(-epoch / max(total_epochs / 5, 1))
        new_std = self.noise_std_init * decay_factor
        self.current_noise_std.copy_(torch.tensor(new_std, device=self.current_noise_std.device))

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape

        # ---- Random Fixed mode: always return same K bands ----
        if self.mode == 'random_fixed':
            idx = self.fixed_indices.unsqueeze(0).expand(b, -1)  # [B, K]
            idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(b, self.topk, h, w)
            output = torch.gather(x, 1, idx_exp)
            self.last_topk_indices = idx.detach()
            self.last_attention_weights = torch.zeros(b, c, device=x.device)
            return output.float()

        # 计算通道注意力权重
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = y.float()
        y = self.conv(y)  # [B, 1, C]
        y = y.transpose(-1, -2)  # [B, C, 1]
        y = self.sigmoid(y).squeeze(-1)  # [B, C]

        # 添加退火高斯噪声（只在训练时添加）
        if self.training:
            noise = torch.randn_like(y) * self.current_noise_std
            y = y + noise
            # 重新归一化到0-1范围
            y = torch.sigmoid((y - y.mean(dim=1, keepdim=True)) / (y.std(dim=1, keepdim=True) + 1e-6))

        # ---- Soft mode: apply weights as multipliers ----
        if self.mode == 'soft':
            weights = y.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            output = x * weights  # [B, C, H, W] — all bands, scaled
            self.last_topk_indices = torch.zeros(b, self.topk, dtype=torch.long, device=x.device)
            self.last_attention_weights = y.detach()
            return output.float()

        # ---- TopK mode (default): hard truncation ----
        # 获取top-k通道索引
        topk_values, topk_indices = torch.topk(y, k=self.topk, dim=1)  # [B, K]

        # 保存top-k索引和注意力权重（用于loss计算）
        self.last_topk_indices = topk_indices.detach()  # [B, K]
        self.last_attention_weights = y.detach()

        # 物理截取topk波段
        topk_indices_expanded = topk_indices.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
        topk_indices_expanded = topk_indices_expanded.expand(-1, -1, h, w)  # [B, K, H, W]

        # 截取topk通道
        output = torch.gather(x, 1, topk_indices_expanded)  # [B, K, H, W]

        return output.float()

class ECANet(nn.Module):
    """
    ECANet block with physical top-k channel selection
    输出通道数变为topk（不再是in_channels）
    """
    def __init__(self, in_channels, out_channels=None, k_num=3, topk=3, noise_std_init=0.3):
        super(ECANet, self).__init__()
        self.k_num = k_num
        self.topk = topk
        self.eca = ECALayer(in_channels, k_size=k_num, topk=topk, noise_std_init=noise_std_init)
        # 注意：out_channels不再是in_channels，而是topk

    def set_epoch(self, epoch, total_epochs):
        """传递epoch信息给ECA层"""
        self.eca.set_epoch(epoch, total_epochs)

    def forward(self, x):
        x = self.eca(x)  # [B, K, H, W]
        return x
