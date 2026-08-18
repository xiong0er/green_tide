import torch
import torch.nn as nn

class DyT(torch.nn.Module):
    def __init__(self, dim, init_a=0.1):
        super(DyT, self).__init__()
        self.a = nn.Parameter(torch.ones(1) * init_a)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.dim = dim

    def forward(self, x):
        # 处理输入，确保可以正确应用归一化
        if len(x.shape) == 3:  # [B, L, C]
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x_norm = (x - mean) / torch.rsqrt(var + 1e-5)
            return x_norm * self.gamma.view(1, 1, -1) + self.beta.view(1, 1, -1)
        elif len(x.shape) == 4:  # [B, C, H, W]
            mean = x.mean(dim=[2, 3], keepdim=True)
            var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
            x_norm = (x - mean) / (var + 1e-5).sqrt()
            return x_norm * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")