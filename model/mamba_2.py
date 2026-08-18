import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# --- 硬件适配全局配置 ---
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name()
    if "NVIDIA" in device_name:
        torch.set_float32_matmul_precision('high')
    elif "Hygon" in device_name or "DCU" in device_name:
        torch.backends.cuda.matmul.allow_tf32 = False

class OptimizedMamba2(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        expand=2,
        headdim=64,
        chunk_size=64,
        conv_kernel=4,
        dt_min=0.001,
        dt_max=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.headdim = headdim
        self.d_state = d_state
        self.chunk_size = chunk_size
        self.nheads = self.d_inner // headdim
        self.dt_min = dt_min
        self.dt_max = dt_max

        # 算子融合投影层
        self.d_in_proj = 2 * self.d_inner + 2 * self.nheads * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, self.d_in_proj, bias=False)

        # 深度可分离卷积
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=conv_kernel,
            groups=self.d_inner,
            padding=conv_kernel - 1,
        )

        # 离散化参数 - 使用更稳定的初始化
        # dt_proj_bias用于将dt限制在合理范围
        dt_init_std = self.d_inner ** -0.5
        self.dt_bias = nn.Parameter(torch.rand(self.nheads) * (dt_max - dt_min) + dt_min)

        # A参数: 使用负值确保衰减
        # 标准Mamba初始化: A = -torch.arange(1, nheads+1)
        A = -torch.arange(1, self.nheads + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(-A))  # 存储log(-A)

        # D跳跃连接参数
        self.D = nn.Parameter(torch.ones(self.nheads))

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(self.d_inner)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 正交初始化
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)

    def forward(self, u):
        # 在float32中运行整个前向传播以确保数值稳定性
        input_dtype = u.dtype
        u_f32 = u.float()

        batch, seqlen, _ = u_f32.shape

        # 1. 输入投影
        zxbcdt = self.in_proj(u_f32)

        d_bc = self.nheads * self.d_state
        x, z, B, C, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner, d_bc, d_bc, self.nheads],
            dim=-1
        )

        # 2. 卷积
        x = x.transpose(1, 2)
        x_conv = self.conv1d(x)
        if x_conv.shape[-1] < seqlen:
            x_conv = F.pad(x_conv, (0, seqlen - x_conv.shape[-1]))
        x = x_conv[:, :, :seqlen].transpose(1, 2)
        x = F.silu(x)

        # 3. SSD计算
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (h n) -> b l h n", n=self.d_state)
        C = rearrange(C, "b l (h n) -> b l h n", n=self.d_state)

        # dt处理: softplus确保正值，并限制范围
        dt = F.softplus(dt + self.dt_bias)
        dt = torch.clamp(dt, min=self.dt_min, max=self.dt_max)

        # A: 负值
        A = -torch.exp(self.A_log)

        # 保存x用于残差连接 (在ssd_minimal会改变x之前保存)
        x_for_residual = rearrange(x, "b l h p -> b l (h p)")

        # 调用矩阵运算SSD实现
        y = self.ssd_chunk_scan(x, dt, A, B, C, seqlen)

        # 4. 输出 - 添加skip connection和输出限制
        y = rearrange(y, "b l h p -> b l (h p)")

        # 添加残差连接 (skip connection) - 使用原始x
        y = y + x_for_residual * 0.1  # 缩放残差连接

        # 限制输出范围
        y = torch.clamp(y, min=-10.0, max=10.0)

        y = self.norm(y)
        output = self.out_proj(y * F.silu(z))

        # 再次限制输出范围
        output = torch.clamp(output, min=-10.0, max=10.0)

        # 转回原始数据类型
        return output.to(input_dtype)

    def ssd_chunk_scan(self, x, dt, A, B, C, orig_seqlen):
        """
        矩阵运算版SSD实现，利用并行计算提高效率
        基于Mamba2的Structured State Space Duality
        """
        b, l, h, p = x.shape
        n = self.d_state
        c_size = self.chunk_size

        # Padding
        if l % c_size != 0:
            pad_len = c_size - (l % c_size)
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
            dt = F.pad(dt, (0, 0, 0, pad_len))
            B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
            C = F.pad(C, (0, 0, 0, 0, 0, pad_len))
            l = x.shape[1]

        # 1. 离散化 A
        log_dA = dt * A.view(1, 1, h)  # (b, l, h)
        dA = torch.exp(log_dA)

        # 2. 重新排列为 Chunk 格式
        x = rearrange(x, "b (m c) h p -> b m c h p", c=c_size)
        dt = rearrange(dt, "b (m c) h -> b m c h", c=c_size)
        dA = rearrange(dA, "b (m c) h -> b m c h", c=c_size)
        B = rearrange(B, "b (m c) h n -> b m c h n", c=c_size)
        C = rearrange(C, "b (m c) h n -> b m c h n", c=c_size)
        log_dA_chunk = rearrange(log_dA, "b (m c) h -> b m c h", c=c_size)

        # 3. 计算块内矩阵 (Intra-chunk)
        log_dA_cumsum = torch.cumsum(log_dA_chunk, dim=2)
        dA_pairwise = torch.exp(log_dA_cumsum.unsqueeze(3) - log_dA_cumsum.unsqueeze(2))
        dA_pairwise = torch.tril(dA_pairwise)

        dt_x = x * dt.unsqueeze(-1)

        # 4. 计算块间状态传递 (Inter-chunk)
        chunk_states = torch.einsum("bmchn,bmchp->bmhnp", B, dt_x)
        chunk_dA = torch.exp(log_dA_cumsum[:, :, -1])

        # 跨块递归 (Scan)
        prev_state = torch.zeros(b, h, n, p, device=x.device, dtype=x.dtype)
        all_chunk_states = []
        for i in range(chunk_states.shape[1]):
            curr_state = chunk_dA[:, i, :, None, None] * prev_state + chunk_states[:, i]
            all_chunk_states.append(prev_state)
            prev_state = curr_state
        inter_states = torch.stack(all_chunk_states, dim=1)

        # 5. 合并块内与块间
        exp_log_dA_cumsum = torch.exp(log_dA_cumsum)
        y_inter = torch.einsum("bmchn,bmhnp,bmch->bmchp", C, inter_states, exp_log_dA_cumsum)
        y_intra = torch.einsum("bmcih,bmihn,bmihp->bmchp", dA_pairwise, B, dt_x)
        y_intra = torch.einsum("bmchn,bmchp->bmchp", C, y_intra)

        out = y_intra + y_inter
        out = rearrange(out, "b m c h p -> b (m c) h p")
        return out[:, :orig_seqlen, :, :]

# --- 编译接口 ---
def get_compiled_mamba2(d_model=128, device="cuda"):
    model = OptimizedMamba2(d_model=d_model).to(device)

    if "NVIDIA" in torch.cuda.get_device_name():
        return torch.compile(model, mode="max-autotune")
    else:
        return torch.compile(model, mode="reduce-overhead")

# --- 测试代码 ---
if __name__ == "__main__":
    batch, seq_len, d_model = 2, 512, 128
    x = torch.randn(batch, seq_len, d_model).cuda()

    model = get_compiled_mamba2(d_model).cuda()

    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output has nan: {torch.isnan(out).any().item()}")
    print(f"Output min: {out.min().item():.4f}, max: {out.max().item():.4f}")
