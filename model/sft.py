import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _sft_fwd_kernel(
    X, Gamma, Beta, Out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 算子融合：在一个线程中同时处理 乘法(gamma) 和 加法(beta)
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 合并访存：同时读取数据
    x = tl.load(X + offsets, mask=mask)
    g = tl.load(Gamma + offsets, mask=mask)
    b = tl.load(Beta + offsets, mask=mask)

    # 算子融合计算：y = g * x + b
    # 在 4090 上会编译为 FFMA 指令
    res = g * x + b

    tl.store(Out + offsets, res, mask=mask)

@torch.compiler.disable
def triton_sft(x, gamma, beta):
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # 动态调整 BLOCK_SIZE 以适配 4090 (建议 1024) 和 3390 (建议 512/256)
    _sft_fwd_kernel[grid](
        x, gamma, beta, out,
        n_elements,
        BLOCK_SIZE=1024 if "NVIDIA" in torch.cuda.get_device_name() else 512
    )
    return out

class SpatialFeatureTransform(nn.Module):
    def __init__(self, in_channels, cond_channels):
        super().__init__()
        # 优化：将生成 gamma 和 beta 的卷积合并为一个，减少 Kernel Launch 次数
        self.conv_gamma_beta = nn.Sequential(
            nn.Conv2d(cond_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1)
        )

    def forward(self, x, cond):
        """
        x: (B, C, H, W) - 基础特征图
        cond: (B, d_cond, H, W) - 空间条件（如语义分割图）
        """
        # 1. 自动混合精度：确保卷积阶段使用 bfloat16/float16
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # 一次性生成所有缩放和偏移系数
            stats = self.conv_gamma_beta(cond)
            # 算子融合：在通道维度分割 gamma 和 beta
            gamma, beta = torch.chunk(stats, 2, dim=1)

        # 2. 调用高性能 Triton Kernel 或经过编译的 PyTorch 算子
        # 在 4090/3390 上，这种 element-wise 操作受限于带宽而非计算
        # Triton 实现可以显著减少中间张量的显存读写
        if x.is_cuda and x.is_contiguous():
            return triton_sft(x, gamma, beta)
        else:
            # 回退到 PyTorch 编译模式
            return gamma * x + beta

# 针对 4090/3390 的全局编译配置
def compile_sft_module(module):
    # reduce-overhead 模式在 3390 上稳定性更好
    # max-autotune 在 4090 上能通过 Inductor 生成更强的融合代码
    mode = "max-autotune" if "NVIDIA" in torch.cuda.get_device_name() else "reduce-overhead"
    return torch.compile(module, mode=mode)