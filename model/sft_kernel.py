import torch
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