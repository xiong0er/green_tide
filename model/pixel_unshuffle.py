import torch
import triton
import triton.language as tl

@triton.jit
def _pixel_unshuffle_kernel(
    X, Out,
    n_elements,
    C, H, W,
    r,  # downscale_factor
    out_C, out_H, out_W,
    BLOCK_SIZE: tl.constexpr,
):
    # 计算输出张量的索引
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements

    # 输出索引分解：(b, oc, oh, ow)
    # 假设输出布局是 NCHW (为了兼容性)
    ow = idx % out_W
    oh = (idx // out_W) % out_H
    oc = (idx // (out_W * out_H)) % out_C
    b = idx // (out_W * out_H * out_C)

    # 映射回输入索引 (N, C, H, W)
    # PixelUnshuffle 逻辑：
    # oc = c * r^2 + ry * r + rx
    # 所以输入 c = oc % C
    # 输入 h = oh * r + (oc // C) // r
    # 输入 w = ow * r + (oc // C) % r
    
    in_c = oc % C
    r_offset = oc // C
    ry = r_offset // r
    rx = r_offset % r
    
    in_h = oh * r + ry
    in_w = ow * r + rx
    
    in_idx = b * (C * H * W) + in_c * (H * W) + in_h * W + in_w
    
    # 核心优化：利用 Triton 的异步加载
    data = tl.load(X + in_idx, mask=mask)
    tl.store(Out + idx, data, mask=mask)

def triton_pixel_unshuffle(x, r):
    batch, c, h, w = x.shape
    out_c, out_h, out_w = c * (r**2), h // r, w // r
    out = torch.empty((batch, out_c, out_h, out_w), device=x.device, dtype=x.dtype)
    
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # 4090 建议 1024, 3390 建议 512 以对齐其 Wavefront
    _pixel_unshuffle_kernel[grid](
        x, out,
        n_elements,
        c, h, w,
        r, out_c, out_h, out_w,
        BLOCK_SIZE=1024 if "NVIDIA" in torch.cuda.get_device_name() else 512
    )
    return out

class FastPixelUnshuffle(torch.nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.r = upscale_factor

    def forward(self, x):
        # 1. 自动利用编译融合
        # 如果是简单的 reshape+permute，torch.compile 会优化它
        # 但我们这里提供一个更强的融合入口
        if x.is_cuda:
            if x.stride(-1) == 1: # 已经是 NCHW 连续
                return triton_pixel_unshuffle(x, self.r)
            else:
                # 如果是 NHWC，PyTorch 原生实现其实很快
                # 我们利用 rearrange 配合 compile
                from einops import rearrange
                return rearrange(x, 'b c (h r1) (w r2) -> b (c r1 r2) h w', r1=self.r, r2=self.r)
        return torch.pixel_unshuffle(x, self.r)

# 针对硬件的编译策略
def compile_unshuffle(model):
    if "NVIDIA" in torch.cuda.get_device_name():
        # 4090 使用 max-autotune 会尝试将 unshuffle 与前后的 Conv 融合
        return torch.compile(model, mode="max-autotune")
    else:
        # 海光 3390 使用 reduce-overhead 减少指令分发开销
        return torch.compile(model, mode="reduce-overhead")