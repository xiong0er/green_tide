import triton
import triton.language as tl

# 自动调优：4090 和 3390 的计算/访存比不同，需要不同的 BLOCK 尺寸
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'num_warps': 4, 'num_stages': 3}, num_stages=3 if "NVIDIA" in torch.cuda.get_device_name() else 1),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'num_warps': 8, 'num_stages': 2}, num_stages=2),
    ],
    key=['W', 'K'],
)
@triton.jit
def window_topk_kernel(
    Q, K, V, Out,
    stride_qm, stride_kn, stride_vn, stride_om,
    n_heads, d_model, seq_len,
    W,  # Window Size
    K_val, # Top-K
    sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 1. 索引计算
    pid = tl.program_id(0)
    head_id = tl.program_id(1)
    batch_id = tl.program_id(2)
    
    # 针对海光 3390 的合并访存优化：确保连续线程访问连续地址
    # 对于 4090，L2 缓存会自动处理合并，但显式对齐仍有帮助
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rk = tl.arange(0, d_model)

    # 2. 加载 Q (SRAM)
    off_q = batch_id * (n_heads * seq_len * d_model) + head_id * (seq_len * d_model) + rm[:, None] * d_model + rk[None, :]
    q = tl.load(Q + off_q, mask=rm[:, None] < seq_len, other=0.0)

    # 3. 局部窗口扫描 (Flash Attention 2 逻辑)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, d_model], dtype=tl.float32)

    # 确定窗口范围：[m-W, m+W]
    m_start = pid * BLOCK_M
    lo = tl.maximum(0, m_start - W)
    hi = tl.minimum(seq_len, m_start + BLOCK_M + W)

    for n_start in range(lo, hi, BLOCK_N):
        # 针对 3390 的 LDS 优化：BLOCK_N 保持为 32/64 以对齐 Wavefront
        off_k = batch_id * (n_heads * seq_len * d_model) + head_id * (seq_len * d_model) + (n_start + rn)[None, :] * d_model + rk[:, None]
        k = tl.load(K + off_k, mask=(n_start + rn)[None, :] < seq_len, other=0.0)
        
        # 计算点积
        qk = tl.dot(q, k) * sm_scale
        
        # 窗口掩码
        dist = tl.abs(rm[:, None] - (n_start + rn)[None, :])
        qk = tl.where(dist <= W, qk, float("-inf"))
        
        # --- Top-k 优化逻辑 ---
        # 针对 4090: 使用阈值过滤，减少分枝预测开销
        # 针对 3390: 利用其较强的原子加和并行性
        if K_val < BLOCK_N:
            # 这里的 Top-k 是基于块的简化实现
            # 真正的高性能 Top-k 在 Triton 中通常通过 tl.sort 实现
            qk_sorted = tl.sort(qk, dim=1)
            thresh = tl.view(qk_sorted[:, BLOCK_N - K_val], [BLOCK_M, 1])
            qk = tl.where(qk >= thresh, qk, float("-inf"))

        # Softmax 更新
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        prev_m_i = m_i
        m_i = tl.maximum(m_i, m_ij)
        alpha = tl.exp(prev_m_i - m_i)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]
        off_v = batch_id * (n_heads * seq_len * d_model) + head_id * (seq_len * d_model) + (n_start + rn)[:, None] * d_model + rk[None, :]
        v = tl.load(V + off_v, mask=(n_start + rn)[:, None] < seq_len, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)

    # 写回结果
    off_o = batch_id * (n_heads * seq_len * d_model) + head_id * (seq_len * d_model) + rm[:, None] * d_model + rk[None, :]
    tl.store(Out + off_o, (acc / l_i[:, None]).to(Out.dtype.element_ty), mask=rm[:, None] < seq_len)