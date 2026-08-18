import torch
import torch.nn as nn
from einops import rearrange
import importlib.util
import sys
from pathlib import Path

# 动态导入mamba_2.py（因为文件名包含下划线，使用正确的文件名）
mamba2_path = Path(__file__).parent / "mamba_2.py"
spec = importlib.util.spec_from_file_location("mamba2_module", mamba2_path)
mamba2_module = importlib.util.module_from_spec(spec)
sys.modules["mamba2_module"] = mamba2_module
spec.loader.exec_module(mamba2_module)
Mamba2Block = mamba2_module.OptimizedMamba2


class VerticalVisionMamba2Block(nn.Module):
    """原始版本：在整个列上扫描"""
    def __init__(self, d_model, d_state=64, expand=2):
        super().__init__()
        # 向上扫描 (Bottom-to-Top) 和 向下扫描 (Top-to-Bottom)
        self.down_mamba = Mamba2Block(d_model=d_model, d_state=d_state, expand=expand)
        self.up_mamba = Mamba2Block(d_model=d_model, d_state=d_state, expand=expand)

        self.norm = nn.LayerNorm(d_model)
        self.final_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x, h, w):
        """
        x: (B, L, D)  其中 L = H * W
        h, w: 图像的高度和宽度
        """
        b, l, d = x.shape

        # 1. 空间重组：从行优先转为列优先 (垂直化)
        # (B, L, D) -> (B, H, W, D)
        x_2d = x.view(b, h, w, d)

        # 垂直向下扫描序列：(B, W, H, D) -> (B, W*H, D)
        # 这样在 Mamba 看来，序列是沿着"列"走的
        x_vert = rearrange(x_2d, 'b h w d -> b (w h) d')

        # 2. 准备垂直反向序列 (向上扫描)
        x_vert_down = x_vert
        x_vert_up = x_vert.flip(dims=[1])  # 沿垂直序列翻转

        # 3. 执行 SSD 计算
        # 在 4090/3390 上，这两个分支会并行发射
        out_down = self.down_mamba(x_vert_down)
        out_up = self.up_mamba(x_vert_up).flip(dims=[1])

        # 4. 融合并还原空间顺序
        out_vert = torch.cat([out_down, out_up], dim=-1)  # (B, W*H, 2*D)

        # 还原回原始的行优先顺序 (B, W*H, 2*D) -> (B, H, W, 2*D) -> (B, H*W, 2*D)
        out = rearrange(out_vert, 'b (w h) d2 -> b (h w) d2', h=h, w=w)

        return self.final_proj(out)


class WindowedVisionMamba2Block(nn.Module):
    """
    窗口分块版本：将特征图分成不重叠的窗口，每个窗口内独立运行Mamba2
    大幅降低序列长度和内存占用
    """
    def __init__(self, d_model, d_state=64, expand=2, window_size=8):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size

        # 向上扫描 (Bottom-to-Top) 和 向下扫描 (Top-to-Bottom)
        self.down_mamba = Mamba2Block(d_model=d_model, d_state=d_state, expand=expand)
        self.up_mamba = Mamba2Block(d_model=d_model, d_state=d_state, expand=expand)

        self.norm = nn.LayerNorm(d_model)
        self.final_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x, h, w):
        """
        x: (B, L, D)  其中 L = H * W
        h, w: 图像的高度和宽度
        """
        b, l, d = x.shape
        ws = self.window_size

        # 确保 H, W 能被窗口大小整除
        assert h % ws == 0 and w % ws == 0, f"H={h}, W={w} must be divisible by window_size={ws}"

        # 1. 重塑为窗口格式
        # (B, H*W, D) -> (B, H, W, D)
        x_2d = x.view(b, h, w, d)

        # 2. 分窗口: (B, H, W, D) -> (B, num_h, num_w, ws, ws, D)
        # 其中 num_h = H/ws, num_w = W/ws
        x_windows = rearrange(
            x_2d,
            'b (num_h ws_h) (num_w ws_w) d -> b num_h num_w (ws_h ws_w) d',
            ws_h=ws, ws_w=ws
        )

        # 合并batch和窗口维度: (B*num_h*num_w, ws*ws, D)
        num_h, num_w = h // ws, w // ws
        num_windows = num_h * num_w
        x_windows = x_windows.reshape(b * num_windows, ws * ws, d)

        # 3. 准备垂直反向序列 (翻转每个窗口内的序列)
        x_windows_down = x_windows
        x_windows_up = x_windows.flip(dims=[1])

        # 4. 执行 SSD 计算（每个窗口独立并行）
        out_down = self.down_mamba(x_windows_down)
        out_up = self.up_mamba(x_windows_up).flip(dims=[1])

        # 5. 融合双向结果
        out_windows = torch.cat([out_down, out_up], dim=-1)  # (B*num_windows, ws*ws, 2*D)
        out_windows = self.final_proj(out_windows)  # (B*num_windows, ws*ws, D)

        # 6. 恢复窗口到原始空间布局
        # (B*num_windows, ws*ws, D) -> (B, num_h, num_w, ws*ws, D)
        out_windows = out_windows.view(b, num_h, num_w, ws * ws, d)

        # 7. 反窗口化: (B, num_h, num_w, ws*ws, D) -> (B, H, W, D)
        out_2d = rearrange(
            out_windows,
            'b num_h num_w (ws_h ws_w) d -> b (num_h ws_h) (num_w ws_w) d',
            ws_h=ws, ws_w=ws
        )

        # 8. 展平回序列格式: (B, H*W, D)
        out = out_2d.reshape(b, h * w, d)

        return out


class CrossScanVisionMamba2Block(nn.Module):
    """
    交叉扫描版本：列+行四方向扫描（替代 WVM2B 的仅列扫描）

    对每个 8×8 窗口独立运行 4 个方向的 Mamba-2 SSM：
    - 列向下 (top→bottom) + 列向上 (bottom→top)
    - 行向右 (left→right) + 行向左 (right→left)

    四路输出拼接后投影回 d_model，提供真正的 2D 上下文感知。
    """
    def __init__(self, d_model, d_state=64, expand=1, window_size=8):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size

        # 四个方向的 Mamba-2
        self.col_down = Mamba2Block(d_model=d_model, d_state=d_state, expand=expand)
        self.col_up   = Mamba2Block(d_model=d_model, d_state=d_state, expand=expand)
        self.row_right = Mamba2Block(d_model=d_model, d_state=d_state, expand=expand)
        self.row_left  = Mamba2Block(d_model=d_model, d_state=d_state, expand=expand)

        self.norm = nn.LayerNorm(d_model)
        self.final_proj = nn.Linear(d_model * 4, d_model)

    def forward(self, x, h, w):
        b, l, d = x.shape
        ws = self.window_size
        assert h % ws == 0 and w % ws == 0

        # 窗口化: (B, H*W, D) → (B*N, ws*ws, D)
        x_2d = x.view(b, h, w, d)
        x_windows = rearrange(x_2d, 'b (nh ws_h) (nw ws_w) d -> b nh nw (ws_h ws_w) d',
                              ws_h=ws, ws_w=ws)
        nh, nw = h // ws, w // ws
        N = nh * nw
        xw = x_windows.reshape(b * N, ws * ws, d)  # [B*N, 64, D]

        # ---- 列扫描 (col-major: H 在内循环) ----
        xw_col = rearrange(x_windows, 'b nh nw (ws_h ws_w) d -> b nh nw (ws_h ws_w) d',
                          ws_h=ws, ws_w=ws)
        # 重塑为 col-major: 每一列是 ws 个像素按 H 方向排列
        xw_col = x_windows.reshape(b, nh, nw, ws, ws, d)   # [B, nh, nw, ws_h, ws_w, D]
        xw_col = rearrange(xw_col, 'b nh nw sh sw d -> b nh nw (sh sw) d', sh=ws, sw=ws)
        xw_col = xw_col.reshape(b * N, ws * ws, d)

        # 列向下 + 列向上
        out_col_down = self.col_down(xw_col)
        out_col_up   = self.col_up(xw_col.flip(dims=[1])).flip(dims=[1])

        # ---- 行扫描 (row-major: W 在内循环) ----
        # 转置窗口内 H/W
        xw_row = rearrange(x_windows, 'b nh nw (ws_h ws_w) d -> b nh nw (ws_w ws_h) d',
                           ws_h=ws, ws_w=ws)
        xw_row = xw_row.reshape(b * N, ws * ws, d)

        out_row_right = self.row_right(xw_row)
        out_row_left  = self.row_left(xw_row.flip(dims=[1])).flip(dims=[1])

        # 融合四路
        out_windows = torch.cat([out_col_down, out_col_up, out_row_right, out_row_left], dim=-1)
        out_windows = self.final_proj(out_windows)  # [B*N, 64, D]

        # 反窗口化
        out_windows = out_windows.view(b, nh, nw, ws * ws, d)
        out_2d = rearrange(out_windows, 'b nh nw (ws_h ws_w) d -> b (nh ws_h) (nw ws_w) d',
                           ws_h=ws, ws_w=ws)
        return out_2d.reshape(b, h * w, d)


class HorizontalScanVisionMamba2Block(nn.Module):
    """
    行扫描版本：仅行方向（左右）双向扫描

    对每个 8×8 窗口独立运行 2 个方向的 Mamba-2 SSM：
    - 行向右 (left→right) + 行向左 (right→left)

    与 CrossScan (四方向) 相比更轻量，与 WVM2B (列双向) 扫描方向正交。
    """
    def __init__(self, d_model, d_state=64, expand=1, window_size=8):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size

        self.row_right = Mamba2Block(d_model=d_model, d_state=d_state, expand=expand)
        self.row_left  = Mamba2Block(d_model=d_model, d_state=d_state, expand=expand)

        self.norm = nn.LayerNorm(d_model)
        self.final_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x, h, w):
        b, l, d = x.shape
        ws = self.window_size
        assert h % ws == 0 and w % ws == 0, f"H={h}, W={w} must be divisible by window_size={ws}"

        # 窗口化: (B, H*W, D) → (B*N, ws*ws, D)
        x_2d = x.view(b, h, w, d)
        x_windows = rearrange(
            x_2d, 'b (nh ws_h) (nw ws_w) d -> b nh nw (ws_h ws_w) d',
            ws_h=ws, ws_w=ws
        )
        nh, nw = h // ws, w // ws
        N = nh * nw

        # 行扫描 (row-major 序列)
        xw = x_windows.reshape(b * N, ws * ws, d)

        out_row_right = self.row_right(xw)
        out_row_left  = self.row_left(xw.flip(dims=[1])).flip(dims=[1])

        # 融合两路
        out_windows = torch.cat([out_row_right, out_row_left], dim=-1)
        out_windows = self.final_proj(out_windows)  # [B*N, 64, D]

        # 反窗口化
        out_windows = out_windows.view(b, nh, nw, ws * ws, d)
        out_2d = rearrange(
            out_windows, 'b nh nw (ws_h ws_w) d -> b (nh ws_h) (nw ws_w) d',
            ws_h=ws, ws_w=ws
        )
        return out_2d.reshape(b, h * w, d)


class WindowTransformerBlock(nn.Module):
    """
    窗口 Transformer 块：8×8 窗口内 Multi-Head Self-Attention + MLP

    替代 WVM2B 用于前两层浅层特征提取。与 Swin Transformer 的 W-MSA 设计一致:
    窗口分区 → LN → QKV → 窗口内 Attention → 残差 → LN → MLP(GELU, 4x) → 残差
    """
    def __init__(self, d_model, num_heads=8, window_size=8, mlp_ratio=4):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(self, x, h, w):
        b, l, d = x.shape
        ws = self.window_size
        assert h % ws == 0 and w % ws == 0

        shortcut = x
        x_n = self.norm1(x)

        # QKV: (B, L, D) → (B, L, 3*D)
        qkv = self.qkv(x_n).reshape(b, l, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]   # each (B, heads, L, head_dim)

        # 窗口化: (B, heads, H*W, D) → (B, heads, Nw, ws*ws, D)
        q = q.reshape(b, self.num_heads, h//ws, ws, w//ws, ws, self.head_dim)
        q = q.permute(0, 1, 2, 4, 3, 5, 6).reshape(b, self.num_heads, -1, ws*ws, self.head_dim)
        k = k.reshape(b, self.num_heads, h//ws, ws, w//ws, ws, self.head_dim)
        k = k.permute(0, 1, 2, 4, 3, 5, 6).reshape(b, self.num_heads, -1, ws*ws, self.head_dim)
        v = v.reshape(b, self.num_heads, h//ws, ws, w//ws, ws, self.head_dim)
        v = v.permute(0, 1, 2, 4, 3, 5, 6).reshape(b, self.num_heads, -1, ws*ws, self.head_dim)

        # 窗口内 Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B, heads, Nw, ws*ws, head_dim)

        # 反窗口化
        nh, nw = h//ws, w//ws
        out = out.reshape(b, self.num_heads, nh, nw, ws, ws, self.head_dim)
        out = out.permute(0, 1, 2, 4, 3, 5, 6).reshape(b, self.num_heads, l, self.head_dim)
        out = out.transpose(1, 2).reshape(b, l, d)
        out = self.proj(out)

        # Residual + MLP
        out = shortcut + out
        out = out + self.mlp(self.norm2(out))
        return out

