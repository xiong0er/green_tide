import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedMorphologicalLoss(nn.Module):
    def __init__(self, device_type="nvidia", use_circularity=False,
                 use_shape=True, use_connectivity=True, use_multiscale=True):
        super().__init__()
        # 预设 Sobel 算子
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kx)
        self.register_buffer('kernel_y', ky)

        # 针对 4090 或 3390 调整
        self.is_nvidia = "nvidia" in device_type.lower()
        self.use_circularity = use_circularity

        # 子项消融开关
        self.use_shape = use_shape
        self.use_connectivity = use_connectivity
        self.use_multiscale = use_multiscale

    def forward(self, y_pred, y_true):
        # 禁用 AMP 以确保数值稳定性 - 形态学操作需要精度
        with torch.cuda.amp.autocast(enabled=False):
            return self._compute_loss(y_pred.float(), y_true.float())

    # 使用 torch.compile 装饰核心计算，Inductor 会自动将 Sobel, Sqrt, MaxPool 融合成一个 Triton Kernel
    def _compute_loss(self, y_pred, y_true):
        B, _, H, W = y_pred.shape
        n_pixels = H * W  # 归一化到 [0,1] 每像素

        # 1. 归一化面积 (algae fraction per batch item)
        area_p = torch.sum(y_pred, dim=(1, 2, 3)) / n_pixels  # [B], ~0-1
        area_t = torch.sum(y_true, dim=(1, 2, 3)) / n_pixels  # [B], ~0-1

        # 2. 归一化周长 (edge fraction per batch item)
        perim_p = self._fast_perimeter(y_pred) / n_pixels
        perim_t = self._fast_perimeter(y_true) / n_pixels

        # 3. 形状损失：紧凑度对比
        # compactness = perimeter² / area, 用 log 稳定数值
        total = 0.0
        if self.use_shape:
            compact_p = torch.log((perim_p**2) / (area_p + 1e-4) + 1)
            compact_t = torch.log((perim_t**2) / (area_t + 1e-4) + 1)
            l_shape = F.smooth_l1_loss(compact_p, compact_t)  # ~0-1
            total = total + l_shape

        # 4. 归一化骨架连通性
        if self.use_connectivity:
            skel_p = self._fast_skeleton(y_pred, iterations=3) / n_pixels
            skel_t = self._fast_skeleton(y_true, iterations=3) / n_pixels
            l_connectivity = F.smooth_l1_loss(skel_p, skel_t)  # ~0-1
            total = total + 0.3 * l_connectivity

        # 5. 归一化多尺度面积
        if self.use_multiscale:
            l_complexity = self._multi_scale_area_loss(y_pred, y_true, H, W)  # ~0-1
            total = total + 0.1 * l_complexity

        return total

    def _fast_perimeter(self, x):
        # 确保 kernel 与输入在同一设备和 dtype
        kernel_x = self.kernel_x.to(x.device, x.dtype)
        kernel_y = self.kernel_y.to(x.device, x.dtype)
        grad_x = F.conv2d(x.float(), kernel_x, padding=1)
        grad_y = F.conv2d(x.float(), kernel_y, padding=1)
        return torch.sum(torch.sqrt(grad_x**2 + grad_y**2 + 1e-6), dim=(1, 2, 3))

    def _fast_skeleton(self, x, iterations=3):
        skeleton = 0
        curr = x
        # 编译器会优化这里的迭代。对于海光 3390，显存带宽是瓶颈，
        # 这里编译后会利用缓存减少对 curr 的重复读取
        for _ in range(iterations):
            eroded = -F.max_pool2d(-curr, kernel_size=3, stride=1, padding=1)
            skeleton += torch.sum(curr - eroded, dim=(1, 2, 3))
            curr = eroded
        return skeleton

    def _multi_scale_area_loss(self, yp, yt, H, W):
        loss = 0
        n_pixels = H * W
        for s in [2, 4, 8]:
            yp_s = torch.sum(F.avg_pool2d(yp, s), dim=(1, 2, 3)) / (n_pixels / s / s)
            yt_s = torch.sum(F.avg_pool2d(yt, s), dim=(1, 2, 3)) / (n_pixels / s / s)
            loss += F.mse_loss(yp_s, yt_s)
        return loss
