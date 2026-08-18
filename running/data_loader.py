import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
import rasterio
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob
import warnings

warnings.filterwarnings('ignore')



class MSIDataset(Dataset):
    """
    MSI 单支路数据加载器 (仅 MSI 输入，Label 嵌入在最后一个波段)
    - MSI 输入：前 N-1 个波段，Label: 第 N 个波段
    """
    def __init__(self,
                 msi_files,
                 transform=None,
                 mode='train',
                 target_size=512,
                 filter_empty_labels=False):
        super().__init__()
        if filter_empty_labels:
            # 过滤全零标签文件
            filtered = []
            for f in msi_files:
                try:
                    with rasterio.open(f) as src:
                        label = src.read(src.count)
                        if label.max() > 0:
                            filtered.append(f)
                except Exception:
                    pass
            print(f"标签过滤: {len(msi_files)} -> {len(filtered)} 文件 (移除全零标签)")
            self.msi_files = filtered
        else:
            self.msi_files = msi_files
        self.transform = transform
        self.mode = mode
        self.target_size = target_size

    def _load_data(self, filepath):
        ext = Path(filepath).suffix.lower()
        if ext == '.npy':
            data = np.load(filepath)
        elif ext in ['.tif', '.tiff']:
            with rasterio.open(filepath) as src:
                data = src.read()  # [C, H, W]
        else:
            raise ValueError(f"Unsupported format: {ext}")

        # 统一确保为 [C, H, W]
        if data.ndim == 3 and data.shape[-1] in [8, 13]:  # 针对 HWC 格式的兼容处理
            data = np.transpose(data, (2, 0, 1))
        return data

    def _resize_to_target(self, data, order=1):
        """将数据 resize 到目标尺寸. order=1 双线性(影像), order=0 最近邻(标签)"""
        c, h, w = data.shape
        if h != self.target_size or w != self.target_size:
            from scipy.ndimage import zoom
            zoom_h = self.target_size / h
            zoom_w = self.target_size / w
            data = zoom(data, (1, zoom_h, zoom_w), order=order)
        return data

    def __len__(self):
        return len(self.msi_files)

    def __getitem__(self, idx):
        # 1. 加载 MSI (假设总共 13 通道：12 数据 + 1 标签)
        msi_raw = self._load_data(self.msi_files[idx])

        # 2. 有效像素掩膜 (在 nan_to_num 之前，NaN 出现在任一光谱波段即无效)
        valid_mask = np.all(np.isfinite(msi_raw[:-1]), axis=0).astype(np.float32)  # [H, W]
        valid_mask = np.expand_dims(valid_mask, axis=0)  # [1, H, W]

        # 3. 处理 NaN 值
        if np.isnan(msi_raw).any():
            msi_raw = np.nan_to_num(msi_raw, nan=0.0)

        msi_img = msi_raw[:-1, :, :].astype(np.float32)  # [C, H, W]
        msi_label = msi_raw[-1:, :, :].astype(np.float32)  # [1, H, W]

        # 4. 分别 resize: 影像用双线性, 标签/掩膜用最近邻
        msi_img = self._resize_to_target(msi_img, order=1)
        msi_label = self._resize_to_target(msi_label, order=0)
        valid_mask = self._resize_to_target(valid_mask, order=0)

        # 5. 基础归一化
        msi_img = self._safe_minmax(msi_img)

        # 6. 数据增强 (valid_mask 作为额外 mask 同步变换)
        if self.transform:
            # 拼接 label 和 valid_mask → [2, H, W]，一起做几何变换后拆分
            combined = np.concatenate([msi_label, valid_mask], axis=0).transpose(1, 2, 0)  # [H, W, 2]
            augmented = self.transform(
                image=msi_img.transpose(1, 2, 0),
                mask=combined
            )
            msi_tensor = augmented['image']  # [11, H, W]
            combined_out = augmented['mask']  # [2, H, W]
            msi_label_tensor = combined_out[0:1]  # [1, H, W]
            valid_mask_tensor = combined_out[1:2]  # [1, H, W]
        else:
            msi_tensor = torch.from_numpy(msi_img)
            msi_label_tensor = torch.from_numpy(msi_label)
            valid_mask_tensor = torch.from_numpy(valid_mask)

        return {
            'msi': msi_tensor,
            'msi_seg_gt': msi_label_tensor,
            'msi_valid_mask': valid_mask_tensor,
        }

    def _safe_minmax(self, data):
        """安全的 minmax 归一化"""
        if np.isnan(data).any():
            data = np.nan_to_num(data, nan=0.0)

        low, high = np.percentile(data, (0.5, 99.5))
        denom = high - low + 1e-8
        result = np.clip((data - low) / denom, 0, 1)
        return result.astype(np.float32)


class MSIOLIDataset(Dataset):
    """
    MSI和OLI数据加载器 (Label 嵌入在最后一个波段)
    - MSI 输入: 前 N-1 个波段, Label: 第 N 个波段
    - OLI 输入: 前 7 个波段, Label: 第 8 个波段
    """
    def __init__(self,
                 msi_files,
                 oli_files,
                 transform=None,
                 mode='train',
                 target_size=512):
        super().__init__()
        self.msi_files = msi_files
        self.oli_files = oli_files
        self.transform = transform
        self.mode = mode
        self.target_size = target_size

    def _load_data(self, filepath):
        ext = Path(filepath).suffix.lower()
        if ext == '.npy':
            data = np.load(filepath)
        elif ext in ['.tif', '.tiff']:
            with rasterio.open(filepath) as src:
                data = src.read()  # [C, H, W]
        else:
            raise ValueError(f"Unsupported format: {ext}")

        # 统一确保为 [C, H, W]
        if data.ndim == 3 and data.shape[-1] in [8, 13]:  # 针对 HWC 格式的兼容处理
            data = np.transpose(data, (2, 0, 1))
        return data

    def _resize_to_target(self, data, order=1):
        """将数据 resize 到目标尺寸. order=1 双线性(影像), order=0 最近邻(标签)"""
        c, h, w = data.shape
        if h != self.target_size or w != self.target_size:
            from scipy.ndimage import zoom
            zoom_h = self.target_size / h
            zoom_w = self.target_size / w
            data = zoom(data, (1, zoom_h, zoom_w), order=order)
        return data

    def __len__(self):
        return len(self.msi_files)

    def __getitem__(self, idx):
        # 1. 加载 MSI (假设总共13通道: 12数据 + 1标签)
        msi_raw = self._load_data(self.msi_files[idx])

        # 2. 加载 OLI (假设总共8通道: 7数据 + 1标签)
        oli_raw = self._load_data(self.oli_files[idx])

        # 3. 有效像素掩膜 (在 nan_to_num 之前)
        msi_valid = np.all(np.isfinite(msi_raw[:-1]), axis=0).astype(np.float32)
        msi_valid = np.expand_dims(msi_valid, axis=0)  # [1, H, W]
        oli_valid = np.all(np.isfinite(oli_raw[:-1]), axis=0).astype(np.float32)
        oli_valid = np.expand_dims(oli_valid, axis=0)  # [1, H, W]

        # 4. 处理NaN值：用0填充
        if np.isnan(msi_raw).any():
            msi_raw = np.nan_to_num(msi_raw, nan=0.0)
        if np.isnan(oli_raw).any():
            oli_raw = np.nan_to_num(oli_raw, nan=0.0)

        msi_img = msi_raw[:-1, :, :].astype(np.float32)  # [C, H, W]
        msi_label = msi_raw[-1:, :, :].astype(np.float32)  # [1, H, W]

        oli_img = oli_raw[:-1, :, :].astype(np.float32)  # [C, H, W]
        oli_label = oli_raw[-1:, :, :].astype(np.float32)  # [1, H, W]

        # 5. 分别 resize
        msi_img = self._resize_to_target(msi_img, order=1)
        msi_label = self._resize_to_target(msi_label, order=0)
        msi_valid = self._resize_to_target(msi_valid, order=0)
        oli_img = self._resize_to_target(oli_img, order=1)
        oli_label = self._resize_to_target(oli_label, order=0)
        oli_valid = self._resize_to_target(oli_valid, order=0)

        # 6. 基础归一化
        msi_img = self._safe_minmax(msi_img)
        oli_img = self._safe_minmax(oli_img)

        # 7. 数据增强 (label+valid_mask 拼接后同步变换，再拆分)
        if self.transform:
            msi_combined = np.concatenate([msi_label, msi_valid], axis=0).transpose(1, 2, 0)  # [H, W, 2]
            oli_combined = np.concatenate([oli_label, oli_valid], axis=0).transpose(1, 2, 0)  # [H, W, 2]
            augmented = self.transform(
                image=msi_img.transpose(1, 2, 0),
                mask=msi_combined,
                oli_image=oli_img.transpose(1, 2, 0),
                oli_mask=oli_combined
            )
            msi_tensor = augmented['image']
            msi_combined_out = augmented['mask']        # [2, H, W]
            msi_label_tensor = msi_combined_out[0:1]    # [1, H, W]
            msi_valid_tensor = msi_combined_out[1:2]    # [1, H, W]
            oli_tensor = augmented['oli_image']
            oli_combined_out = augmented['oli_mask']     # [2, H, W]
            oli_label_tensor = oli_combined_out[0:1]    # [1, H, W]
            oli_valid_tensor = oli_combined_out[1:2]    # [1, H, W]
        else:
            msi_tensor = torch.from_numpy(msi_img)
            msi_label_tensor = torch.from_numpy(msi_label)
            msi_valid_tensor = torch.from_numpy(msi_valid)
            oli_tensor = torch.from_numpy(oli_img)
            oli_label_tensor = torch.from_numpy(oli_label)
            oli_valid_tensor = torch.from_numpy(oli_valid)

        return {
            'msi': msi_tensor,
            'oli': oli_tensor,
            'msi_seg_gt': msi_label_tensor,
            'oli_seg_gt': oli_label_tensor,
            'msi_valid_mask': msi_valid_tensor,
            'oli_valid_mask': oli_valid_tensor,
        }

    def _safe_minmax(self, data):
        """安全的minmax归一化，处理NaN值"""
        # 检查并处理NaN值（理论上应该已经被处理，但以防万一）
        if np.isnan(data).any():
            data = np.nan_to_num(data, nan=0.0)

        low, high = np.percentile(data, (0.5, 99.5))
        # 防止除以零
        denom = high - low + 1e-8
        result = np.clip((data - low) / denom, 0, 1)
        return result.astype(np.float32)


def get_msi_single_transforms(mode='train', img_size=512):
    """MSI 单支路数据增强"""
    if mode == 'train':
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=0.3),
            ToTensorV2(transpose_mask=True),
        ])
    else:
        return A.Compose([
            ToTensorV2(transpose_mask=True),
        ])


def get_transforms(mode='train', img_size=512):
    # 定义多分支增强目标
    # image/mask 是默认名，oli_image/oli_mask 是额外名
    additional_targets = {
        'oli_image': 'image',
        'oli_mask': 'mask'
    }

    if mode == 'train':
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=0.3),
            # 注意：不再这里做 Normalize，在 Dataset 里做物理量程缩放
            ToTensorV2(transpose_mask=True),
        ], additional_targets=additional_targets)
    else:
        return A.Compose([
            ToTensorV2(transpose_mask=True),
        ], additional_targets=additional_targets)


class MixedS2L8Dataset(Dataset):
    """
    S2+L8 混合单支路数据集
    - 将 S2 (11ch) 和 L8 (7ch→pad到11ch) 合并为统一 11ch 输入
    - 每张图带 sensor_type 标记: 's2' 或 'l8'
    """
    def __init__(self, file_pairs, transform=None, mode='train', target_size=256,
                 filter_empty_labels=False):
        """
        file_pairs: list of (filepath, sensor_type) tuples
        """
        super().__init__()
        if filter_empty_labels:
            filtered = []
            for f, stype in file_pairs:
                try:
                    with rasterio.open(f) as src:
                        label = src.read(src.count)
                        if label.max() > 0:
                            filtered.append((f, stype))
                except Exception:
                    pass
            print(f"标签过滤: {len(file_pairs)} -> {len(filtered)} 文件 (移除全零标签)")
            self.file_pairs = filtered
        else:
            self.file_pairs = file_pairs
        self.transform = transform
        self.mode = mode
        self.target_size = target_size
        self.s2_channels = 11  # S2: 11 data channels + 1 label
        self.l8_channels = 7   # L8: 7 data channels + 1 label
        self.target_channels = 11  # pad to 11

    def _load_data(self, filepath):
        ext = Path(filepath).suffix.lower()
        if ext == '.npy':
            data = np.load(filepath)
        elif ext in ['.tif', '.tiff']:
            with rasterio.open(filepath) as src:
                data = src.read()
        else:
            raise ValueError(f"Unsupported format: {ext}")
        if data.ndim == 3 and data.shape[-1] in [8, 13]:
            data = np.transpose(data, (2, 0, 1))
        return data

    def _resize_to_target(self, data, order=1):
        c, h, w = data.shape
        if h != self.target_size or w != self.target_size:
            from scipy.ndimage import zoom
            zoom_h = self.target_size / h
            zoom_w = self.target_size / w
            data = zoom(data, (1, zoom_h, zoom_w), order=order)
        return data

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        filepath, sensor_type = self.file_pairs[idx]
        raw = self._load_data(filepath)

        # Valid pixel mask (before nan_to_num, for spectral bands)
        valid_mask = np.all(np.isfinite(raw[:-1]), axis=0).astype(np.float32)
        valid_mask = np.expand_dims(valid_mask, axis=0)  # [1, H, W]

        if np.isnan(raw).any():
            raw = np.nan_to_num(raw, nan=0.0)

        # Separate data and label before resize
        img = raw[:-1, :, :].astype(np.float32)     # [C, H, W]
        label = raw[-1:, :, :].astype(np.float32)    # [1, H, W]
        # Data: bilinear, Label/Mask: nearest-neighbor
        img = self._resize_to_target(img, order=1)
        label = self._resize_to_target(label, order=0)
        valid_mask = self._resize_to_target(valid_mask, order=0)

        if sensor_type == 'l8':
            # Pad L8 from 7→11 channels (zero-pad missing bands)
            img = np.pad(img, ((0, self.target_channels - self.l8_channels), (0, 0), (0, 0)),
                         mode='constant', constant_values=0)

        img = self._safe_minmax(img)

        if self.transform:
            combined = np.concatenate([label, valid_mask], axis=0).transpose(1, 2, 0)
            augmented = self.transform(
                image=img.transpose(1, 2, 0),
                mask=combined
            )
            img_tensor = augmented['image']
            combined_out = augmented['mask']
            label_tensor = combined_out[0:1]
            valid_tensor = combined_out[1:2]
        else:
            img_tensor = torch.from_numpy(img)
            label_tensor = torch.from_numpy(label)
            valid_tensor = torch.from_numpy(valid_mask)

        return {
            'msi': img_tensor,
            'msi_seg_gt': label_tensor,
            'sensor_type': sensor_type,
            'msi_valid_mask': valid_tensor,
        }

    def _safe_minmax(self, data):
        if np.isnan(data).any():
            data = np.nan_to_num(data, nan=0.0)
        low, high = np.percentile(data, (0.5, 99.5))
        denom = high - low + 1e-8
        result = np.clip((data - low) / denom, 0, 1)
        return result.astype(np.float32)


def create_mixed_dataloaders(config):
    """
    创建 S2+L8 混合单支路数据加载器
    将 S2 (11ch) 和 L8 (7ch→pad 11ch) 文件合并为统一数据集
    """
    s2_dir = config.get('s2_dir', config['msi_dir'])
    l8_dir = config['l8_dir']

    s2_files = sorted(glob.glob(os.path.join(s2_dir, '*.tif')))
    l8_files = sorted(glob.glob(os.path.join(l8_dir, '*.tif')))

    if len(s2_files) == 0:
        raise ValueError(f"在 {s2_dir} 中没有找到 S2 .tif 文件")
    if len(l8_files) == 0:
        raise ValueError(f"在 {l8_dir} 中没有找到 L8 .tif 文件")

    print(f"混合数据集: S2={len(s2_files)} 文件, L8={len(l8_files)} 文件")

    # Build file_pairs: (filepath, sensor_type)
    s2_pairs = [(f, 's2') for f in s2_files]
    l8_pairs = [(f, 'l8') for f in l8_files]
    all_pairs = s2_pairs + l8_pairs

    total = len(all_pairs)
    print(f"总文件数: {total} (S2: {len(s2_pairs)}, L8: {len(l8_pairs)})")

    # K-fold CV on combined dataset
    k_fold = config.get('k_fold', 10)
    train_folds = config.get('train_folds', 7)
    val_folds = config.get('val_folds', 3)
    fold_idx = config.get('fold_idx', 0)
    seed = config.get('seed', 42)

    indices = np.arange(total)
    np.random.seed(seed)
    np.random.shuffle(indices)

    fold_size = total // k_fold
    remainder = total % k_fold
    fold_boundaries = [0]
    for i in range(k_fold):
        size = fold_size + (1 if i < remainder else 0)
        fold_boundaries.append(fold_boundaries[-1] + size)

    train_indices = []
    val_indices = []
    for i in range(train_folds):
        fold = (fold_idx + i) % k_fold
        train_indices.extend(indices[fold_boundaries[fold]:fold_boundaries[fold+1]])
    for i in range(val_folds):
        fold = (fold_idx + train_folds + i) % k_fold
        val_indices.extend(indices[fold_boundaries[fold]:fold_boundaries[fold+1]])

    print(f"K折CV: {k_fold}折, 训练={train_folds}折, 验证={val_folds}折")
    print(f"训练: {len(train_indices)}, 验证: {len(val_indices)}")

    train_pairs = [all_pairs[i] for i in train_indices]
    val_pairs = [all_pairs[i] for i in val_indices]

    # Count per sensor
    train_s2 = sum(1 for _, t in train_pairs if t == 's2')
    train_l8 = sum(1 for _, t in train_pairs if t == 'l8')
    val_s2 = sum(1 for _, t in val_pairs if t == 's2')
    val_l8 = sum(1 for _, t in val_pairs if t == 'l8')
    print(f"训练集: S2={train_s2}, L8={train_l8}")
    print(f"验证集: S2={val_s2}, L8={val_l8}")

    train_dataset = MixedS2L8Dataset(
        train_pairs,
        transform=get_msi_single_transforms('train', config.get('img_size', 256)),
        mode='train',
        target_size=config.get('img_size', 256),
        filter_empty_labels=config.get('filter_empty_labels', False)
    )
    val_dataset = MixedS2L8Dataset(
        val_pairs,
        transform=get_msi_single_transforms('val', config.get('img_size', 256)),
        mode='val',
        target_size=config.get('img_size', 256),
        filter_empty_labels=config.get('filter_empty_labels', False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=True
    )

    return train_loader, val_loader


def create_msi_single_dataloaders(config):
    """
    创建 MSI 单支路数据加载器（仅 MSI 输入）
    支持单一数据集或 S2+L8 混合训练
    配置参数:
        - k_fold: 总折数（默认 10）
        - train_folds: 用于训练的折数（默认 7）
        - val_folds: 用于验证的折数（默认 3）
        - fold_idx: 当前轮次的起始折索引（默认 0）
        - mixed_data: 是否启用混合训练（默认 False）
        - s2_dir: Sentinel-2 数据目录（混合训练时使用）
        - l8_dir: Landsat-8 数据目录（混合训练时使用）
        - s2_ratio: S2 数据在混合训练中的比例（默认 0.5）
    """
    # 检查是否启用混合训练
    mixed_data = config.get('mixed_data', False)

    if mixed_data:
        return create_mixed_dataloaders(config)

    # 1. 扫描文件系统
    msi_path_list = sorted(glob.glob(os.path.join(config['msi_dir'], "*.tif")))

    if len(msi_path_list) == 0:
        raise ValueError(f"在 {config['msi_dir']} 中没有找到 .tif 文件")

    print(f"找到 {len(msi_path_list)} 个 MSI 文件")

    # 2. K 折交叉验证配置
    k_fold = config.get('k_fold', 10)
    train_folds = config.get('train_folds', 7)
    val_folds = config.get('val_folds', 3)
    fold_idx = config.get('fold_idx', 0)
    seed = config.get('seed', 42)

    total_samples = len(msi_path_list)

    assert train_folds + val_folds <= k_fold, f"训练折数 ({train_folds})+验证折数 ({val_folds}) 不能超过总折数 ({k_fold})"

    # 3. 使用固定的随机种子打乱数据
    indices = np.arange(total_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)

    # 4. 计算每折的样本数
    fold_size = total_samples // k_fold
    remainder = total_samples % k_fold

    # 5. 创建折的边界
    fold_boundaries = [0]
    for i in range(k_fold):
        size = fold_size + (1 if i < remainder else 0)
        fold_boundaries.append(fold_boundaries[-1] + size)

    # 6. 获取训练和验证的索引
    train_indices = []
    val_indices = []

    for i in range(train_folds):
        fold = (fold_idx + i) % k_fold
        start = fold_boundaries[fold]
        end = fold_boundaries[fold + 1]
        train_indices.extend(indices[start:end])

    for i in range(val_folds):
        fold = (fold_idx + train_folds + i) % k_fold
        start = fold_boundaries[fold]
        end = fold_boundaries[fold + 1]
        val_indices.extend(indices[start:end])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    print(f"K 折交叉验证配置:")
    print(f"  总折数：{k_fold}, 训练折数：{train_folds}, 验证折数：{val_folds}")
    print(f"  当前起始折索引：{fold_idx}")
    print(f"  训练样本数：{len(train_indices)}, 验证样本数：{len(val_indices)}")

    # 7. 创建数据集
    train_dataset = MSIDataset(
        msi_files=[msi_path_list[i] for i in train_indices],
        transform=get_msi_single_transforms('train', config.get('img_size', 512)),
        mode='train',
        target_size=config.get('img_size', 512),
        filter_empty_labels=config.get('filter_empty_labels', False)
    )

    val_dataset = MSIDataset(
        msi_files=[msi_path_list[i] for i in val_indices],
        transform=get_msi_single_transforms('val', config.get('img_size', 512)),
        mode='val',
        target_size=config.get('img_size', 512),
        filter_empty_labels=config.get('filter_empty_labels', False)
    )

    # 8. 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 1),
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 1),
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=True
    )

    return train_loader, val_loader

def create_dataloaders(config):
    """
    创建数据加载器，支持K折交叉验证
    配置参数:
        - k_fold: 总折数（默认10）
        - train_folds: 用于训练的折数（默认7）
        - val_folds: 用于验证的折数（默认3）
        - fold_idx: 当前轮次的起始折索引（默认0）
    """
    # 1. 扫描文件系统
    msi_path_list = sorted(glob.glob(os.path.join(config['msi_dir'], "*.tif")))

    if len(msi_path_list) == 0:
        raise ValueError(f"在 {config['msi_dir']} 中没有找到 .tif 文件")

    # 2. 匹配 OLI 文件 - 按索引一一对应（假设文件已按相同顺序排序）
    oli_files = sorted(glob.glob(os.path.join(config['oli_dir'], "*.tif")))

    if len(oli_files) == 0:
        raise ValueError(f"在 {config['oli_dir']} 中没有找到 .tif 文件")

    # 使用最小数量，确保配对
    min_count = min(len(msi_path_list), len(oli_files))
    if len(msi_path_list) != len(oli_files):
        print(f"[WARNING] MSI文件数({len(msi_path_list)})与OLI文件数({len(oli_files)})不匹配，使用较小的数量: {min_count}")

    msi_path_list = msi_path_list[:min_count]
    oli_path_list = oli_files[:min_count]

    print(f"找到 {len(msi_path_list)} 对 MSI-OLI 文件")

    # 3. K折交叉验证配置

    # 3. K折交叉验证配置
    k_fold = config.get('k_fold', 10)  # 总折数
    train_folds = config.get('train_folds', 7)  # 训练折数
    val_folds = config.get('val_folds', 3)  # 验证折数
    fold_idx = config.get('fold_idx', 0)  # 当前起始折索引
    seed = config.get('seed', 42)

    total_samples = len(msi_path_list)

    # 确保折数配置正确
    assert train_folds + val_folds <= k_fold, f"训练折数({train_folds})+验证折数({val_folds})不能超过总折数({k_fold})"

    # 4. 使用固定的随机种子打乱数据
    indices = np.arange(total_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)

    # 5. 计算每折的样本数
    fold_size = total_samples // k_fold
    remainder = total_samples % k_fold

    # 6. 创建折的边界
    fold_boundaries = [0]
    for i in range(k_fold):
        # 将余数分配到前面的折
        size = fold_size + (1 if i < remainder else 0)
        fold_boundaries.append(fold_boundaries[-1] + size)

    # 7. 获取训练和验证的索引
    # 使用轮转的fold_idx来确定当前轮次的训练和验证折
    train_indices = []
    val_indices = []

    for i in range(train_folds):
        fold = (fold_idx + i) % k_fold
        start = fold_boundaries[fold]
        end = fold_boundaries[fold + 1]
        train_indices.extend(indices[start:end])

    for i in range(val_folds):
        fold = (fold_idx + train_folds + i) % k_fold
        start = fold_boundaries[fold]
        end = fold_boundaries[fold + 1]
        val_indices.extend(indices[start:end])

    # 去重并转为numpy数组
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    print(f"K折交叉验证配置:")
    print(f"  总折数: {k_fold}, 训练折数: {train_folds}, 验证折数: {val_folds}")
    print(f"  当前起始折索引: {fold_idx}")
    print(f"  训练样本数: {len(train_indices)}, 验证样本数: {len(val_indices)}")
    print(f"  训练折: {[(fold_idx + i) % k_fold for i in range(train_folds)]}")
    print(f"  验证折: {[(fold_idx + train_folds + i) % k_fold for i in range(val_folds)]}")

    # 8. 创建独立的训练集实例
    train_dataset = MSIOLIDataset(
        msi_files=[msi_path_list[i] for i in train_indices],
        oli_files=[oli_path_list[i] for i in train_indices],
        transform=get_transforms('train', config.get('img_size', 512)),
        mode='train'
    )

    # 9. 创建独立的验证集实例
    val_dataset = MSIOLIDataset(
        msi_files=[msi_path_list[i] for i in val_indices],
        oli_files=[oli_path_list[i] for i in val_indices],
        transform=get_transforms('val', config.get('img_size', 512)),
        mode='val'
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers'], pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers'], pin_memory=True
    )

    return train_loader, val_loader


def create_kfold_dataloaders(config, fold_idx=0):
    """
    创建指定折的数据加载器
    用于多轮交叉验证训练

    参数:
        config: 配置字典
        fold_idx: 当前轮次的起始折索引

    返回:
        train_loader, val_loader
    """
    config['fold_idx'] = fold_idx
    return create_dataloaders(config)


def get_kfold_training_config(base_config, num_rounds=None):
    """
    生成K折训练的配置列表

    参数:
        base_config: 基础配置
        num_rounds: 训练轮数，默认为 k_fold // (train_folds + val_folds) 的整数倍

    返回:
        配置列表，每个配置对应一轮训练
    """
    k_fold = base_config.get('k_fold', 10)
    train_folds = base_config.get('train_folds', 7)
    val_folds = base_config.get('val_folds', 3)

    # 计算可以进行的完整轮数
    step = train_folds + val_folds
    max_rounds = k_fold // step if step > 0 else 1

    if num_rounds is None:
        num_rounds = max_rounds
    else:
        num_rounds = min(num_rounds, max_rounds)

    configs = []
    for i in range(num_rounds):
        fold_idx = (i * step) % k_fold
        cfg = base_config.copy()
        cfg['fold_idx'] = fold_idx
        cfg['round'] = i + 1
        cfg['total_rounds'] = num_rounds
        configs.append(cfg)

    return configs
