"""
单传感器基线模型训练脚本
每个模型分别在 S2 (11ch) 和 L8 (7ch) 上独立训练，使用统一配置。

支持的模型:
  unet, unetpp, attention_unet, segnet, deeplabv3plus, segformer,
  swin_unet, algae_net, algae_mamba

用法:
  python -m running.train_single_sensor_baselines --model unet --sensor s2
  python -m running.train_single_sensor_baselines --model unet --sensor l8
"""
import os, sys, glob, argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'segmentation_models'))

from running.data_loader import MSIDataset, get_transforms


# ============================================================
# Loss functions
# ============================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return 1 - (2 * intersection + self.smooth) / (union + self.smooth)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=10):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
    def forward(self, pred, target):
        target = target.clamp(0, 1)
        pt = torch.where(target > 0.5, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        bce = -torch.log(pt.clamp(1e-6, 1 - 1e-6))
        weight = torch.where(target > 0.5, self.pos_weight, 1.0)
        return (weight * focal_weight * bce).mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha; self.beta = beta; self.smooth = smooth
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        tp = (pred * target).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()
        return 1 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)


# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def compute_metrics(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    tp = (pred_bin * target_bin).sum()
    fp = (pred_bin * (1 - target_bin)).sum()
    fn = ((1 - pred_bin) * target_bin).sum()
    tn = ((1 - pred_bin) * (1 - target_bin)).sum()
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    specificity = (tn + 1e-6) / (tn + fp + 1e-6)
    return iou.item(), f1.item(), precision.item(), recall.item(), specificity.item()


# ============================================================
# Model builders
# ============================================================

def build_model(model_name, in_channels, img_size=256):
    """构建模型，支持 segmentation_models + algae_net + algae_mamba"""
    device = torch.device('cuda')

    if model_name == 'algae_net':
        from running.algae_net_model import ResNet18_UNet
        return ResNet18_UNet(in_channels=in_channels, out_channels=1).to(device)

    elif model_name == 'algae_mamba':
        from running.algae_mamba_model import SegMamba2D
        return SegMamba2D(in_chans=in_channels, out_chans=1).to(device)

    elif model_name == 'vanilla_unet':
        from models.unet import UNet
        return UNet(in_channels=in_channels, num_classes=1, base_channels=64, bilinear=True).to(device)

    elif model_name == 'unetpp':
        from models.unet import NestedUNet
        return NestedUNet(in_channels=in_channels, num_classes=1, base_channels=32).to(device)

    elif model_name == 'attention_unet':
        from models.unet import AttentionUNet
        return AttentionUNet(in_channels=in_channels, num_classes=1, base_channels=64, bilinear=True).to(device)

    elif model_name == 'segnet':
        from models.segnet import SegNet
        return SegNet(in_channels=in_channels, num_classes=1).to(device)

    elif model_name == 'deeplabv3plus':
        from models.deeplabv3plus import DeepLabV3Plus
        return DeepLabV3Plus(in_channels=in_channels, num_classes=1, backbone='resnet50', output_stride=16).to(device)

    elif model_name == 'segformer':
        from models.segformer import SegFormer
        return SegFormer(in_channels=in_channels, num_classes=1, img_size=img_size).to(device)

    elif model_name == 'swin_unet':
        from models.swin_unet import SwinUperNet
        return SwinUperNet(in_channels=in_channels, num_classes=1, img_size=img_size,
                           embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# Training
# ============================================================

def train(config):
    device = torch.device('cuda')
    exp_id = f"Baseline-{config['model']}-{config['sensor'].upper()}"

    # ---- Data ----
    sensor = config['sensor']
    if sensor == 's2':
        data_dir = config.get('s2_dir', 'dataset3.0/s2_filtered_3pct')
        in_channels = 11
    else:
        data_dir = config.get('l8_dir', 'dataset3.0/l8_algae_256_filtered')
        in_channels = 7

    all_files = sorted(glob.glob(os.path.join(data_dir, '*.tif')))
    print(f"[{exp_id}] {sensor.upper()} files: {len(all_files)}, channels: {in_channels}")

    # 10-fold CV: 7 train, 3 val
    np.random.seed(42)
    indices = np.random.permutation(len(all_files))
    split = int(len(all_files) * 0.7)
    train_files = [all_files[i] for i in indices[:split]]
    val_files = [all_files[i] for i in indices[split:]]
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    train_dataset = MSIDataset(train_files, transform=get_transforms('train', config.get('img_size', 256)),
                               mode='train', target_size=config.get('img_size', 256))
    val_dataset = MSIDataset(val_files, transform=get_transforms('val', config.get('img_size', 256)),
                             mode='val', target_size=config.get('img_size', 256))

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config.get('num_workers', 0), pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config.get('num_workers', 0), pin_memory=True)

    # ---- Model ----
    model = build_model(config['model'], in_channels, config.get('img_size', 256))
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {config['model']}, Params: {total_params:.2f}M")

    # ---- Optimizer ----
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10, min_lr=1e-6)

    # ---- Loss ----
    dice_loss = DiceLoss()
    focal_loss = FocalLoss(pos_weight=10)
    tversky_loss = TverskyLoss()

    # ---- Checkpoint dir ----
    save_dir = Path(config['save_dir']) / exp_id
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Train loop ----
    best_iou = 0
    best_epoch = 0
    print(f"Epochs: {config['epochs']}, LR: {config['lr']}, Batch: {config['batch_size']}")

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            img = batch['msi'].to(device)
            label = batch['msi_seg_gt'].to(device)

            optimizer.zero_grad()
            pred = model(img)
            if isinstance(pred, dict):
                pred = pred['out']
            pred = torch.sigmoid(pred)

            loss = 0.5 * dice_loss(pred, label) + 1.0 * focal_loss(pred, label) + 0.5 * tversky_loss(pred, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        val_iou = 0; val_f1 = 0; val_precision = 0; val_recall = 0; val_spec = 0; count = 0
        for batch in val_loader:
            img = batch['msi'].to(device)
            label = batch['msi_seg_gt'].to(device)

            with torch.no_grad():
                pred = model(img)
                if isinstance(pred, dict):
                    pred = pred['out']
                pred = torch.sigmoid(pred)

            for b in range(pred.shape[0]):
                iou, f1, prec, rec, spec = compute_metrics(pred[b:b+1], label[b:b+1])
                val_iou += iou; val_f1 += f1; val_precision += prec; val_recall += rec; val_spec += spec
                count += 1

        val_iou /= count; val_f1 /= count

        # ReduceLROnPlateau monitors (1 - IoU) as a minimization proxy
        scheduler.step(1.0 - val_iou)

        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'best_iou': best_iou,
                'config': config,
            }, save_dir / 'best.pth')

        print(f"Epoch {epoch+1:3d}/{config['epochs']} | Loss: {avg_loss:.4f} | "
              f"Val IoU: {val_iou:.4f} | Val F1: {val_f1:.4f} | Best: {best_iou:.4f} @ {best_epoch}")

    print(f"\n[{exp_id}] Done. Best IoU: {best_iou:.4f} @ epoch {best_epoch}")

    # Save results
    result = {
        'exp_id': exp_id, 'model': config['model'], 'sensor': sensor,
        'in_channels': in_channels, 'params_M': total_params,
        'best_iou': best_iou, 'best_f1': val_f1,
        'best_precision': val_precision, 'best_recall': val_recall,
        'best_specificity': val_spec, 'best_epoch': best_epoch,
    }
    import json
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(result, f, indent=2)
    return result


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        choices=['vanilla_unet', 'unetpp', 'attention_unet', 'segnet',
                                 'deeplabv3plus', 'segformer', 'swin_unet',
                                 'algae_net', 'algae_mamba'])
    parser.add_argument('--sensor', required=True, choices=['s2', 'l8'])
    parser.add_argument('--s2_dir', default='dataset3.0/s2_filtered_3pct')
    parser.add_argument('--l8_dir', default='dataset3.0/l8_algae_256_filtered')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--save_dir', default='checkpoints/experiments')
    args = parser.parse_args()

    config = vars(args)
    train(config)
