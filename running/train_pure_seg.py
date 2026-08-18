"""
纯分割训练脚本 - 无 SFT、无重构分支
支持：MSI 单支路分割
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import sys
import time
import json
import math
import logging
import traceback
from collections import defaultdict
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 导入项目模块
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from running.msi_single_unet import MSISingleUNet
from running.routed_unet import RoutedSingleUNet
from running.multi_sensor_baselines import build_multisensor_model
from running.data_loader import create_msi_single_dataloaders as create_dataloaders

# 邮件通知
try:
    from running.email_notifier import EmailNotifier
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False


# ============ 配置日志 ============
def setup_logger(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f'train_{time.strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


# ============ 损失函数 ============
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=10.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
        bce = -self.pos_weight * target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        p_t = (target * pred) + ((1 - target) * (1 - pred))
        focal_weight = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_weight * bce
        return loss.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=(2, 3))
        fp = (pred * (1 - target)).sum(dim=(2, 3))
        fn = ((1 - pred) * target).sum(dim=(2, 3))
        tversky = (intersection + self.smooth) / (intersection + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - tversky.mean()


class CombinedLoss(nn.Module):
    """分割损失 + 可选形态学损失"""
    def __init__(self, device='cuda', pos_weight=10.0, use_morph=False,
                 use_circularity=False, morph_weight=0.01):
        super().__init__()
        self.device = device
        self.use_morph = use_morph
        self.morph_weight = morph_weight

        # 分割损失
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=0.25, pos_weight=pos_weight)
        self.tversky_loss = TverskyLoss(alpha=0.3, beta=0.7)

        # 形态学损失（可选）
        if use_morph:
            from loss.morphological import OptimizedMorphologicalLoss
            self.morph_loss = OptimizedMorphologicalLoss(
                device_type=device, use_circularity=use_circularity
            )

        # 权重配置
        self.weights = {
            'seg_dice': 0.5,
            'seg_focal': 1.0,
            'seg_tversky': 0.5,
        }

    def forward(self, outputs, targets):
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=outputs['msi_seg'].device)

        pred_seg = outputs['msi_seg']
        target_seg = targets['msi_seg_gt']

        if pred_seg.shape[2:] != target_seg.shape[2:]:
            pred_seg = F.interpolate(pred_seg, size=target_seg.shape[2:], mode='bilinear', align_corners=False)

        loss_dice = self.dice_loss(pred_seg, target_seg)
        loss_focal = self.focal_loss(pred_seg, target_seg)
        loss_tversky = self.tversky_loss(pred_seg, target_seg)

        loss_dict['seg_dice'] = loss_dice.item()
        loss_dict['seg_focal'] = loss_focal.item()
        loss_dict['seg_tversky'] = loss_tversky.item()

        total_loss += (self.weights['seg_dice'] * loss_dice +
                      self.weights['seg_focal'] * loss_focal +
                      self.weights['seg_tversky'] * loss_tversky)

        # 形态学损失（低权重，仅做形状微调）
        if self.use_morph:
            loss_morph = self.morph_loss(pred_seg, target_seg)
            total_loss += self.morph_weight * loss_morph
            loss_dict['morph'] = loss_morph.item()

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


# ============ 训练器 ============
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logger(config['log_dir'])
        self.logger.info(f"Using device: {self.device}")
        self.mixed_data = config.get('mixed_data', False)

        # 模型：混合数据用路由模型或新基线，单传感器用 MSISingleUNet
        self._model_type = config.get('multi_sensor_model', 'routed')
        if self.mixed_data:
            if self._model_type == 'routed':
                self.model = RoutedSingleUNet(
                    msi_in_channels=config.get('msi_channels', 11),
                    oli_in_channels=config.get('oli_channels', 7),
                    hidden_dim=config.get('hidden_dim', 1024),
                    bottleneck_size=config.get('bottleneck_size', 16),
                    num_mamba_layers=config.get('num_mamba_layers', 2),
                    num_topk_layers=config.get('num_topk_layers', 2),
                    eca_k_size=config.get('eca_k_size', 3),
                    eca_topk_msi=config.get('eca_topk_msi', 6),
                    eca_topk_oli=config.get('eca_topk_oli', 4),
                    mamba_dim=config.get('mamba_dim', 64),
                    topk_rate=config.get('topk_rate', 0.4),
                ).to(self.device)
            else:
                self.model = build_multisensor_model(
                    self._model_type,
                    hidden_dim=config.get('hidden_dim', 512),
                    num_mamba_layers=config.get('num_mamba_layers', 2),
                    num_topk_layers=config.get('num_topk_layers', 2),
                    mamba_dim=config.get('mamba_dim', 64),
                    topk_rate=config.get('topk_rate', 0.4),
                    adapter_dim=config.get('adapter_dim', 64),
                    shared_pixel_unshuffle=config.get('use_pixel_unshuffle', False),
                ).to(self.device)
        else:
            self.model = MSISingleUNet(
                msi_in_channels=config.get('msi_channels', 11),
                hidden_dim=config.get('hidden_dim', 256),
                bottleneck_size=config.get('bottleneck_size', 16),
                num_mamba_layers=config.get('num_mamba_layers', 1),
                num_topk_layers=config.get('num_topk_layers', 2),
                eca_k_size=config.get('eca_k_size', 3),
                eca_topk_msi=config.get('eca_topk_msi', 6),
                mamba_dim=config.get('mamba_dim', 32),
                topk_rate=config.get('topk_rate', 0.4),
                use_pixel_unshuffle=config.get('use_pixel_unshuffle', True)
            ).to(self.device)

        # 优化器
        base_lr = config['lr']
        encoder_lr_mult = config.get('encoder_lr_mult', 1.0)

        if self._model_type != 'routed':
            # 新基线：共享编码器，使用 encoder_parameters() 方法
            encoder_params = list(self.model.encoder_parameters())
            encoder_prefixes = ('shared_encoder', 's2_adapter', 'l8_adapter', 'spectral_encoder')
        elif self.mixed_data:
            encoder_params = (list(self.model.msi_encoder.parameters()) +
                             list(self.model.oli_encoder.parameters()))
            encoder_prefixes = ('msi_encoder', 'oli_encoder')
        else:
            encoder_params = list(self.model.msi_encoder.parameters())
            encoder_prefixes = ('msi_encoder',)
        other_params = [p for n, p in self.model.named_parameters()
                       if not any(n.startswith(p) for p in encoder_prefixes)]

        self.optimizer = optim.AdamW([
            {'params': other_params, 'lr': base_lr, 'name': 'decoder_head'},
            {'params': encoder_params, 'lr': base_lr * encoder_lr_mult, 'name': 'encoder'}
        ], weight_decay=config.get('weight_decay', 0.01))

        # torch.compile 优化 (必须在优化器之后)
        if config.get('use_compile', True):
            self.logger.info("Applying torch.compile to model...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
            self.logger.info("torch.compile applied successfully")

        # 学习率调度器
        self.scheduler = self._create_scheduler(self.optimizer)

        # 损失函数
        self.criterion = CombinedLoss(
            device=str(self.device), pos_weight=10.0,
            use_morph=config.get('use_morph', False),
            use_circularity=config.get('use_circularity', False),
            morph_weight=config.get('morph_weight', 0.01)
        )

        # 早停
        self.patience = config.get('patience', 15)
        self.early_stop_delta = config.get('early_stop_delta', 1e-6)
        self.epochs_no_improve = 0
        self.early_stop_triggered = False

        # 历史记录
        self.loss_history = {
            'train': defaultdict(list),
            'val': defaultdict(list),
            'learning_rate': [],
            'learning_rate_encoder': []
        }

        self.epoch = 0
        self.best_loss = float('inf')

        # 邮件通知
        self.email_notifier = None
        if config.get('use_email', False) and EMAIL_AVAILABLE:
            email_config_path = config.get('email_config_path', 'email_config.json')
            if Path(email_config_path).exists():
                self.email_notifier = EmailNotifier(config_path=email_config_path)

        # 创建目录
        Path(config['log_dir']).mkdir(parents=True, exist_ok=True)
        Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)

        self.plot_dir = Path(config['log_dir']) / 'plots'
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # 验证集评价指标
        self.val_metrics = {
            'best_iou': 0.0,
            'best_precision': 0.0,
            'best_recall': 0.0,
            'best_f1': 0.0,
        }
        if self.mixed_data:
            self.val_metrics.update({
                'best_iou_s2': 0.0, 'best_iou_l8': 0.0,
                'best_f1_s2': 0.0, 'best_f1_l8': 0.0,
            })

    def _create_scheduler(self, optimizer):
        if self.config.get('use_reduce_lr_on_plateau', False):
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min',
                factor=self.config.get('reduce_lr_factor', 0.7),
                patience=self.config.get('reduce_lr_patience', 10),
                min_lr=1e-6
            )
        else:
            total_steps = self.config['epochs'] * self.config.get('steps_per_epoch', 100)
            warmup_steps = self.config.get('warmup_steps', 500)

            def make_lr_lambda(group_idx):
                def lr_lambda(step):
                    if step < warmup_steps:
                        return step / warmup_steps
                    else:
                        progress = (step - warmup_steps) / (total_steps - warmup_steps)
                        return 0.5 * (1 + math.cos(math.pi * progress))
                return lr_lambda

            lr_lambdas = [make_lr_lambda(i) for i in range(len(optimizer.param_groups))]
            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas)

    def train_epoch(self, dataloader, steps_per_epoch=None):
        """
        训练一个 epoch

        参数:
            dataloader: 训练数据加载器
            steps_per_epoch: 可选，限制每个 epoch 的迭代步数（用于加速训练）
        """
        self.model.train()
        epoch_losses = defaultdict(list)

        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        steps_completed = 0

        for batch in pbar:
            # 如果设置了 steps_per_epoch 且已达到限制，提前结束
            if steps_per_epoch is not None and steps_completed >= steps_per_epoch:
                break

            msi = batch['msi'].to(self.device)

            # 前向传播 (混合模式传入 sensor_types)
            self.optimizer.zero_grad()
            if self.mixed_data and 'sensor_type' in batch:
                outputs = self.model(msi, batch['sensor_type'])
            else:
                outputs = self.model(msi)

            # 计算损失
            targets = {k: v.to(self.device) for k, v in batch.items() if k not in ('msi', 'sensor_type')}
            loss, loss_dict = self.criterion(outputs, targets)

            # 反向传播
            if loss.requires_grad:
                loss.backward()
                self.optimizer.step()

                if not self.config.get('use_reduce_lr_on_plateau', False):
                    self.scheduler.step()

            # 记录损失
            for k, v in loss_dict.items():
                epoch_losses[k].append(v)

            decoder_lr = self.optimizer.param_groups[0]['lr']
            encoder_lr = self.optimizer.param_groups[1]['lr']

            pbar.set_postfix({
                'Loss': f"{loss_dict.get('total', 0):.4f}",
                'lr_dec': f"{decoder_lr:.6f}",
                'lr_enc': f"{encoder_lr:.6f}"
            })

            steps_completed += 1

        return {k: np.mean(v) for k, v in epoch_losses.items()}

    def _calculate_metrics(self, preds, targets):
        """计算分割评价指标：IoU, Precision, Recall, F1-Score"""
        pred_binary = (preds > 0.5).float()
        target_binary = targets

        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection

        iou = (intersection + 1e-6) / (union + 1e-6)
        precision = (intersection + 1e-6) / (pred_binary.sum() + 1e-6)
        recall = (intersection + 1e-6) / (target_binary.sum() + 1e-6)
        f1 = (2 * precision * recall + 1e-6) / (precision + recall + 1e-6)

        return iou.item(), precision.item(), recall.item(), f1.item()

    def validate(self, dataloader, steps_per_epoch=None):
        """
        验证一个 epoch。混合模式时按 sensor_type 分别统计 S2/L8 指标。
        """
        self.model.eval()
        epoch_losses = defaultdict(list)
        metrics = {'iou': [], 'precision': [], 'recall': [], 'f1': []}
        if self.mixed_data:
            metrics.update({'iou_s2': [], 'iou_l8': [], 'f1_s2': [], 'f1_l8': []})

        with torch.no_grad():
            steps_completed = 0
            for batch in tqdm(dataloader, desc="Validation"):
                if steps_per_epoch is not None and steps_completed >= steps_per_epoch:
                    break

                msi = batch['msi'].to(self.device)

                if self.mixed_data and 'sensor_type' in batch:
                    outputs = self.model(msi, batch['sensor_type'])
                else:
                    outputs = self.model(msi)

                targets = {k: v.to(self.device) for k, v in batch.items() if k != 'msi' and k != 'sensor_type'}
                loss, loss_dict = self.criterion(outputs, targets)

                for k, v in loss_dict.items():
                    epoch_losses[k].append(v)

                # 计算评价指标
                pred_seg = outputs['msi_seg'].float()
                target_seg = targets['msi_seg_gt']
                iou, precision, recall, f1 = self._calculate_metrics(pred_seg, target_seg)
                metrics['iou'].append(iou)
                metrics['precision'].append(precision)
                metrics['recall'].append(recall)
                metrics['f1'].append(f1)

                # 混合模式: 按传感器分别统计
                if self.mixed_data and 'sensor_type' in batch:
                    sensor_types = batch['sensor_type']
                    for i, st in enumerate(sensor_types):
                        single_iou, _, _, single_f1 = self._calculate_metrics(
                            pred_seg[i:i+1], target_seg[i:i+1])
                        if st == 's2':
                            metrics['iou_s2'].append(single_iou)
                            metrics['f1_s2'].append(single_f1)
                        else:
                            metrics['iou_l8'].append(single_iou)
                            metrics['f1_l8'].append(single_f1)

                steps_completed += 1

        result = {k: np.mean(v) for k, v in epoch_losses.items()}
        result['val_iou'] = np.mean(metrics['iou'])
        result['val_precision'] = np.mean(metrics['precision'])
        result['val_recall'] = np.mean(metrics['recall'])
        result['val_f1'] = np.mean(metrics['f1'])
        if self.mixed_data:
            result['val_iou_s2'] = np.mean(metrics['iou_s2']) if metrics['iou_s2'] else 0
            result['val_iou_l8'] = np.mean(metrics['iou_l8']) if metrics['iou_l8'] else 0
            result['val_f1_s2'] = np.mean(metrics['f1_s2']) if metrics['f1_s2'] else 0
            result['val_f1_l8'] = np.mean(metrics['f1_l8']) if metrics['f1_l8'] else 0
        return result

    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }

        latest_path = Path(self.config['checkpoint_dir']) / 'latest.pth'
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = Path(self.config['checkpoint_dir']) / 'best.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at epoch {self.epoch}")

    def train(self, train_loader, val_loader):
        start_time_total = time.time()

        # 训练开始通知
        if self.email_notifier:
            try:
                self.email_notifier.send_training_start(self.config)
            except Exception as e:
                self.logger.error(f"发送训练开始邮件失败：{e}")

        email_interval = self.config.get('email_interval', 10)
        # 获取每个 epoch 的最大步数（用于加速训练）
        steps_per_epoch = self.config.get('steps_per_epoch', None)
        val_steps = self.config.get('val_steps_per_epoch', steps_per_epoch)

        try:
            for epoch in range(self.epoch, self.config['epochs']):
                self.epoch = epoch
                start_time = time.time()

                # 更新 ECA 层的噪声退火
                if hasattr(self.model, 'set_epoch'):
                    self.model.set_epoch(epoch, self.config['epochs'])

                train_losses = self.train_epoch(train_loader, steps_per_epoch=steps_per_epoch)
                val_losses = self.validate(val_loader, steps_per_epoch=val_steps)

                elapsed = time.time() - start_time

                self.logger.info(
                    f"Epoch {epoch}/{self.config['epochs']} - "
                    f"Train: {train_losses.get('total', 0):.4f}, "
                    f"Val: {val_losses.get('total', 0):.4f}, "
                    f"Time: {elapsed:.1f}s"
                )

                # 打印验证集评价指标
                if self.mixed_data:
                    self.logger.info(
                        f"  Val Metrics - IoU: {val_losses.get('val_iou', 0):.4f} "
                        f"(S2: {val_losses.get('val_iou_s2', 0):.4f}, L8: {val_losses.get('val_iou_l8', 0):.4f}), "
                        f"F1: {val_losses.get('val_f1', 0):.4f} "
                        f"(S2: {val_losses.get('val_f1_s2', 0):.4f}, L8: {val_losses.get('val_f1_l8', 0):.4f})"
                    )
                else:
                    self.logger.info(
                        f"  Val Metrics - IoU: {val_losses.get('val_iou', 0):.4f}, "
                        f"Precision: {val_losses.get('val_precision', 0):.4f}, "
                        f"Recall: {val_losses.get('val_recall', 0):.4f}, "
                        f"F1: {val_losses.get('val_f1', 0):.4f}"
                    )

                is_best = val_losses.get('total', float('inf')) < (self.best_loss - self.early_stop_delta)
                if is_best:
                    self.best_loss = val_losses['total']
                    self.epochs_no_improve = 0
                    self.logger.info(f"  New best: {self.best_loss:.6f}")
                    self.val_metrics['best_iou'] = val_losses.get('val_iou', 0)
                    self.val_metrics['best_precision'] = val_losses.get('val_precision', 0)
                    self.val_metrics['best_recall'] = val_losses.get('val_recall', 0)
                    self.val_metrics['best_f1'] = val_losses.get('val_f1', 0)
                    if self.mixed_data:
                        self.val_metrics['best_iou_s2'] = val_losses.get('val_iou_s2', 0)
                        self.val_metrics['best_iou_l8'] = val_losses.get('val_iou_l8', 0)
                        self.val_metrics['best_f1_s2'] = val_losses.get('val_f1_s2', 0)
                        self.val_metrics['best_f1_l8'] = val_losses.get('val_f1_l8', 0)
                    self.logger.info(
                        f"  Best Metrics - IoU: {self.val_metrics['best_iou']:.4f}, "
                        f"Precision: {self.val_metrics['best_precision']:.4f}, "
                        f"Recall: {self.val_metrics['best_recall']:.4f}, "
                        f"F1: {self.val_metrics['best_f1']:.4f}"
                    )
                    if self.mixed_data:
                        self.logger.info(
                            f"  Best per-sensor - S2 IoU: {self.val_metrics['best_iou_s2']:.4f}, "
                            f"L8 IoU: {self.val_metrics['best_iou_l8']:.4f}"
                        )
                else:
                    self.epochs_no_improve += 1

                # 早停检查
                if self.epochs_no_improve >= self.patience:
                    self.logger.warning(f"Early stopping after {self.epoch} epochs")
                    self.early_stop_triggered = True
                    break

                # 学习率调度
                if self.config.get('use_reduce_lr_on_plateau', False):
                    self.scheduler.step(val_losses.get('total', 0))

                self.save_checkpoint(is_best)

                # 记录历史
                for k, v in train_losses.items():
                    self.loss_history['train'][k].append(v)
                for k, v in val_losses.items():
                    self.loss_history['val'][k].append(v)

                self.loss_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
                self.loss_history['learning_rate_encoder'].append(self.optimizer.param_groups[1]['lr'])

                if self.epochs_no_improve > 0:
                    self.logger.info(f"Epochs without improvement: {self.epochs_no_improve}/{self.patience}")

        except Exception as e:
            if self.email_notifier:
                try:
                    error_msg = traceback.format_exc()
                    self.email_notifier.send_error_notification(error_msg)
                except:
                    pass
            raise

        # 训练完成
        total_training_time = time.time() - start_time_total
        stop_reason = "Early Stopping" if self.early_stop_triggered else "Completed"
        self.logger.info(f"Training finished: {stop_reason}")
        self.logger.info(f"Total training time: {total_training_time/60:.1f} minutes")

        # 保存训练曲线
        self.save_training_curves()

    def save_training_curves(self):
        """保存训练曲线图"""
        import warnings
        warnings.filterwarnings('ignore')

        epochs = range(1, len(self.loss_history['train']['total']) + 1)

        # 1. 损失曲线
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Train & Val Loss
        ax = axes[0, 0]
        train_loss = self.loss_history['train']['total']
        val_loss = self.loss_history['val']['total']
        ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # IoU
        ax = axes[0, 1]
        val_iou = self.loss_history['val'].get('val_iou', [])
        if val_iou:
            ax.plot(epochs, val_iou, 'g-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('IoU')
            ax.set_title('Validation IoU')
            ax.grid(True, alpha=0.3)

        # F1 Score
        ax = axes[1, 0]
        val_f1 = self.loss_history['val'].get('val_f1', [])
        if val_f1:
            ax.plot(epochs, val_f1, 'm-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('F1 Score')
            ax.set_title('Validation F1 Score')
            ax.grid(True, alpha=0.3)

        # Learning Rate
        ax = axes[1, 1]
        lr = self.loss_history['learning_rate']
        if lr:
            ax.plot(epochs, lr, 'c-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.plot_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"Training curves saved to: {save_path}")

        # 2. 单独的损失曲线大图
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = self.plot_dir / 'loss_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"Loss curves saved to: {save_path}")

        # 3.  metrics 曲线
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        metrics = ['val_iou', 'val_precision', 'val_recall', 'val_f1']
        titles = ['IoU', 'Precision', 'Recall', 'F1 Score']
        colors = ['g', 'b', 'r', 'm']

        for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
            if idx < len(axes):
                ax = axes[idx]
                values = self.loss_history['val'].get(metric, [])
                if values:
                    ax.plot(epochs, values, f'{color}-', linewidth=2)
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel(title)
                    ax.set_title(f'Validation {title}')
                    ax.grid(True, alpha=0.3)
                    # 标注最佳值
                    best_val = max(values)
                    best_epoch = epochs[values.index(best_val)]
                    ax.annotate(f'Best: {best_val:.4f}\\nEpoch {best_epoch}',
                               xy=(best_epoch, best_val),
                               xytext=(best_epoch * 0.7, best_val * 0.85),
                               arrowprops=dict(arrowstyle='->', color='black'),
                               fontsize=10,
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

        plt.tight_layout()
        save_path = self.plot_dir / 'metrics_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"Metrics curves saved to: {save_path}")


# ============ 主函数 ============
def train_single_fold(config, fold_idx=0):
    """训练单折模型"""
    config['fold_idx'] = fold_idx
    config['log_dir'] = f"logs/experiments/{config['exp_id']}"
    config['checkpoint_dir'] = f"checkpoints/experiments/{config['exp_id']}"

    train_loader, val_loader = create_dataloaders(config)
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)

    return trainer.best_loss


def main():
    """主训练函数"""
    import argparse

    parser = argparse.ArgumentParser(description='纯分割训练脚本')
    parser.add_argument('--exp_id', type=str, default='Exp-1A', help='实验 ID')
    parser.add_argument('--msi_dir', type=str, default=None, help='MSI 数据目录')
    parser.add_argument('--msi_channels', type=int, default=11, help='MSI 通道数')
    parser.add_argument('--oli_channels', type=int, default=7, help='OLI 通道数')
    parser.add_argument('--eca_topk_msi', type=int, default=6, help='ECA TopK 通道数 (S2)')
    parser.add_argument('--eca_topk_oli', type=int, default=4, help='ECA TopK 通道数 (L8)')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--steps_per_epoch', type=int, default=None, help='每个 epoch 的最大训练步数（用于加速训练）')
    parser.add_argument('--val_steps', type=int, default=None, help='每个 epoch 的最大验证步数')
    parser.add_argument('--val_during_training', action='store_true', help='在训练过程中每个 epoch 后进行验证（默认只在训练结束后验证）')
    parser.add_argument('--num_mamba_layers', type=int, default=1, help='Mamba 层数')
    parser.add_argument('--mamba_dim', type=int, default=None, help='Mamba 维度')
    parser.add_argument('--topk_rate', type=float, default=0.4, help='TopK Rate (注意力中保留的 token 比例)')
    parser.add_argument('--use_morph', action='store_true', help='启用形态学损失')
    parser.add_argument('--use_circularity', action='store_true', help='使用圆度约束（默认紧凑度）')
    parser.add_argument('--morph_weight', type=float, default=0.01, help='形态学损失权重')
    parser.add_argument('--use_pixel_unshuffle', action='store_true', default=True, help='使用 PixelUnshuffle 下采样')
    parser.add_argument('--no_pixel_unshuffle', action='store_true', help='使用 Strided Conv 下采样')
    parser.add_argument('--filter_empty_labels', action='store_true', help='过滤全零标签文件')
    parser.add_argument('--mixed_data', action='store_true', help='启用 S2+L8 混合训练')
    parser.add_argument('--multi_sensor_model', type=str, default='routed',
                        choices=['routed', 'common_shared', 'sensor_adapter', 'spectral_set'],
                        help='多传感器模型类型 (routed=RoutedSingleUNet, 其余为新基线)')
    parser.add_argument('--adapter_dim', type=int, default=64,
                        help='SensorAdapter / SpectralSet 的适配器维度 (默认 64)')
    parser.add_argument('--s2_dir', type=str, default=None, help='S2 数据目录（混合模式）')
    parser.add_argument('--l8_dir', type=str, default=None, help='L8 数据目录（混合模式）')

    args = parser.parse_args()

    # 默认配置
    config = {
        'exp_id': args.exp_id,
        'msi_dir': args.msi_dir or str(PROJECT_ROOT / 'dataset3.0' / 's2_filtered_3pct'),
        'msi_channels': args.msi_channels,
        'oli_channels': args.oli_channels,
        'eca_topk_msi': args.eca_topk_msi,
        'eca_topk_oli': args.eca_topk_oli,
        'hidden_dim': args.hidden_dim,
        'num_mamba_layers': args.num_mamba_layers,
        'num_topk_layers': 2,
        'eca_k_size': 3,
        'mamba_dim': args.mamba_dim or (64 if args.mixed_data else 32),
        'img_size': 256,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'encoder_lr_mult': 1.0,
        'weight_decay': 0.01,
        'warmup_steps': 500,
        'use_reduce_lr_on_plateau': True,
        'reduce_lr_patience': 10,
        'reduce_lr_factor': 0.7,
        'patience': 15,
        'early_stop_delta': 1e-6,
        'k_fold': 10,
        'train_folds': 7,
        'val_folds': 3,
        'fold_idx': 0,
        'seed': 42,
        'use_email': False,
        'val_split': 0.2,
        'num_workers': 0,
        'steps_per_epoch': args.steps_per_epoch,
        'val_steps_per_epoch': args.val_steps or args.steps_per_epoch,
        'val_during_training': args.val_during_training,
        'topk_rate': args.topk_rate,
        'use_morph': args.use_morph,
        'use_circularity': args.use_circularity,
        'morph_weight': args.morph_weight,
        'use_pixel_unshuffle': not args.no_pixel_unshuffle,
        'filter_empty_labels': args.filter_empty_labels,
        'multi_sensor_model': args.multi_sensor_model,
        'adapter_dim': args.adapter_dim,
        'mixed_data': args.mixed_data,
        's2_dir': args.s2_dir or str(PROJECT_ROOT / 'dataset3.0' / 's2_filtered_3pct'),
        'l8_dir': args.l8_dir or str(PROJECT_ROOT / 'dataset3.0' / 'l8_algae_256_filtered'),
    }

    print("=" * 60)
    print(f"纯分割训练启动：{config['exp_id']}")
    print("=" * 60)
    if config['mixed_data']:
        print(f"模式：混合训练 S2+L8")
        print(f"S2 数据集：{config['s2_dir']}")
        print(f"L8 数据集：{config['l8_dir']}")
    else:
        print(f"数据集：{config['msi_dir']}")
    print(f"模型：MSISingleUNet (无 SFT、无重构)")
    print(f"输入通道：{config['msi_channels']}")
    print(f"ECA TopK: {config['eca_topk_msi']}")
    print(f"Hidden Dim: {config['hidden_dim']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Learning Rate: {config['lr']}")
    print(f"Mamba Layers: {config['num_mamba_layers']}, TopK Rate: {config['topk_rate']}")
    if config['use_morph']:
        print(f"形态损失: 启用 (weight={config['morph_weight']})")
    print("=" * 60)

    train_single_fold(config)


if __name__ == "__main__":
    main()
