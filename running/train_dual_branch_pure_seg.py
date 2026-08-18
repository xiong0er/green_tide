"""
双支路 U-Net 纯分割训练脚本
支持：
- MSI + OLI 混合训练
- 单支路/双支路动态切换
- 纯分割（无重构）
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
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

# 导入项目模块
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from running.dual_branch_seg import DualBranchSegUNet
from running.mae import FastPerlinMasker
from running.data_loader import (
    MSIDataset, MSIOLIDataset,
    get_msi_single_transforms, get_transforms
)

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
    """分割+重建损失（双支路）"""
    def __init__(self, device='cuda', pos_weight=10.0, use_recon=False, recon_weight=0.3,
                 disable_seg=False, use_algae_classifier=False, algae_cls_weight=0.5,
                 empty_l8_weight=1.0, l8_nonempty_weight=2.0, use_morph=False, morph_weight=0.01,
                 dynamic_empty_weight=False,
                 morph_use_shape=True, morph_use_connectivity=True, morph_use_multiscale=True,
                 train_sensor='both'):
        super().__init__()
        self.device = device
        self.use_recon = use_recon
        self.recon_weight = recon_weight
        self.disable_seg = disable_seg
        self.use_algae_cls = use_algae_classifier
        self.empty_l8_weight = empty_l8_weight
        self.l8_nonempty_weight = l8_nonempty_weight
        self.algae_cls_weight = algae_cls_weight
        self.use_morph = use_morph
        self.morph_weight = morph_weight
        self.dynamic_empty_weight = dynamic_empty_weight
        self.train_sensor = train_sensor

        # 分割损失
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(pos_weight=pos_weight)
        self.tversky_loss = TverskyLoss(alpha=0.3, beta=0.7)

        # 形态学损失
        if use_morph:
            from loss.morphological import OptimizedMorphologicalLoss
            self.morph_loss = OptimizedMorphologicalLoss(
                device_type=device, use_circularity=False,
                use_shape=morph_use_shape,
                use_connectivity=morph_use_connectivity,
                use_multiscale=morph_use_multiscale,
            )

        # 分类损失
        if use_algae_classifier:
            self.cls_loss_fn = nn.CrossEntropyLoss()

        # 权重配置
        self.weights = {
            'seg_dice': 0.5,
            'seg_focal': 1.0,
            'seg_tversky': 0.5,
        }

    def forward(self, outputs, targets, msi_input=None, oli_input=None):
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=outputs['head1_msi_seg'].device)

        if not self.disable_seg:
            # Head1 (MSI) 分割损失
            if self.train_sensor in ('both', 's2'):
                head1_seg = outputs['head1_msi_seg']
                if 'msi_seg_gt' in targets:
                    msi_seg_gt = targets['msi_seg_gt']
                    if head1_seg.shape[2:] != msi_seg_gt.shape[2:]:
                        head1_seg = F.interpolate(head1_seg, size=msi_seg_gt.shape[2:], mode='bilinear', align_corners=False)
                    # 有效像素掩膜: 排除 NaN 像素
                    if 'msi_valid_mask' in targets:
                        msi_valid = targets['msi_valid_mask'].to(device=head1_seg.device, dtype=head1_seg.dtype)
                        head1_seg = head1_seg * msi_valid
                        msi_seg_gt = msi_seg_gt * msi_valid
                    loss_dice = self.dice_loss(head1_seg, msi_seg_gt)
                    loss_focal = self.focal_loss(head1_seg, msi_seg_gt)
                    loss_tversky = self.tversky_loss(head1_seg, msi_seg_gt)
                    loss_dict['msi_seg_dice'] = loss_dice.item()
                    loss_dict['msi_seg_focal'] = loss_focal.item()
                    loss_dict['msi_seg_tversky'] = loss_tversky.item()
                    total_loss += (self.weights['seg_dice'] * loss_dice +
                                  self.weights['seg_focal'] * loss_focal +
                                  self.weights['seg_tversky'] * loss_tversky)

            # Head2 (OLI) 分割损失
            if self.train_sensor in ('both', 'l8'):
                head2_seg = outputs['head2_oli_seg']
                if 'oli_seg_gt' in targets:
                    oli_seg_gt = targets['oli_seg_gt']
                    if head2_seg.shape[2:] != oli_seg_gt.shape[2:]:
                        head2_seg = F.interpolate(head2_seg, size=oli_seg_gt.shape[2:], mode='bilinear', align_corners=False)
                    # 有效像素掩膜: 排除 NaN 像素
                    if 'oli_valid_mask' in targets:
                        oli_valid = targets['oli_valid_mask'].to(device=head2_seg.device, dtype=head2_seg.dtype)
                        head2_seg = head2_seg * oli_valid
                        oli_seg_gt = oli_seg_gt * oli_valid
                    # 空标签样本降权：让模型专注从非空样本学藻类特征
                    if self.dynamic_empty_weight and self.use_algae_cls and 'oli_algae_logits' in outputs:
                        # 动态系数：基于分类器置信度的逐样本权重
                        probs = F.softmax(outputs['oli_algae_logits'], dim=1)
                        prob_has_algae = probs[:, 1].detach()  # [B], 分类器判断"有藻"的概率
                        has_algae = (oli_seg_gt.amax(dim=(1,2,3)) > 0)
                        # 逐样本计算 seg loss, 用 prob_has_algae 加权
                        # 非空样本额外乘 l8_nonempty_weight
                        sample_weights = prob_has_algae.clone()
                        if has_algae.any():
                            sample_weights[has_algae] *= getattr(self, 'l8_nonempty_weight', 2.0)
                        # 逐样本 loss
                        for b_idx in range(head2_seg.size(0)):
                            if sample_weights[b_idx] < 1e-6:
                                continue
                            dice_b = self.dice_loss(head2_seg[b_idx:b_idx+1], oli_seg_gt[b_idx:b_idx+1])
                            focal_b = self.focal_loss(head2_seg[b_idx:b_idx+1], oli_seg_gt[b_idx:b_idx+1])
                            tver_b = self.tversky_loss(head2_seg[b_idx:b_idx+1], oli_seg_gt[b_idx:b_idx+1])
                            total_loss += sample_weights[b_idx] * (
                                self.weights['seg_dice'] * dice_b +
                                self.weights['seg_focal'] * focal_b +
                                self.weights['seg_tversky'] * tver_b
                            )
                    elif self.empty_l8_weight < 1.0:
                        has_algae = (oli_seg_gt.amax(dim=(1,2,3)) > 0)
                        if has_algae.any() and (~has_algae).any():
                            loss_ne = (self.weights['seg_dice'] * self.dice_loss(head2_seg[has_algae], oli_seg_gt[has_algae]) +
                                       self.weights['seg_focal'] * self.focal_loss(head2_seg[has_algae], oli_seg_gt[has_algae]) +
                                       self.weights['seg_tversky'] * self.tversky_loss(head2_seg[has_algae], oli_seg_gt[has_algae]))
                            loss_emp = (self.weights['seg_dice'] * self.dice_loss(head2_seg[~has_algae], oli_seg_gt[~has_algae]) +
                                        self.weights['seg_focal'] * self.focal_loss(head2_seg[~has_algae], oli_seg_gt[~has_algae]) +
                                        self.weights['seg_tversky'] * self.tversky_loss(head2_seg[~has_algae], oli_seg_gt[~has_algae]))
                            total_loss += self.l8_nonempty_weight * loss_ne + self.empty_l8_weight * loss_emp
                        else:
                            total_loss += (self.weights['seg_dice'] * self.dice_loss(head2_seg, oli_seg_gt) +
                                          self.weights['seg_focal'] * self.focal_loss(head2_seg, oli_seg_gt) +
                                          self.weights['seg_tversky'] * self.tversky_loss(head2_seg, oli_seg_gt))
                    else:
                        total_loss += (self.weights['seg_dice'] * self.dice_loss(head2_seg, oli_seg_gt) +
                                      self.weights['seg_focal'] * self.focal_loss(head2_seg, oli_seg_gt) +
                                      self.weights['seg_tversky'] * self.tversky_loss(head2_seg, oli_seg_gt))
                # End of L8 seg loss block

        # 形态学损失（紧凑度 + 骨架连通性 + 多尺度面积）
        if self.use_morph:
            if self.train_sensor in ('both', 's2') and 'msi_seg_gt' in targets:
                total_loss += self.morph_weight * self.morph_loss(
                    outputs['head1_msi_seg'], targets['msi_seg_gt'])
            if self.train_sensor in ('both', 'l8') and 'oli_seg_gt' in targets:
                total_loss += self.morph_weight * self.morph_loss(
                    outputs['head2_oli_seg'], targets['oli_seg_gt'])

        # 二分类辅助损失（预测 L8 tile 是否有藻类）
        if self.train_sensor in ('both', 'l8') and self.use_algae_cls and 'oli_algae_logits' in outputs and 'oli_seg_gt' in targets:
            oli_algae_logits = outputs['oli_algae_logits']
            # label: 1 if any algae pixel > 0
            has_algae = (targets['oli_seg_gt'].amax(dim=(1,2,3)) > 0).float()
            loss_cls = self.cls_loss_fn(oli_algae_logits, has_algae.long())
            total_loss += self.algae_cls_weight * loss_cls
            loss_dict['algae_cls'] = loss_cls.item()
            pred_cls = oli_algae_logits.argmax(dim=1)
            loss_dict['algae_cls_acc'] = (pred_cls == has_algae.long()).float().mean().item()

        # 重建损失（可选）
        if self.use_recon and msi_input is not None and oli_input is not None:
            recon_loss = torch.tensor(0.0, device=total_loss.device)
            msi_recon = outputs.get('head1_msi_recon')
            msi_indices = outputs.get('msi_topk_indices')
            if msi_recon is not None and msi_indices is not None:
                b, k, h, w = msi_recon.shape
                msi_target = torch.stack([msi_input[bi, msi_indices[bi]] for bi in range(b)])
                recon_loss += F.mse_loss(msi_recon, msi_target)
            oli_recon = outputs.get('head2_oli_recon')
            oli_indices = outputs.get('oli_topk_indices')
            if oli_recon is not None and oli_indices is not None:
                b, k, h, w = oli_recon.shape
                oli_target = torch.stack([oli_input[bi, oli_indices[bi]] for bi in range(b)])
                recon_loss += F.mse_loss(oli_recon, oli_target)
            total_loss += self.recon_weight * recon_loss
            loss_dict['recon'] = recon_loss.item()

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


# ============ 训练器 ============
class DualBranchPureSegTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logger(config['log_dir'])
        self.logger.info(f"Using device: {self.device}")

        # 模型（无融合 + 共享解码器）
        self.model = DualBranchSegUNet(
            msi_in_channels=config.get('msi_channels', 11),
            oli_in_channels=config.get('oli_channels', 7),
            hidden_dim=config.get('hidden_dim', 1024),
            num_mamba_layers=config.get('num_mamba_layers', 2),
            num_topk_layers=config.get('num_topk_layers', 2),
            eca_k_size=config.get('eca_k_size', 3),
            eca_topk_msi=config.get('eca_topk_msi', 6),
            eca_topk_oli=config.get('eca_topk_oli', 4),
            mamba_dim=config.get('mamba_dim', 64),
            topk_rate=config.get('topk_rate', 0.4),
            shared_seg_head=config.get('shared_head', True),
            use_algae_classifier=config.get('use_algae_classifier', False),
            drop_l8_shallow_skips=config.get('drop_l8_shallow_skips', 0),
            cross_scan_layers=config.get('cross_scan_layers', 0),
            horizontal_scan_layers_msi=config.get('horizontal_scan_layers', 0),
            horizontal_scan_layers_oli=config.get('horizontal_scan_layers_oli', 0),
            transformer_layers_msi=config.get('transformer_layers', 0),
            transformer_layers_oli=config.get('transformer_layers_oli', 0),
            deep_topk_rate=config.get('deep_topk_rate', None),
            eca_mode=config.get('eca_mode', 'topk'),
            disable_eca=config.get('disable_eca', False),
        ).to(self.device)

        # 优化器 (必须在 torch.compile 之前创建，否则参数分组冲突)
        base_lr = config['lr']
        encoder_lr_mult = config.get('encoder_lr_mult', 1.0)

        # 分离编码器和其他参数（MSI/OLI 可用不同 LR）
        oli_lr_mult = config.get('oli_lr_mult', 1.0)
        msi_enc_params = list(self.model.msi_encoder.parameters())
        oli_enc_params = list(self.model.oli_encoder.parameters())
        other_params = [p for n, p in self.model.named_parameters()
                       if not n.startswith('msi_encoder') and not n.startswith('oli_encoder')]

        self.optimizer = optim.AdamW([
            {'params': other_params, 'lr': base_lr, 'name': 'decoder_head'},
            {'params': msi_enc_params, 'lr': base_lr * encoder_lr_mult, 'name': 'msi_encoder'},
            {'params': oli_enc_params, 'lr': base_lr * encoder_lr_mult * oli_lr_mult, 'name': 'oli_encoder'}
        ], weight_decay=config.get('weight_decay', 0.01))

        # torch.compile 优化 (必须在优化器之后，避免参数名冲突)
        # MAE 实验跳过 compile，避免编译图在遮挡输入时触发 NaN
        if not config.get('use_mae', False):
            self.logger.info("Applying torch.compile to model...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
            self.logger.info("torch.compile applied successfully")
        else:
            self.logger.info("MAE mode: skipping torch.compile to avoid NaN with masked inputs")

        # MAE 云掩膜（可选）
        self.use_mae = config.get('use_mae', False)
        self.mae_mask_ratio = config.get('mae_mask_ratio', 0.1)
        if self.use_mae:
            self.mae_masker = FastPerlinMasker(
                shape=(config.get('img_size', 256), config.get('img_size', 256)),
                base_res=(16, 16), octaves=3, persistence=0.5
            ).to(self.device)
            self.logger.info(f"MAE cloud masking enabled (ratio={self.mae_mask_ratio})")

        # 学习率调度器
        self.scheduler = self._create_scheduler(self.optimizer)

        # 损失函数
        self.criterion = CombinedLoss(
            device=str(self.device), pos_weight=10.0,
            use_recon=config.get('use_recon', False),
            recon_weight=config.get('recon_weight', 0.3),
            disable_seg=config.get('disable_seg', False),
            use_algae_classifier=config.get('use_algae_classifier', False),
            algae_cls_weight=config.get('algae_cls_weight', 0.5),
            empty_l8_weight=config.get('empty_l8_weight', 1.0),
            l8_nonempty_weight=config.get('l8_nonempty_weight', 2.0),
            use_morph=config.get('use_morph', False),
            morph_weight=config.get('morph_weight', 0.01),
            dynamic_empty_weight=config.get('dynamic_empty_weight', False),
            morph_use_shape=config.get('morph_use_shape', True),
            morph_use_connectivity=config.get('morph_use_connectivity', True),
            morph_use_multiscale=config.get('morph_use_multiscale', True),
            train_sensor=config.get('train_sensor', 'both'),
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

        # 验证集评价指标记录
        self.val_metrics = {
            'best_iou': 0.0,
            'best_precision': 0.0,
            'best_recall': 0.0,
            'best_f1': 0.0,
        }

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
            oli = batch['oli'].to(self.device) if 'oli' in batch else None

            # MAE 云掩膜
            if self.use_mae:
                msi_input = msi.clone()
                msi, _ = self.mae_masker(msi, mask_ratio=self.mae_mask_ratio)

            # 前向传播（全程 fp32；AMP bfloat16 会导致 NaN）
            self.optimizer.zero_grad()

            if oli is not None:
                outputs = self.model(msi, oli)
            else:
                # 单支路模式（只传 MSI）
                outputs = self.model(msi, torch.zeros_like(msi)[:, :6, :, :])

            # 计算损失
            targets = {k: v.to(self.device) for k, v in batch.items()
                      if k not in ['msi', 'oli']}

            loss, loss_dict = self.criterion(outputs, targets, msi, oli)

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

    def validate(self, dataloader, steps_per_epoch=None):
        """
        验证一个 epoch

        参数:
            dataloader: 验证数据加载器
            steps_per_epoch: 可选，限制验证迭代步数
        """
        self.model.eval()
        epoch_losses = defaultdict(list)
        metrics = {'iou': [], 'precision': [], 'recall': [], 'f1': []}

        self.logger.info(f"Starting validation with {len(dataloader)} batches...")
        val_start_time = time.time()

        with torch.no_grad():
            steps_completed = 0
            for idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
                if steps_per_epoch is not None and steps_completed >= steps_per_epoch:
                    break

                msi = batch['msi'].to(self.device)
                oli = batch['oli'].to(self.device) if 'oli' in batch else None

                if oli is not None:
                    outputs = self.model(msi, oli)
                else:
                    outputs = self.model(msi, torch.zeros_like(msi)[:, :6, :, :])

                # loss 在 fp32 下计算以保证数值精度
                targets = {k: v.to(self.device) for k, v in batch.items()
                          if k not in ['msi', 'oli']}

                loss, loss_dict = self.criterion(outputs, targets, msi, oli)

                for k, v in loss_dict.items():
                    epoch_losses[k].append(v)

                # 计算评价指标 (使用 head1 的分割输出)
                pred_seg = outputs.get('head1_msi_seg', outputs.get('head2_oli_seg'))
                target_seg = targets.get('msi_seg_gt', targets.get('oli_seg_gt'))
                if pred_seg is not None and target_seg is not None:
                    iou, precision, recall, f1 = self._calculate_metrics(pred_seg, target_seg)
                    metrics['iou'].append(iou)
                    metrics['precision'].append(precision)
                    metrics['recall'].append(recall)
                    metrics['f1'].append(f1)

                steps_completed += 1

        result = {k: np.mean(v) for k, v in epoch_losses.items()}
        result['val_iou'] = np.mean(metrics['iou']) if metrics['iou'] else 0
        result['val_precision'] = np.mean(metrics['precision']) if metrics['precision'] else 0
        result['val_recall'] = np.mean(metrics['recall']) if metrics['recall'] else 0
        result['val_f1'] = np.mean(metrics['f1']) if metrics['f1'] else 0
        val_elapsed = time.time() - val_start_time
        self.logger.info(f"Validation completed in {val_elapsed:.1f}s")
        return result

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
        # 是否在每个 epoch 后进行验证（默认 False，只在训练结束后验证）
        val_during_training = self.config.get('val_during_training', False)
        val_steps = self.config.get('val_steps_per_epoch', steps_per_epoch)

        try:
            for epoch in range(self.epoch, self.config['epochs']):
                self.epoch = epoch
                start_time = time.time()

                # 更新 ECA 层的噪声退火
                if hasattr(self.model, 'set_epoch'):
                    self.model.set_epoch(epoch, self.config['epochs'])

                train_losses = self.train_epoch(train_loader, steps_per_epoch=steps_per_epoch)

                elapsed = time.time() - start_time

                # 如果开启验证且不是最后一个 epoch，才进行验证
                if val_during_training:
                    val_losses = self.validate(val_loader, steps_per_epoch=val_steps)
                    elapsed += time.time() - start_time

                    self.logger.info(
                        f"Epoch {epoch}/{self.config['epochs']} - "
                        f"Train: {train_losses.get('total', 0):.4f}, "
                        f"Val: {val_losses.get('total', 0):.4f}, "
                        f"Time: {elapsed:.1f}s"
                    )

                    # 打印验证集评价指标
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
                        # 记录最佳评价指标
                        self.val_metrics['best_iou'] = val_losses.get('val_iou', 0)
                        self.val_metrics['best_precision'] = val_losses.get('val_precision', 0)
                        self.val_metrics['best_recall'] = val_losses.get('val_recall', 0)
                        self.val_metrics['best_f1'] = val_losses.get('val_f1', 0)
                        self.logger.info(
                            f"  Best Metrics - IoU: {self.val_metrics['best_iou']:.4f}, "
                            f"Precision: {self.val_metrics['best_precision']:.4f}, "
                            f"Recall: {self.val_metrics['best_recall']:.4f}, "
                            f"F1: {self.val_metrics['best_f1']:.4f}"
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
                else:
                    # 仅训练，不验证
                    self.logger.info(
                        f"Epoch {epoch}/{self.config['epochs']} - "
                        f"Train Loss: {train_losses.get('total', 0):.4f}, "
                        f"Time: {elapsed:.1f}s"
                    )

                    # 仅根据训练损失保存 checkpoint
                    if train_losses.get('total', float('inf')) < self.best_loss:
                        self.best_loss = train_losses.get('total')
                        self.save_checkpoint(is_best=True)

                # 定期邮件通知
                if self.email_notifier and (epoch + 1) % email_interval == 0:
                    try:
                        if val_during_training:
                            self.email_notifier.send_epoch_update(epoch, train_losses, val_losses, self.val_metrics)
                        else:
                            self.email_notifier.send_epoch_update(epoch, train_losses, None, None)
                    except Exception as e:
                        self.logger.error(f"发送 epoch 更新邮件失败：{e}")

        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")
            self.save_checkpoint()

        except Exception as e:
            self.logger.exception(f"Training failed with error: {e}")
            if self.email_notifier:
                try:
                    self.email_notifier.send_failure_notification(str(e))
                except Exception as email_error:
                    self.logger.error(f"发送失败邮件失败：{email_error}")
            raise

        finally:
            total_elapsed = time.time() - start_time_total
            self.logger.info(f"Training completed in {total_elapsed/60:.1f} minutes")

        # 训练结束后进行一次完整验证
        self.logger.info("Starting final validation after training completion...")
        final_val_losses = self.validate(val_loader, steps_per_epoch=None)
        self.logger.info(
            f"Final Validation - Loss: {final_val_losses.get('total', 0):.4f}, "
            f"IoU: {final_val_losses.get('val_iou', 0):.4f}, "
            f"Precision: {final_val_losses.get('val_precision', 0):.4f}, "
            f"Recall: {final_val_losses.get('val_recall', 0):.4f}, "
            f"F1: {final_val_losses.get('val_f1', 0):.4f}"
        )

        # 保存最终模型
        self.save_checkpoint(is_best=True)

        # 训练结束通知
        if self.email_notifier:
            try:
                self.email_notifier.send_training_completion(
                    total_time=total_elapsed,
                    best_loss=self.best_loss,
                    final_val_losses=final_val_losses,
                    val_metrics=self.val_metrics
                )
            except Exception as e:
                self.logger.error(f"发送训练完成邮件失败：{e}")


# ============ 数据加载 ============
def _filter_empty_label_files(files):
    """过滤掉标签全零的 tif（标签在最后一个波段）"""
    import rasterio
    kept = []
    for f in files:
        try:
            with rasterio.open(f) as src:
                if src.read(src.count).max() > 0:
                    kept.append(f)
        except Exception:
            pass
    return kept


def create_dataloaders(config):
    """创建双支路数据加载器"""
    msi_dir = Path(config['msi_dir'])
    oli_dir = config.get('oli_dir')

    # 扫描文件
    msi_files = sorted(list(msi_dir.glob('*.tif')))
    print(f"找到 {len(msi_files)} 个 MSI 文件")

    if oli_dir:
        oli_files = sorted(list(Path(oli_dir).glob('*.tif')))
        print(f"找到 {len(oli_files)} 个 OLI 文件")
    else:
        oli_files = None

    # 过滤空标签（L8 49% 全零，对 L8 性能至关重要；S2 全含藻类，过滤后数量不变）
    if config.get('filter_empty_labels', True):
        n0 = len(msi_files)
        msi_files = _filter_empty_label_files(msi_files)
        print(f"MSI 空标签过滤: {n0} -> {len(msi_files)}")
        if oli_files:
            n0 = len(oli_files)
            oli_files = _filter_empty_label_files(oli_files)
            print(f"OLI 空标签过滤: {n0} -> {len(oli_files)}")

    # 数据集划分
    val_split = config.get('val_split', 0.2)
    val_size = int(len(msi_files) * val_split)

    # 打乱索引
    np.random.seed(config.get('seed', 42))
    indices = np.random.permutation(len(msi_files))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    print(f"训练集：{len(train_indices)} 样本，验证集：{len(val_indices)} 样本")

    # 创建数据集
    img_size = config.get('img_size', 256)

    if oli_dir and oli_files:
        # 双支路数据集
        train_dataset = MSIOLIDataset(
            msi_files=[msi_files[i] for i in train_indices],
            oli_files=[oli_files[i % len(oli_files)] for i in train_indices],
            transform=get_transforms('train', img_size),
            target_size=img_size
        )
        val_dataset = MSIOLIDataset(
            msi_files=[msi_files[i] for i in val_indices],
            oli_files=[oli_files[i % len(oli_files)] for i in val_indices],
            transform=get_transforms('val', img_size),
            target_size=img_size
        )
    else:
        # 单支路数据集
        train_dataset = MSIDataset(
            msi_files=[msi_files[i] for i in train_indices],
            transform=get_msi_single_transforms('train', img_size),
            target_size=img_size
        )
        val_dataset = MSIDataset(
            msi_files=[msi_files[i] for i in val_indices],
            transform=get_msi_single_transforms('val', img_size),
            target_size=img_size
        )

    # 创建 DataLoader
    batch_size = config.get('batch_size', 8)
    num_workers = config.get('num_workers', 0)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader


# ============ 主函数 ============
def train_dual_branch_pure_seg(config):
    """训练双支路纯分割模型"""
    config['exp_id'] = config.get('exp_id', 'dual_branch_pure_seg')
    config['log_dir'] = f"logs/experiments/{config['exp_id']}"
    config['checkpoint_dir'] = f"checkpoints/experiments/{config['exp_id']}"

    train_loader, val_loader = create_dataloaders(config)
    trainer = DualBranchPureSegTrainer(config)
    trainer.train(train_loader, val_loader)

    return trainer.best_loss


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='双支路纯分割训练脚本')
    parser.add_argument('--exp_id', type=str, default='Exp-1D', help='实验 ID')
    parser.add_argument('--msi_dir', type=str, default=None, help='MSI 数据目录')
    parser.add_argument('--oli_dir', type=str, default=None, help='OLI 数据目录')
    parser.add_argument('--msi_channels', type=int, default=11, help='MSI 通道数')
    parser.add_argument('--oli_channels', type=int, default=7, help='OLI 通道数')
    parser.add_argument('--eca_topk_msi', type=int, default=6, help='ECA TopK MSI 通道数')
    parser.add_argument('--eca_topk_oli', type=int, default=3, help='ECA TopK OLI 通道数')
    parser.add_argument('--hidden_dim', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--num_mamba_layers', type=int, default=2, help='Mamba 层数')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--steps_per_epoch', type=int, default=None, help='每个 epoch 的最大训练步数（用于加速训练）')
    parser.add_argument('--val_steps', type=int, default=None, help='每个 epoch 的最大验证步数')
    parser.add_argument('--val_during_training', action='store_true', help='在训练过程中每个 epoch 后进行验证（默认只在训练结束后验证）')
    parser.add_argument('--use_recon', action='store_true', help='启用重建损失（默认仅分割）')
    parser.add_argument('--recon_weight', type=float, default=0.3, help='重建损失权重')
    parser.add_argument('--use_mae', action='store_true', help='启用 MAE 云模拟')
    parser.add_argument('--mae_mask_ratio', type=float, default=0.1, help='MAE 云掩膜比例')
    parser.add_argument('--disable_seg', action='store_true', help='禁用分割损失（仅重建）')
    parser.add_argument('--head_mode', type=str, default='shared', choices=['shared', 'separate'],
                        help='分割头模式：shared=共享头(变体A), separate=每传感器独立头(变体B)')
    parser.add_argument('--no_filter_empty', action='store_true',
                        help='不过滤空标签（默认过滤，L8 空标签对性能影响极大）')
    parser.add_argument('--use_algae_classifier', action='store_true',
                        help='启用 L8 二分类辅助头（预测是否有藻类）')
    parser.add_argument('--algae_cls_weight', type=float, default=0.5,
                        help='二分类辅助损失权重（默认 0.5）')
    parser.add_argument('--drop_l8_shallow_skips', type=int, default=1,
                        help='L8 丢弃前 N 个浅层 skip（默认 1，0 为保留所有 skip）')
    parser.add_argument('--l8_nonempty_weight', type=float, default=2.0,
                        help='非空 L8 样本 seg loss 权重（已废弃，用 --empty_l8_weight）')
    parser.add_argument('--empty_l8_weight', type=float, default=0.3,
                        help='空标签 L8 样本 seg loss 降权（默认 0.3，保护非空 L8 训练质量）')
    parser.add_argument('--dynamic_empty_weight', action='store_true',
                        help='空标签权重改为基于 AlgaeClassifier 置信度的动态系数')
    parser.add_argument('--eca_mode', type=str, default='topk',
                        choices=['topk', 'soft', 'random_fixed'],
                        help='ECA 模式: topk(硬截断,默认), soft(软加权), random_fixed(随机固定K波段)')
    parser.add_argument('--disable_eca', action='store_true',
                        help='禁用 ECA，使用全部波段 (Full Bands 基线)')
    parser.add_argument('--train_sensor', type=str, default='both',
                        choices=['both', 's2', 'l8'],
                        help='训练传感器: both(默认), s2(仅S2), l8(仅L8)')
    parser.add_argument('--use_morph', action='store_true',
                        help='启用形态学损失（紧凑度+骨架+多尺度面积）')
    parser.add_argument('--morph_weight', type=float, default=0.01,
                        help='形态学损失权重（默认 0.01）')
    parser.add_argument('--morph_use_shape', action='store_true', default=True,
                        help='形态学损失子项：紧凑度形状损失（默认启用）')
    parser.add_argument('--no-morph_use_shape', action='store_false', dest='morph_use_shape',
                        help='禁用形态学形状损失子项')
    parser.add_argument('--morph_use_connectivity', action='store_true', default=True,
                        help='形态学损失子项：骨架连通性损失（默认启用）')
    parser.add_argument('--no-morph_use_connectivity', action='store_false', dest='morph_use_connectivity',
                        help='禁用形态学连通性损失子项')
    parser.add_argument('--morph_use_multiscale', action='store_true', default=True,
                        help='形态学损失子项：多尺度面积损失（默认启用）')
    parser.add_argument('--no-morph_use_multiscale', action='store_false', dest='morph_use_multiscale',
                        help='禁用形态学多尺度损失子项')
    parser.add_argument('--cross_scan_layers', type=int, default=2,
                        help='前 N 层使用 CrossScan Mamba（列+行四方向，默认 2）')
    parser.add_argument('--horizontal_scan_layers', type=int, default=0,
                        help='前 N 层使用 HorizontalScan Mamba（仅行左右双向，默认 0）')
    parser.add_argument('--horizontal_scan_layers_oli', type=int, default=None,
                        help='L8 前 N 层使用 HorizontalScan（覆盖 --horizontal_scan_layers 对 L8 的设置，默认跟随 --horizontal_scan_layers）')
    parser.add_argument('--topk_rate', type=float, default=0.4,
                        help='TopK attention 保留率（默认 0.4, 最佳 0.2）')
    parser.add_argument('--deep_topk_rate', type=float, default=0.15,
                        help='深层 TopK attention 保留率（默认 0.15，更稀疏）')
    parser.add_argument('--transformer_layers', type=int, default=0,
                        help='前 N 层使用 Window Transformer（MSI+L8 统一，默认 0）')
    parser.add_argument('--transformer_layers_oli', type=int, default=None,
                        help='L8 前 N 层使用 Transformer（覆盖 --transformer_layers 对 L8 的设置，默认跟随 --transformer_layers）')
    parser.add_argument('--oli_lr_mult', type=float, default=1.0,
                        help='L8 encoder 学习率倍率（默认 1.0）')

    args = parser.parse_args()

    # 默认配置
    config = {
        'exp_id': args.exp_id,
        'msi_dir': args.msi_dir or str(PROJECT_ROOT / 'dataset3.0' / 's2_filtered_3pct'),
        'oli_dir': args.oli_dir or str(PROJECT_ROOT / 'dataset3.0' / 'l8_algae_data'),
        'msi_channels': args.msi_channels,
        'oli_channels': args.oli_channels,
        'eca_topk_msi': args.eca_topk_msi,
        'eca_topk_oli': args.eca_topk_oli,
        'hidden_dim': args.hidden_dim,
        'num_mamba_layers': args.num_mamba_layers,
        'num_topk_layers': 2,
        'eca_k_size': 3,
        'mamba_dim': 64,
        'img_size': 256,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'encoder_lr_mult': 1.0,
        'steps_per_epoch': args.steps_per_epoch,
        'val_steps_per_epoch': args.val_steps or args.steps_per_epoch,
        'val_during_training': args.val_during_training,
        'use_recon': args.use_recon,
        'recon_weight': args.recon_weight,
        'use_mae': args.use_mae,
        'mae_mask_ratio': args.mae_mask_ratio,
        'disable_seg': args.disable_seg,
        'shared_head': (args.head_mode == 'shared'),
        'filter_empty_labels': (not args.no_filter_empty),
        'use_algae_classifier': args.use_algae_classifier,
        'algae_cls_weight': args.algae_cls_weight,
        'drop_l8_shallow_skips': args.drop_l8_shallow_skips,
        'l8_nonempty_weight': args.l8_nonempty_weight,
        'empty_l8_weight': args.empty_l8_weight,
        'dynamic_empty_weight': args.dynamic_empty_weight,
        'use_morph': args.use_morph,
        'morph_weight': args.morph_weight,
        'morph_use_shape': args.morph_use_shape,
        'morph_use_connectivity': args.morph_use_connectivity,
        'morph_use_multiscale': args.morph_use_multiscale,
        'cross_scan_layers': args.cross_scan_layers,
        'horizontal_scan_layers': args.horizontal_scan_layers,
        'horizontal_scan_layers_oli': args.horizontal_scan_layers_oli if args.horizontal_scan_layers_oli is not None else args.horizontal_scan_layers,
        'topk_rate': args.topk_rate,
        'deep_topk_rate': args.deep_topk_rate,
        'transformer_layers': args.transformer_layers,
        'transformer_layers_oli': args.transformer_layers_oli if args.transformer_layers_oli is not None else args.transformer_layers,
        'eca_mode': args.eca_mode,
        'disable_eca': args.disable_eca,
        'train_sensor': args.train_sensor,
        'oli_lr_mult': args.oli_lr_mult,
        'weight_decay': 0.01,
        'warmup_steps': 500,
        'use_reduce_lr_on_plateau': True,
        'reduce_lr_patience': 10,
        'reduce_lr_factor': 0.7,
        'patience': 15,
        'early_stop_delta': 1e-6,
        'val_split': 0.2,
        'seed': 42,
        'use_email': False,
        'num_workers': 0,
    }

    print("=" * 60)
    print(f"双支路纯分割训练启动：{config['exp_id']}")
    print("=" * 60)
    print(f"MSI 数据集：{config['msi_dir']}")
    print(f"OLI 数据集：{config['oli_dir']}")
    print(f"模型：DualBranchUNet (纯分割，无重构)")
    print(f"MSI 输入通道：{config['msi_channels']}")
    print(f"OLI 输入通道：{config['oli_channels']}")
    print(f"ECA TopK (MSI): {config['eca_topk_msi']}")
    print(f"ECA TopK (OLI): {config['eca_topk_oli']}")
    print(f"Hidden Dim: {config['hidden_dim']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Learning Rate: {config['lr']}")
    print("=" * 60)

    train_dual_branch_pure_seg(config)


if __name__ == "__main__":
    main()
