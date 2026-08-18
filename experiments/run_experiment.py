# ============================================================================
# 绿潮分割实验 - 启动脚本
# 用法：python run_experiment.py <experiment_id> <device_id>
# ============================================================================

import os
# 必须在导入 numpy/torch 之前设置
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import yaml
import torch
from pathlib import Path

# PROJECT_ROOT 是项目根目录 (experiments 的父目录)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# 实验配置
EXPERIMENT_CONFIG = {
    # ========== 第 1 阶段：基线确定 ==========
    "Exp-1A": {
        "name": "S2_SingleUNet_Baseline",
        "model": "MSISingleUNet",
        "msi_dir": str(PROJECT_ROOT / "dataset3.0" / "s2_filtered_3pct"),
        "val_split": 0.2,
        "config": {
            "msi_channels": 11,
            "hidden_dim": 256,
            "bottleneck_size": 16,
            "num_mamba_layers": 1,
            "num_topk_layers": 2,
            "eca_k_size": 3,
            "eca_topk_msi": 6,
            "mamba_dim": 32,
            "seg_only_epochs": 50,  # 纯分割实验，不引入 reconstruction loss
        }
    },
    "Exp-1B": {
        "name": "S2_to_L8_CrossSensor",
        "model": "MSISingleUNet",
        "msi_dir": str(PROJECT_ROOT / "dataset3.0" / "s2_filtered_3pct"),
        "cross_sensor_test": True,
        "cross_sensor_data": str(PROJECT_ROOT / "dataset3.0" / "l8_algae_data"),
        "config": {
            "msi_channels": 11,
            "hidden_dim": 256,
            "bottleneck_size": 16,
            "num_mamba_layers": 1,
            "num_topk_layers": 2,
            "eca_k_size": 3,
            "eca_topk_msi": 6,
            "mamba_dim": 32,
            "seg_only_epochs": 50,  # 纯分割实验，不引入 reconstruction loss
        }
    },
    "Exp-1C": {
        "name": "L8_SingleUNet_Baseline",
        "model": "MSISingleUNet",
        "msi_dir": str(PROJECT_ROOT / "dataset3.0" / "l8_algae_data"),
        "val_split": 0.2,
        "config": {
            "msi_channels": 7,  # L8 数据有 7 个波段 (B1, B2, B3, B4, B5, B6, B7)
            "hidden_dim": 256,
            "bottleneck_size": 16,
            "num_mamba_layers": 1,
            "num_topk_layers": 2,
            "eca_k_size": 3,
            "eca_topk_msi": 4,
            "mamba_dim": 32,
            "seg_only_epochs": 50,  # 纯分割实验，不引入 reconstruction loss
        }
    },
    "Exp-1D": {
        "name": "Mixed_DualBranch_Baseline",
        "model": "DualBranchUNet",
        "msi_dir": str(PROJECT_ROOT / "dataset3.0" / "s2_filtered_3pct"),
        "oli_dir": str(PROJECT_ROOT / "dataset3.0" / "l8_algae_data"),
        "val_split": 0.2,
        "batch_size": 4,  # 双支路模型显存占用大，减小 batch size
        "config": {
            "msi_channels": 11,
            "oli_channels": 7,  # L8 数据有 7 个波段 (B1-B7)
            "hidden_dim": 512,
            "bottleneck_size": 16,
            "num_mamba_layers": 2,
            "num_topk_layers": 2,
            "eca_k_size": 3,
            "eca_topk_msi": 6,
            "eca_topk_oli": 3,
            "mamba_dim": 64,  # 必须能被 headdim=64 整除
            "seg_only_epochs": 50,  # 纯分割实验，不引入 reconstruction loss
        }
    },
    # ========== 第 2 阶段：架构 Ablation ==========
    # Exp-2A 系列：ECA TopK 消融（S2 单支路）
    "Exp-2A-1": {
        "name": "ECA_TopK_3_S2",
        "model": "MSISingleUNet",
        "msi_dir": str(PROJECT_ROOT / "dataset3.0" / "s2_filtered_3pct"),
        "config": {"eca_topk_msi": 3, "hidden_dim": 256, "num_mamba_layers": 1, "seg_only_epochs": 50,
                   "msi_channels": 11, "bottleneck_size": 16, "eca_k_size": 3, "num_topk_layers": 2, "mamba_dim": 32}
    },
    "Exp-2A-2": {
        "name": "ECA_TopK_5_S2",
        "model": "MSISingleUNet",
        "msi_dir": str(PROJECT_ROOT / "dataset3.0" / "s2_filtered_3pct"),
        "config": {"eca_topk_msi": 5, "hidden_dim": 256, "num_mamba_layers": 1, "seg_only_epochs": 50,
                   "msi_channels": 11, "bottleneck_size": 16, "eca_k_size": 3, "num_topk_layers": 2, "mamba_dim": 32}
    },
    "Exp-2A-3": {
        "name": "ECA_TopK_7_S2",
        "model": "MSISingleUNet",
        "msi_dir": str(PROJECT_ROOT / "dataset3.0" / "s2_filtered_3pct"),
        "config": {"eca_topk_msi": 7, "hidden_dim": 256, "num_mamba_layers": 1, "seg_only_epochs": 50,
                   "msi_channels": 11, "bottleneck_size": 16, "eca_k_size": 3, "num_topk_layers": 2, "mamba_dim": 32}
    },
    # Exp-2B 系列：Mamba 层数消融（S2+L8 双分支）
    "Exp-2B-1": {
        "name": "Mamba_0_Layer_S2L8",
        "model": "DualBranchUNet",
        "msi_dir": str(PROJECT_ROOT / "dataset3.0" / "s2_filtered_3pct"),
        "oli_dir": str(PROJECT_ROOT / "dataset3.0" / "l8_algae_data"),
        "config": {"num_mamba_layers": 0, "num_topk_layers": 4, "hidden_dim": 256, "seg_only_epochs": 50,
                   "msi_channels": 11, "oli_channels": 7, "bottleneck_size": 16, "eca_k_size": 3, "eca_topk_msi": 6, "eca_topk_oli": 6, "mamba_dim": 32}
    },
    "Exp-2B-2": {
        "name": "Mamba_2_Layer_S2L8",
        "model": "DualBranchUNet",
        "msi_dir": str(PROJECT_ROOT / "dataset3.0" / "s2_filtered_3pct"),
        "oli_dir": str(PROJECT_ROOT / "dataset3.0" / "l8_algae_data"),
        "config": {"num_mamba_layers": 2, "num_topk_layers": 2, "hidden_dim": 256, "seg_only_epochs": 50,
                   "msi_channels": 11, "oli_channels": 7, "bottleneck_size": 16, "eca_k_size": 3, "eca_topk_msi": 6, "eca_topk_oli": 6, "mamba_dim": 32}
    },
    # Exp-2C 系列：ECA TopK 消融（L8 单支路）
    "Exp-2C-1": {
        "name": "ECA_TopK_3_L8",
        "model": "MSISingleUNet",
        "msi_dir": str(PROJECT_ROOT / "dataset3.0" / "l8_algae_data"),
        "config": {"eca_topk_msi": 3, "hidden_dim": 256, "num_mamba_layers": 1, "seg_only_epochs": 50,
                   "msi_channels": 7, "bottleneck_size": 16, "eca_k_size": 3, "num_topk_layers": 2, "mamba_dim": 32}
    },
    "Exp-2C-2": {
        "name": "ECA_TopK_5_L8",
        "model": "MSISingleUNet",
        "msi_dir": str(PROJECT_ROOT / "dataset3.0" / "l8_algae_data"),
        "config": {"eca_topk_msi": 5, "hidden_dim": 256, "num_mamba_layers": 1, "seg_only_epochs": 50,
                   "msi_channels": 7, "bottleneck_size": 16, "eca_k_size": 3, "num_topk_layers": 2, "mamba_dim": 32}
    },
    "Exp-2C-3": {
        "name": "ECA_TopK_7_L8",
        "model": "MSISingleUNet",
        "msi_dir": str(PROJECT_ROOT / "dataset3.0" / "l8_algae_data"),
        "config": {"eca_topk_msi": 7, "hidden_dim": 256, "num_mamba_layers": 1, "seg_only_epochs": 50,
                   "msi_channels": 7, "bottleneck_size": 16, "eca_k_size": 3, "num_topk_layers": 2, "mamba_dim": 32}
    },
}

# 通用配置
COMMON_CONFIG = {
    "img_size": 256,
    "batch_size": 8,
    "epochs": 50,
    "lr": 0.001,
    "encoder_lr_mult": 1.0,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "warmup_steps": 500,
    "steps_per_epoch": 100,
    "use_reduce_lr_on_plateau": True,
    "reduce_lr_patience": 10,
    "reduce_lr_factor": 0.7,
    "patience": 15,
    "early_stop_delta": 1e-6,
    "use_email": True,
    "email_config_path": "email_config.json",
    "email_interval": 5,
    "plot_interval": 5,
    "seed": 42,
    "use_adv": False,
    "mae_mask_prob": 0.5,
    "mask_ratio": 0.25,
    # K 折交叉验证配置 - 使用 7 折训练 +3 折验证
    "k_fold": 10,
    "train_folds": 7,
    "val_folds": 3,
    "fold_idx": 0,
}


def get_config(exp_id):
    """获取实验配置"""
    if exp_id not in EXPERIMENT_CONFIG:
        raise ValueError(f"Unknown experiment ID: {exp_id}")

    exp_config = EXPERIMENT_CONFIG[exp_id].copy()
    config = COMMON_CONFIG.copy()
    config.update(exp_config)

    # 更新实验特定配置
    if "config" in exp_config:
        for k, v in exp_config["config"].items():
            config[k] = v

    # 设置日志和检查点目录
    config["exp_id"] = exp_id
    config["exp_name"] = exp_config["name"]
    config["log_dir"] = f"./logs/experiments/{exp_id}_{exp_config['name']}"
    config["checkpoint_dir"] = f"./checkpoints/experiments/{exp_id}_{exp_config['name']}"

    return config


def print_experiment_info(exp_id, config):
    """打印实验信息"""
    print("=" * 70)
    print(f"实验 ID: {exp_id}")
    print(f"实验名称：{config['exp_name']}")
    print(f"模型：{config['model']}")
    print("=" * 70)
    print(f"训练数据:")
    if "msi_dir" in config:
        print(f"  MSI: {config['msi_dir']}")
    if "oli_dir" in config:
        print(f"  OLI: {config['oli_dir']}")
    print(f"验证集划分：{config.get('val_split', 'N/A')}")
    print("=" * 70)
    print("模型配置:")
    print(f"  Hidden Dim: {config.get('hidden_dim', 'N/A')}")
    print(f"  ECA TopK (MSI): {config.get('eca_topk_msi', 'N/A')}")
    print(f"  ECA TopK (OLI): {config.get('eca_topk_oli', 'N/A')}")
    print(f"  Mamba 层数：{config.get('num_mamba_layers', 'N/A')}")
    print(f"  TopK 层数：{config.get('num_topk_layers', 'N/A')}")
    print("=" * 70)
    print(f"训练配置:")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Learning Rate: {config['lr']}")
    print(f"  图像尺寸：{config['img_size']}x{config['img_size']}")
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print("用法：python run_experiment.py <experiment_id>")
        print("\n可用实验 ID:")
        for exp_id in sorted(EXPERIMENT_CONFIG.keys()):
            print(f"  {exp_id}: {EXPERIMENT_CONFIG[exp_id]['name']}")
        sys.exit(1)

    exp_id = sys.argv[1]
    config = get_config(exp_id)

    # 打印实验信息
    print_experiment_info(exp_id, config)

    # 根据模型类型导入相应的训练模块
    if config["model"] == "MSISingleUNet":
        from running.train import train_single_fold
        train_single_fold(config, fold_idx=0)
    elif config["model"] == "DualBranchUNet":
        from running.train_dual_branch import train_dual_branch
        train_dual_branch(config)
    else:
        print(f"未知模型类型：{config['model']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
