"""
Compare 8 models on picture/ tiles side-by-side.

Models:
  - Ours (DualBranchSegUNet, asymmetric CrossScan + WinTrans)
  - U-Net (vanilla_unet)
  - SEnSeI (SpectralSetSharedUNet reimplementation)
  - AnySat (SensorAdapterSharedUNet reimplementation)
  - AlgaeMamba, Swin U-Net, Attn U-Net, U-Net++

Color scheme:
  - Green = TP (correct algae)
  - Blue  = TN (correct background)
  - Red   = Error (FP false-alarm + FN missed-algae, unified)
  (green IoU>0.85 border highlight removed)
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys, torch, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import zoom
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from running.dual_branch_seg import DualBranchSegUNet
from running.routed_unet import RoutedSingleUNet
from running.multi_sensor_baselines import build_multisensor_model

PIC_DIR = PROJECT_ROOT / 'picture'
CKPT_DIR = PROJECT_ROOT / 'checkpoints' / 'experiments'
OUT_DIR = PROJECT_ROOT / 'visualization' / 'output'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model registry: (display_name, kind, checkpoint_s2, checkpoint_l8)
#   kind = 'ours' | 'single' | 'multi'
# ---------------------------------------------------------------------------
MODELS = [
    ('Ours\n(DualBranch)', 'ours',
     'Exp-Dual-Asym-100ep', 'Exp-Dual-Asym-100ep'),
    ('U-Net', 'single',
     'Baseline-vanilla_unet-S2', 'Baseline-vanilla_unet-L8'),
    ('SEnSeI', 'multi',
     'Exp-Baseline-SpectralSet-1024', 'Exp-Baseline-SpectralSet-1024'),
    ('AnySat', 'multi',
     'Exp-Baseline-SensorAdapter-1024', 'Exp-Baseline-SensorAdapter-1024'),
    ('AlgaeMamba', 'single',
     'Baseline-algae_mamba-S2', 'Baseline-algae_mamba-L8'),
    ('Swin U-Net', 'single',
     'Baseline-swin_unet-S2', 'Baseline-swin_unet-L8'),
    ('Attn U-Net', 'single',
     'Baseline-attention_unet-S2', 'Baseline-attention_unet-L8'),
    ('U-Net++', 'single',
     'Baseline-unetpp-S2', 'Baseline-unetpp-L8'),
]


def load_tile(filepath):
    with rasterio.open(filepath) as src:
        data = src.read()
    data = np.nan_to_num(data, nan=0.0)
    c, h, w = data.shape
    target = 256
    img_data = data[:-1].astype(np.float32)
    label_data = data[-1:].astype(np.float32)
    if h != target or w != target:
        zh, zw = target / h, target / w
        img_data = zoom(img_data, (1, zh, zw), order=1)
        label_data = zoom(label_data, (1, zh, zw), order=0)
    n_ch = img_data.shape[0]
    # RGB composite (S2 B4/B3/B2 or L8 SR_B4/B3/B2 = idx 3,2,1)
    rgb_idx = [3, 2, 1]
    rgb_raw = np.stack([img_data[i] for i in rgb_idx], axis=-1)
    rgb_raw = np.clip(rgb_raw, 0, None)
    for b in range(3):
        vmin, vmax = np.percentile(rgb_raw[..., b], (2, 98))
        rgb_raw[..., b] = np.clip((rgb_raw[..., b] - vmin) / (vmax - vmin + 1e-8), 0, 1)
    # Normalized model input
    low, high = np.percentile(img_data, (0.5, 99.5))
    denom = np.float32(high - low + 1e-8)
    img_norm = np.clip((img_data - np.float32(low)) / denom, 0, 1).astype(np.float32)
    return (torch.from_numpy(img_norm),
            torch.clamp(torch.from_numpy(label_data), 0, 1),
            rgb_raw, n_ch)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def build_ours(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state_dict'].items()}
    cfg = ckpt['config']
    model = DualBranchSegUNet(
        msi_in_channels=cfg.get('msi_channels', 11),
        oli_in_channels=cfg.get('oli_channels', 7),
        hidden_dim=cfg.get('hidden_dim', 1024),
        num_mamba_layers=cfg.get('num_mamba_layers', 2),
        num_topk_layers=cfg.get('num_topk_layers', 2),
        eca_k_size=cfg.get('eca_k_size', 3),
        eca_topk_msi=cfg.get('eca_topk_msi', 6),
        eca_topk_oli=cfg.get('eca_topk_oli', 4),
        mamba_dim=cfg.get('mamba_dim', 64),
        topk_rate=cfg.get('topk_rate', 0.4),
        shared_seg_head=cfg.get('shared_head', True),
        use_algae_classifier=cfg.get('use_algae_classifier', False),
        drop_l8_shallow_skips=cfg.get('drop_l8_shallow_skips', 0),
        cross_scan_layers=cfg.get('cross_scan_layers', 0),
        horizontal_scan_layers_msi=cfg.get('horizontal_scan_layers', 0),
        horizontal_scan_layers_oli=cfg.get('horizontal_scan_layers_oli', 0),
        transformer_layers_msi=cfg.get('transformer_layers', 0),
        transformer_layers_oli=cfg.get('transformer_layers_oli', 0),
        deep_topk_rate=cfg.get('deep_topk_rate', None),
        eca_mode=cfg.get('eca_mode', 'topk'),
        disable_eca=cfg.get('disable_eca', False),
    )
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE).eval()
    return model


def build_single(ckpt_path):
    from running.train_single_sensor_baselines import build_model
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state_dict'].items()}
    cfg = ckpt['config']
    model_name = cfg.get('model', 'vanilla_unet')
    sensor = cfg.get('sensor', 's2')
    in_ch = 11 if sensor == 's2' else 7
    model = build_model(model_name, in_ch, cfg.get('img_size', 256))
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE).eval()
    return model, in_ch


def build_multi(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state_dict'].items()}
    cfg = ckpt['config']
    model_type = cfg.get('multi_sensor_model', 'routed')
    if model_type == 'routed':
        model = RoutedSingleUNet(
            msi_in_channels=cfg.get('msi_channels', 11),
            oli_in_channels=cfg.get('oli_channels', 7),
            hidden_dim=cfg.get('hidden_dim', 512),
            bottleneck_size=cfg.get('bottleneck_size', 16),
            num_mamba_layers=cfg.get('num_mamba_layers', 2),
            num_topk_layers=cfg.get('num_topk_layers', 2),
            eca_k_size=cfg.get('eca_k_size', 3),
            eca_topk_msi=cfg.get('eca_topk_msi', 6),
            eca_topk_oli=cfg.get('eca_topk_oli', 4),
            mamba_dim=cfg.get('mamba_dim', 64),
            topk_rate=cfg.get('topk_rate', 0.4),
        )
    else:
        model = build_multisensor_model(
            model_type,
            hidden_dim=cfg.get('hidden_dim', 512),
            num_mamba_layers=cfg.get('num_mamba_layers', 2),
            num_topk_layers=cfg.get('num_topk_layers', 2),
            mamba_dim=cfg.get('mamba_dim', 64),
            topk_rate=cfg.get('topk_rate', 0.4),
            adapter_dim=cfg.get('adapter_dim', 64),
            shared_pixel_unshuffle=cfg.get('use_pixel_unshuffle', False),
        )
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE).eval()
    return model


# ---------------------------------------------------------------------------
# Prediction dispatch
# ---------------------------------------------------------------------------
def predict_ours(model, img_tensor, n_ch):
    img_gpu = img_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        if n_ch == 11:  # S2 -> MSI branch
            dummy = torch.zeros(1, 7, 256, 256).to(DEVICE)
            pred = model(img_gpu, dummy)['head1_msi_seg']
        else:           # L8 -> OLI branch
            dummy = torch.zeros(1, 11, 256, 256).to(DEVICE)
            pred = model(dummy, img_gpu)['head2_oli_seg']
    return (pred.cpu() > 0.5).float().squeeze().numpy()


def predict_single(model, img_tensor):
    img_gpu = img_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(img_gpu)
        if isinstance(out, dict):
            pred = out['head1_msi_seg'] if 'head1_msi_seg' in out else out.get('seg', out)
        else:
            pred = out
    return (pred.cpu() > 0.5).float().squeeze().numpy()


def predict_multi(model, img_tensor, n_ch):
    if n_ch == 11:
        img11 = img_tensor
        sensor = 's2'
    else:
        img11 = torch.nn.functional.pad(img_tensor, (0, 0, 0, 0, 0, 4))  # 7 -> 11
        sensor = 'l8'
    img_gpu = img11.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(img_gpu, [sensor])['msi_seg']
    return (pred.cpu() > 0.5).float().squeeze().numpy()


# ---------------------------------------------------------------------------
# Color code: Green=TP, Blue=TN, Red=Error (FP+FN unified)
# ---------------------------------------------------------------------------
def colorcode(pred_bin, label_bin):
    h, w = pred_bin.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    tp = (pred_bin == 1) & (label_bin == 1)
    tn = (pred_bin == 0) & (label_bin == 0)
    fp = (pred_bin == 1) & (label_bin == 0)
    fn = (pred_bin == 0) & (label_bin == 1)
    rgb[tp] = [0, 255, 0]      # Green: correct algae
    rgb[tn] = [0, 0, 255]      # Blue: correct background
    rgb[fp] = [255, 0, 0]      # Red: false alarm
    rgb[fn] = [255, 0, 0]      # Red: missed algae (was yellow, now unified red)
    return rgb


def iou_score(pred, gt):
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    eps = 1e-6
    return float((tp + eps) / (tp + fp + fn + eps))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Building models...")
    # Ours + multi-sensor models (single models are built per-sensor below)
    models = {}
    for name, kind, ckpt_s2, ckpt_l8 in MODELS:
        try:
            if kind == 'ours':
                models[name] = ('ours', build_ours(CKPT_DIR / ckpt_s2 / 'best.pth'))
            elif kind == 'multi':
                models[name] = ('multi', build_multi(CKPT_DIR / ckpt_s2 / 'best.pth'))
            print(f"  OK  {name}")
        except Exception as e:
            print(f"  FAIL {name}: {e}")

    # Single-sensor models: build S2 and L8 versions (separate checkpoints)
    single_models = {}
    for name, kind, ckpt_s2, ckpt_l8 in MODELS:
        if kind != 'single':
            continue
        try:
            single_models[(name, 's2')] = build_single(CKPT_DIR / ckpt_s2 / 'best.pth')
            single_models[(name, 'l8')] = build_single(CKPT_DIR / ckpt_l8 / 'best.pth')
            print(f"  OK  {name} (S2+L8)")
        except Exception as e:
            print(f"  FAIL {name}: {e}")

    for sensor, subfolder in [('S2', 's2'), ('L8', 'l8')]:
        files = sorted((PIC_DIR / subfolder).glob('*.tif'))
        if not files:
            continue

        n_tiles = len(files)
        n_models = len(MODELS)
        cols = n_models + 2  # RGB + GT + models

        fig, axes = plt.subplots(n_tiles, cols, figsize=(2.6 * cols, 3.0 * n_tiles))
        if n_tiles == 1:
            axes = axes.reshape(1, -1)

        # Column headers
        headers = ['RGB', 'GT'] + [m[0].replace('\n', ' ') for m in MODELS]
        for ci, h in enumerate(headers):
            axes[0, ci].set_title(h, fontsize=8, fontweight='bold')

        for row_i, fp in enumerate(files):
            img, label, rgb, n_ch = load_tile(fp)
            lbl = (label > 0.5).float().squeeze().numpy()

            axes[row_i, 0].imshow(rgb)
            axes[row_i, 1].imshow(lbl, cmap='gray', vmin=0, vmax=1)
            if row_i == 0:
                axes[row_i, 0].set_ylabel('Tile', fontsize=8)

            for mi, (name, kind, ckpt_s2, ckpt_l8) in enumerate(MODELS):
                if kind == 'ours':
                    pred = predict_ours(models[name][1], img, n_ch)
                elif kind == 'multi':
                    pred = predict_multi(models[name][1], img, n_ch)
                else:  # single
                    m, in_ch = single_models[(name, sensor.lower())]
                    pred = predict_single(m, img)

                iou = iou_score(pred, lbl)
                err_map = colorcode(pred, lbl)
                axes[row_i, mi + 2].imshow(err_map)
                #axes[row_i, mi + 2].set_xlabel(f'IoU={iou:.3f}', fontsize=7)

            for ci in range(cols):
                axes[row_i, ci].set_xticks([])
                axes[row_i, ci].set_yticks([])

        legend_patches = [
            mpatches.Patch(color='#00FF00', label='TP: Correct Algae'),
            mpatches.Patch(color='#0000FF', label='TN: Correct Background'),
            mpatches.Patch(color='#FF0000', label='Error (FP + FN)'),
        ]
        fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=9,
                   framealpha=0.9, bbox_to_anchor=(0.5, -0.015))

        fig.suptitle(f'Model Comparison on {sensor} Tiles\n'
                     f'(Green=TP  Blue=TN  Red=Error)',
                     fontsize=14, fontweight='bold', y=0.998)
        plt.tight_layout()
        out_path = OUT_DIR / f'model_comparison_{subfolder}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out_path}')

    print(f'\nDone. Files in: {OUT_DIR}')


if __name__ == '__main__':
    main()
