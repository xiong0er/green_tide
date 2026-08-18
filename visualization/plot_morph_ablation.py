"""
Visualize morphological-loss ablation models on validation-set tiles.

Models (all DualBranchSegUNet, only morph sub-terms differ):
  - No Morph (baseline)
  - L_shape only
  - L_shape + L_conn
  - L_shape + L_ms
  - Full Morph (50ep)
  - Full Morph (100ep, best)

Samples are re-selected from the validation set (80/20 split, seed=42),
taking the top-6 highest-algae-coverage tiles per sensor.

Color scheme (same as model comparison):
  Green=TP, Blue=TN, Red=Error (FP+FN unified). No green border.
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
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from running.dual_branch_seg import DualBranchSegUNet

S2_DIR = PROJECT_ROOT / 'dataset3.0' / 's2_filtered_3pct'
L8_DIR = PROJECT_ROOT / 'dataset3.0' / 'l8_algae_256_filtered'
CKPT_DIR = PROJECT_ROOT / 'checkpoints' / 'experiments'
OUT_DIR = PROJECT_ROOT / 'visualization' / 'output'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Morph ablation models (all DualBranchSegUNet)
MODELS = [
    ('No Morph', 'Exp-Morph-Ablation-NoMorph'),
    ('L_shape', 'Exp-Morph-Ablation-ShapeOnly'),
    ('L_shape+L_conn', 'Exp-Morph-Ablation-ShapeConn'),
    ('L_shape+L_ms', 'Exp-Morph-Ablation-ShapeMS'),
    ('Full', 'Exp-Dual-Asym-100ep'),
]

N_TILES = 6

# User-specified tiles (S2 and L8)
S2_FILES = [
    '20190608T023551_20190608T023945_T51SUU_tile-0000009984-0000004352.tif',
    '20210622T023549_20210622T024418_T51SUV_tile-0000009216-0000008192.tif',
    '20190608T023551_20190608T023945_T51SUU_tile-0000000768-0000002304.tif',
]
L8_FILES = [
    'LC08_119036_20210623_tile-0000006656-0000003072_0_0.tif',
    'LC08_119036_20210623_tile-0000006656-0000002560_0_256.tif',
    'LC08_119036_20210623_tile-0000009728-0000003584_256_256.tif',
]


def get_val_files():
    """Validation split: 80/20, seed=42 (same as training/eval)."""
    s2_all = sorted(S2_DIR.glob('*.tif'))
    l8_all = sorted(L8_DIR.glob('*.tif'))
    np.random.seed(42)
    indices = np.random.permutation(len(s2_all))
    val_size = int(len(s2_all) * 0.2)
    val_indices = set(indices[:val_size])
    s2_val = [f for i, f in enumerate(s2_all) if i in val_indices]
    l8_val = [l8_all[i % len(l8_all)] for i in indices[:val_size]]
    return s2_val, l8_val


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
    rgb_idx = [3, 2, 1]
    rgb_raw = np.stack([img_data[i] for i in rgb_idx], axis=-1)
    rgb_raw = np.clip(rgb_raw, 0, None)
    for b in range(3):
        vmin, vmax = np.percentile(rgb_raw[..., b], (2, 98))
        rgb_raw[..., b] = np.clip((rgb_raw[..., b] - vmin) / (vmax - vmin + 1e-8), 0, 1)
    low, high = np.percentile(img_data, (0.5, 99.5))
    denom = np.float32(high - low + 1e-8)
    img_norm = np.clip((img_data - np.float32(low)) / denom, 0, 1).astype(np.float32)
    return (torch.from_numpy(img_norm),
            torch.clamp(torch.from_numpy(label_data), 0, 1),
            rgb_raw, n_ch)


def build_model(ckpt_path):
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


def predict(model, img_tensor, n_ch):
    img_gpu = img_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        if n_ch == 11:  # S2 -> MSI branch
            dummy = torch.zeros(1, 7, 256, 256).to(DEVICE)
            pred = model(img_gpu, dummy)['head1_msi_seg']
        else:           # L8 -> OLI branch
            dummy = torch.zeros(1, 11, 256, 256).to(DEVICE)
            pred = model(dummy, img_gpu)['head2_oli_seg']
    return (pred.cpu() > 0.5).float().squeeze().numpy()


def colorcode(pred_bin, label_bin):
    h, w = pred_bin.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    tp = (pred_bin == 1) & (label_bin == 1)
    tn = (pred_bin == 0) & (label_bin == 0)
    fp = (pred_bin == 1) & (label_bin == 0)
    fn = (pred_bin == 0) & (label_bin == 1)
    rgb[tp] = [0, 255, 0]
    rgb[tn] = [0, 0, 255]
    rgb[fp] = [255, 0, 0]
    rgb[fn] = [255, 0, 0]  # unified error red
    return rgb


def iou_score(pred, gt):
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    eps = 1e-6
    return float((tp + eps) / (tp + fp + fn + eps))


def select_tiles(files, sensor, n=N_TILES):
    """Pick top-n highest-algae-coverage tiles from the val set."""
    scored = []
    for fp in tqdm(files, desc=f'Scan {sensor}', leave=False):
        img, label, rgb, n_ch = load_tile(fp)
        cov = (label > 0.5).float().mean().item()
        scored.append((cov, fp))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [fp for cov, fp in scored[:n]]


def main():
    # Build all morph models
    print("Building models...")
    models = {}
    for name, ckpt in MODELS:
        models[name] = build_model(CKPT_DIR / ckpt / 'best.pth')
        print(f"  OK  {name}")

    for sensor, subfolder, fnames in [('S2', 's2', S2_FILES), ('L8', 'l8', L8_FILES)]:
        base_dir = S2_DIR if sensor == 'S2' else L8_DIR
        tiles = [base_dir / f for f in fnames]
        n_tiles = len(tiles)
        print(f"\nUsing {n_tiles} specified tiles for {sensor}...")

        n_models = len(MODELS)
        cols = n_models + 2  # RGB + GT + models
        fig, axes = plt.subplots(n_tiles, cols, figsize=(2.6 * cols, 3.0 * n_tiles))

        headers = ['RGB', 'GT'] + [m[0] for m in MODELS]
        for ci, h in enumerate(headers):
            axes[0, ci].set_title(h, fontsize=8, fontweight='bold')

        for row_i, fp in enumerate(tiles):
            img, label, rgb, n_ch = load_tile(fp)
            lbl = (label > 0.5).float().squeeze().numpy()
            cov = lbl.mean() * 100

            axes[row_i, 0].imshow(rgb)
            axes[row_i, 1].imshow(lbl, cmap='gray', vmin=0, vmax=1)
            if row_i == 0:
                axes[row_i, 0].set_ylabel('Tile', fontsize=8)
            axes[row_i, 0].set_xlabel(f'Algae {cov:.1f}%', fontsize=7)

            for mi, (name, ckpt) in enumerate(MODELS):
                pred = predict(models[name], img, n_ch)
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

        fig.suptitle(f'Morphological Loss Ablation on {sensor} Val Tiles\n'
                     f'(Green=TP  Blue=TN  Red=Error)',
                     fontsize=14, fontweight='bold', y=0.998)
        plt.tight_layout()
        out_path = OUT_DIR / f'morph_ablation_{subfolder}.png'
        fig.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out_path}')

    print(f'\nDone. Files in: {OUT_DIR}')


if __name__ == '__main__':
    main()
