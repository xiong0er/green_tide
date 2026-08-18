"""Quick per-sensor eval — proven working standalone approach"""
import sys, torch, numpy as np, rasterio, json
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from running.dual_branch_seg import DualBranchSegUNet


def build_model(config):
    return DualBranchSegUNet(
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
        transformer_layers_oli=config.get('transformer_layers_oli', config.get('transformer_layers', 0)),
        deep_topk_rate=config.get('deep_topk_rate', None),
        eca_mode=config.get('eca_mode', 'topk'),
        disable_eca=config.get('disable_eca', False),
    )


def load_image(filepath):
    with rasterio.open(filepath) as src:
        data = src.read()
    c, h, w = data.shape
    # Valid pixel mask: pixels where ALL spectral bands are non-NaN & finite BEFORE nan_to_num
    raw_data = data[:-1]  # spectral bands only, not label
    valid_mask = np.all(np.isfinite(raw_data), axis=0).astype(np.float32)  # [H, W]
    # Now safe to replace NaN with 0 for model input
    data = np.nan_to_num(data, nan=0.0)
    target = 256
    # Separate data and label before resize
    img_data = data[:-1].astype(np.float32)
    label_data = data[-1:].astype(np.float32)
    if h != target or w != target:
        from scipy.ndimage import zoom
        zoom_h = target / h
        zoom_w = target / w
        # Data: bilinear (order=1), Label: nearest-neighbor (order=0), Mask: nearest
        img_data = zoom(img_data, (1, zoom_h, zoom_w), order=1)
        label_data = zoom(label_data, (1, zoom_h, zoom_w), order=0)
        valid_mask = zoom(valid_mask, (zoom_h, zoom_w), order=0)
    # Match training: global percentile normalization across all bands
    low, high = np.percentile(img_data, (0.5, 99.5))
    denom = np.float32(high - low + 1e-8)
    img_data = np.clip((img_data - np.float32(low)) / denom, 0, 1).astype(np.float32)
    img = torch.from_numpy(img_data)
    label = torch.from_numpy(label_data)
    return img, torch.clamp(label, 0, 1), torch.from_numpy(valid_mask)


def compute_metrics(pred_bin, label_bin, valid_mask=None):
    """Compute metrics. If valid_mask is provided, only count on valid pixels."""
    if valid_mask is not None:
        mask = valid_mask > 0.5
        pred_bin = pred_bin * mask
        label_bin = label_bin * mask
    has_algae = label_bin.sum() > 0
    tp = (pred_bin * label_bin).sum().item()
    tn = ((1 - pred_bin) * (1 - label_bin)).sum().item()
    fp = (pred_bin * (1 - label_bin)).sum().item()
    fn = ((1 - pred_bin) * label_bin).sum().item()
    eps = 1e-6
    valid_px = int(mask.sum().item()) if valid_mask is not None else (256 * 256)
    return {
        'has_algae': bool(has_algae),
        'iou': float((tp + eps) / (tp + fp + fn + eps)),
        'specificity': float((tn + eps) / (tn + fp + eps)),
        'precision': float((tp + eps) / (tp + fp + eps)),
        'recall': float((tp + eps) / (tp + fn + eps)),
        'f1': float((2*(tp+eps)/(tp+fp+eps)*(tp+eps)/(tp+fn+eps) + eps) / ((tp+eps)/(tp+fp+eps)+(tp+eps)/(tp+fn+eps) + eps)),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'valid_pixels': valid_px,
    }


def summarize(results, label):
    nonempty = [r for r in results if r['has_algae']]
    empty = [r for r in results if not r['has_algae']]
    all_items = results

    print(f"\n{'='*60}")
    print(f"  {label}  (total={len(all_items)}, nonempty={len(nonempty)}, empty={len(empty)})")
    print(f"{'='*60}")

    for name, subset in [("All", all_items), ("NonEmpty", nonempty), ("Empty", empty)]:
        if not subset:
            continue
        ious = [r['iou'] for r in subset]
        specs = [r['specificity'] for r in subset]
        precs = [r['precision'] for r in subset]
        recs = [r['recall'] for r in subset]
        f1s = [r['f1'] for r in subset]
        print(f"\n  {name} ({len(subset)} tiles):")
        print(f"    IoU:         {np.mean(ious):.4f} ± {np.std(ious):.4f}")
        print(f"    Specificity: {np.mean(specs):.4f} ± {np.std(specs):.4f}")
        print(f"    Precision:   {np.mean(precs):.4f} ± {np.std(precs):.4f}")
        print(f"    Recall:      {np.mean(recs):.4f} ± {np.std(recs):.4f}")
        print(f"    F1:          {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # Global pixel-level
    total_tp = sum(r['tp'] for r in all_items)
    total_tn = sum(r['tn'] for r in all_items)
    total_fp = sum(r['fp'] for r in all_items)
    total_fn = sum(r['fn'] for r in all_items)
    eps = 1e-6
    g_iou = (total_tp + eps) / (total_tp + total_fp + total_fn + eps)
    g_spec = (total_tn + eps) / (total_tn + total_fp + eps)
    g_prec = (total_tp + eps) / (total_tp + total_fp + eps)
    g_rec = (total_tp + eps) / (total_tp + total_fn + eps)
    g_f1 = (2 * g_prec * g_rec + eps) / (g_prec + g_rec + eps)

    print(f"\n  Global (pixel-level):")
    print(f"    IoU:         {g_iou:.4f}")
    print(f"    Specificity: {g_spec:.4f}")
    print(f"    Precision:   {g_prec:.4f}")
    print(f"    Recall:      {g_rec:.4f}")
    print(f"    F1:          {g_f1:.4f}")

    if nonempty:
        ne_tp = sum(r['tp'] for r in nonempty)
        ne_fp = sum(r['fp'] for r in nonempty)
        ne_fn = sum(r['fn'] for r in nonempty)
        ne_iou = (ne_tp + eps) / (ne_tp + ne_fp + ne_fn + eps)
        ne_prec = (ne_tp + eps) / (ne_tp + ne_fp + eps)
        ne_rec = (ne_tp + eps) / (ne_tp + ne_fn + eps)
        ne_f1 = (2 * ne_prec * ne_rec + eps) / (ne_prec + ne_rec + eps)
        print(f"\n  NonEmpty Global (pixel-level):")
        print(f"    IoU:       {ne_iou:.4f}")
        print(f"    Precision: {ne_prec:.4f}")
        print(f"    Recall:    {ne_rec:.4f}")
        print(f"    F1:        {ne_f1:.4f}")
    else:
        ne_iou = 0

    # Empty specificity
    if empty:
        e_tn = sum(r['tn'] for r in empty)
        e_fp = sum(r['fp'] for r in empty)
        e_spec = (e_tn + eps) / (e_tn + e_fp + eps)
        print(f"\n  Empty Specificity: {e_spec:.4f}")
    else:
        e_spec = 0

    return {
        'global_iou': float(g_iou),
        'global_iou_nonempty': float(ne_iou) if nonempty else 0,
        'global_specificity': float(g_spec),
        'empty_specificity': float(e_spec),
        'global_f1': float(g_f1),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, default=str(PROJECT_ROOT / 'checkpoints' / 'experiments' / 'Exp-Dual-WVM2B-100ep' / 'best.pth'))
    ap.add_argument('--output', type=str, default=None)
    ap.add_argument('--s2_dir', type=str, default=str(PROJECT_ROOT / 'dataset3.0' / 's2_filtered_3pct'))
    ap.add_argument('--l8_dir', type=str, default=str(PROJECT_ROOT / 'dataset3.0' / 'l8_algae_256_filtered'))
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--val_only', action='store_true',
                    help='仅评估验证集 (80/20 split, seed=42)')
    ap.add_argument('--val_split', type=float, default=0.2,
                    help='验证集比例 (默认 0.2)')
    ap.add_argument('--seed', type=int, default=42,
                    help='数据划分随机种子 (默认 42)')
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt_path = args.checkpoint
    s2_dir = Path(args.s2_dir)
    l8_dir = Path(args.l8_dir)
    exp_id = Path(ckpt_path).parent.name
    out_path = args.output or str(PROJECT_ROOT / 'logs' / 'experiments' / exp_id / 'eval_metrics.json')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoint...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state_dict'].items()}
    config = ckpt['config']
    model = build_model(config)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    print(f"  Epoch: {ckpt['epoch']}, Best Loss: {ckpt['best_loss']}")

    # Data split (same logic as training)
    s2_files = sorted(s2_dir.glob('*.tif'))
    l8_files = sorted(l8_dir.glob('*.tif'))
    if args.val_only:
        np.random.seed(args.seed)
        indices = np.random.permutation(len(s2_files))
        val_size = int(len(s2_files) * args.val_split)
        val_indices = set(indices[:val_size])
        s2_files = [f for i, f in enumerate(s2_files) if i in val_indices]
        # L8: match same indices (cycling as in training)
        l8_files = [l8_files[i % len(l8_files)] for i in indices[:val_size]]
        print(f"Validation split: {len(s2_files)} S2 + {len(l8_files)} L8 tiles (seed={args.seed}, ratio={args.val_split})")
    else:
        print(f"Full dataset: {len(s2_files)} S2 + {len(l8_files)} L8 tiles")

    # S2 eval
    print(f"\nEvaluating {len(s2_files)} S2 tiles...")
    s2_results = []
    for fp in tqdm(s2_files, desc='S2'):
        img, label, valid_mask = load_image(fp)
        img = img.unsqueeze(0).to(device)
        dummy_oli = torch.zeros(1, 7, 256, 256).to(device)
        with torch.no_grad():
            seg = model(img, dummy_oli)
            pred = seg['head1_msi_seg']
        pred_bin = (pred > 0.5).float().cpu()
        label_bin = (label > 0.5).float()
        s2_results.append(compute_metrics(pred_bin, label_bin, valid_mask))

    s2_summary = summarize(s2_results, "S2 (MSI) — Sentinel-2")

    # L8 eval
    print(f"\nEvaluating {len(l8_files)} L8 tiles...")
    l8_results = []
    for fp in tqdm(l8_files, desc='L8'):
        img, label, valid_mask = load_image(fp)
        img = img.unsqueeze(0).to(device)
        dummy_msi = torch.zeros(1, 11, 256, 256).to(device)
        with torch.no_grad():
            seg = model(dummy_msi, img)
            pred = seg['head2_oli_seg']
        pred_bin = (pred > 0.5).float().cpu()
        label_bin = (label > 0.5).float()
        l8_results.append(compute_metrics(pred_bin, label_bin, valid_mask))

    l8_summary = summarize(l8_results, "L8 (OLI) — Landsat-8")

    # Final comparison
    s2_giou = s2_summary['global_iou']
    l8_ne_iou = l8_summary['global_iou_nonempty']
    gap = s2_giou - l8_ne_iou
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"  S2 IoU (global):           {s2_giou:.4f}")
    print(f"  L8 IoU (global):           {l8_summary['global_iou']:.4f}")
    print(f"  L8 IoU (nonempty only):    {l8_ne_iou:.4f}")
    print(f"  L8 Specificity (global):   {l8_summary['global_specificity']:.4f}")
    print(f"  L8 Empty Specificity:      {l8_summary['empty_specificity']:.4f}")
    print(f"  S2-L8 gap (nonempty):      {gap:.4f}")

    # Save
    output = {
        'checkpoint': ckpt_path,
        'config': {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v for k, v in config.items()},
        'epoch': ckpt['epoch'],
        'best_loss': ckpt['best_loss'],
        's2': {'summary': s2_summary, 'num_tiles': len(s2_results)},
        'l8': {'summary': l8_summary, 'num_tiles': len(l8_results)},
        'gap': gap,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
