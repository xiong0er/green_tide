# Green Tide — Multi-Sensor Algal Bloom Segmentation

Pixel-level binary segmentation of filamentous green-tide algae (*Ulva prolifera*)
in the Yellow Sea from **Sentinel-2 (S2, 11-band)** and **Landsat-8 (L8, 7-band)**
imagery, with a single model handling both sensors simultaneously.

The core contribution is **DualBranchSegUNet**: an asymmetric dual-branch
encoder (CrossScan Mamba for the high-SNR S2 branch, windowed Transformer for the
low-SNR L8 branch) with a shared decoder and segmentation head, plus a set of
auxiliary components — hard Top-K ECA band selection, a compactness
morphological loss, and an algae-presence classifier — that together close the
S2–L8 performance gap.

| Sensor | IoU | |
|---|---|---|
| Sentinel-2 (per-image / global) | **0.9225 / 0.9295** | single model, both sensors |
| Landsat-8 non-empty (per-image / global) | **0.9006 / 0.8917** | empty tiles use Specificity |
| S2–L8 gap | **0.022** (per-image) | best single model |

---

## Highlights

- **Asymmetric encoder design** — CrossScan (4-direction SSM) for S2,
  WindowTransformer for L8. Symmetric designs force a trade-off; the asymmetric
  design captures complementary strengths (see `RESULTS.md` for the ablation).
- **Five core components** (see `model/`):
  1. **ECA with hard Top-K** (`model/eca.py`) — per-channel band importance with
     physical Top-K truncation + annealed noise for band exploration.
  2. **Asymmetric encoders** — MSI uses `FastPixelUnshuffle` (Triton space-to-depth,
     `model/pixel_unshuffle.py`); OLI uses strided Conv3×3.
  3. **WVM2B / CrossScan / WindowTransformer** (`model/vertical_mamba.py`) — windowed
     bidirectional Mamba-2 SSM (top↔bottom, and 4-directional CrossScan) and a
     Swin-style windowed transformer.
  4. **TopKAttentionBlock** — windowed multi-head attention retaining only the
     top-*r* fraction of scores before softmax.
  5. **Compactness morphological loss** (`loss/morphological.py`) —
     `L_morph = L_shape + 0.3·L_conn + 0.1·L_ms`, added with weight 0.01.
     (+73% IoU improvement over no morphological loss, zero inference cost.)
- **Custom Mamba-2 SSD** (`model/mamba_2.py`) — no `mamba_ssm` dependency.
- **Multi-sensor baselines** (`running/multi_sensor_baselines.py`) — from-scratch
  reimplementations of SEnSeI-style spectral-set and AnySat-style sensor-adapter
  patterns for controlled comparison.

---

## Directory Structure

```
running/          # training scripts, models, data loaders
  train_dual_branch_pure_seg.py   # main dual-branch training (DualBranchSegUNet)
  train_pure_seg.py               # single-branch / routed / multi-sensor training
  train_single_sensor_baselines.py# classic segmentation baselines (U-Net, ...)
  dual_branch_seg.py              # DualBranchSegUNet
  unet_model.py                   # MSI/OLI branch encoders
  routed_unet.py                  # RoutedSingleUNet (per-sample routing)
  multi_sensor_baselines.py       # SEnSeI / AnySat style baselines
  msi_single_unet.py              # MSISingleUNet (single sensor)
  data_loader.py                  # datasets + dataloaders
  quick_eval.py / eval_*.py       # per-sensor evaluation (S2 IoU, L8 non-empty IoU, empty Specificity)
model/            # reusable neural-network modules
  eca.py, vertical_mamba.py, mamba_2.py, pixel_unshuffle.py, topk.py, dyt.py, sft.py
loss/             # loss functions
  morphological.py               # compactness + skeleton + multi-scale morphological loss
experiments/      # experiment plans, configs, results tracking
  EXPERIMENT_PLAN.md            # full experiment log
visualization/    # qualitative comparison plots
  plot_model_comparison.py      # 8-model side-by-side comparison
  plot_morph_ablation.py        # morphological-loss ablation
```

---

## Installation

```bash
# 1. Install PyTorch for your CUDA version first (see requirements.txt)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Install the rest
pip install -r requirements.txt
```

> **Note on Triton**: `FastPixelUnshuffle` (MSI encoder) uses a Triton kernel.
> If Triton is unavailable on your platform, the OLI branch and routed/single
> models run with strided convs; the MSI PixelUnshuffle path is the only part
> that needs it.

> **Note on `torch.compile`**: training scripts disable `torch.compile`
> (`TORCH_COMPILE_DISABLE=1`) because `torch.compile` + Triton produces NaNs on
> this codebase, and bfloat16 AMP NaNs with the morphological loss + WVM2B.

---

## Data Preparation

Each GeoTIFF stores spectral bands as channels with the binary label in the last
channel (0/1 mask).

| Dataset | Path | Files | Channels | Note |
|---|---|---|---|---|
| S2 filtered | `dataset3.0/s2_filtered_3pct/` | 1941 TIF | 11 data + 1 label | ≤3% cloud, algae-positive |
| L8 filtered | `dataset3.0/l8_algae_256_filtered/` | 1170 TIF | 7 data + 1 label | **49.1% empty** — use `--filter_empty_labels` |

The datasets are **not included** in this repository (see `.gitignore`). Download
or pre-process them into the `dataset3.0/` layout before training.

---

## Training

### Best config — DualBranchSegUNet (asymmetric CrossScan + WinTrans)

```bash
python -m running.train_dual_branch_pure_seg --exp_id Exp-Dual-Asym-100ep \
    --msi_dir dataset3.0/s2_filtered_3pct --oli_dir dataset3.0/l8_algae_256_filtered \
    --msi_channels 11 --oli_channels 7 --eca_topk_msi 6 --eca_topk_oli 4 \
    --hidden_dim 1024 --num_mamba_layers 2 --cross_scan_layers 2 --deep_topk_rate 0.15 \
    --transformer_layers_oli 2 \
    --drop_l8_shallow_skips 1 --use_algae_classifier --algae_cls_weight 0.5 \
    --use_morph --morph_weight 0.01 --empty_l8_weight 0.3 --l8_nonempty_weight 2.0 \
    --epochs 100 --batch_size 2 --lr 0.001
```

### Single-branch routed S2+L8 (Exp-1G style)

```bash
python -m running.train_pure_seg --exp_id Exp-Mixed --mixed_data \
    --s2_dir dataset3.0/s2_filtered_3pct --l8_dir dataset3.0/l8_algae_256_filtered \
    --msi_channels 11 --oli_channels 7 --eca_topk_msi 6 --eca_topk_oli 4 \
    --hidden_dim 512 --num_mamba_layers 2 --epochs 50 --batch_size 4 \
    --use_morph --morph_weight 0.01 --filter_empty_labels
```

### Classic baselines

```bash
python -m running.train_single_sensor_baselines --model vanilla_unet ...
python -m running.train_single_sensor_baselines --model attention_unet ...
```

### Multi-sensor baselines (SEnSeI / AnySat style)

```bash
python -m running.train_pure_seg --exp_id Exp-SEnSeI --mixed_data \
    --multi_sensor_model spectral_set --hidden_dim 512 ...
python -m running.train_pure_seg --exp_id Exp-AnySat --mixed_data \
    --multi_sensor_model sensor_adapter --hidden_dim 512 ...
```

Full experiment commands are documented in `experiments/EXPERIMENT_PLAN.md`.

---

## Evaluation

Per-sensor evaluation (S2 IoU, L8 non-empty IoU, L8 empty Specificity):

```bash
# Dual-branch model
python -m running.quick_eval --checkpoint checkpoints/experiments/<exp_id>/best.pth --val_only

# Full-metric re-evaluation (accuracy/precision/recall/F1/IoU)
python -m running.reeval_full_metrics
```

Empty L8 tiles are evaluated with **Specificity** (background-pixel accuracy)
instead of IoU, because IoU is ill-defined for all-zero predictions vs all-zero
ground truth.

---

## Visualization

```bash
# 8-model comparison (Ours / U-Net / SEnSeI / AnySat / AlgaeMamba / Swin / Attn / U-Net++)
python visualization/plot_model_comparison.py

# Morphological-loss ablation
python visualization/plot_morph_ablation.py
```

Color code: **green** = TP (correct algae), **blue** = TN (correct background),
**red** = error (false-alarm + missed-algae, unified).

---

## Key Results

Full tables and ablation details: **`RESULTS.md`** and
`experiments/EXPERIMENT_PLAN.md`.

### Main model vs classic baselines (single-sensor)

| Model | Params | S2 IoU | L8 NE IoU | L8 Empty Spec |
|---|---|---|---|---|
| U-Net | 13.4M | 0.9498 | 0.7871 | 1.0000 |
| Attn U-Net | 13.8M | 0.9336 | 0.8124 | 1.0000 |
| Swin U-Net | 35.2M | 0.8560 | 0.6084 | 1.0000 |
| AlgaeMamba | 25.7M | 0.9002 | 0.7554 | 0.9972 |
| **Ours (DualBranchSegUNet)** | **99.2M** | **0.9295** | **0.8917** | **0.8157** |

*One model replaces two single-sensor models — S2 within 2.1% of the best
single-sensor U-Net, L8 ahead of Attn U-Net by 9.7%.*

### Multi-sensor handling (hidden=512, 50ep)

| Model | Strategy | S2 IoU | L8 NE IoU |
|---|---|---|---|
| RoutedSingleUNet | per-sensor encoder routing | 0.8618 | 0.6780 |
| SensorAdapter (AnySat-style) | private projection + shared backbone | **0.8963** | **0.7599** |
| SpectralSet (SEnSeI-style) | wavelength-conditioned set encoder | 0.8576 | 0.7249 |

### Encoder ablation (DualBranchSegUNet, 50ep)

| MSI / OLI | S2 IoU | L8 NE IoU | Gap |
|---|---|---|---|
| CrossScan / CrossScan | 0.8734 | 0.7649 | 0.1084 |
| WinTrans / WinTrans | **0.9301** | **0.8599** | **0.0702** |
| CrossScan / WinTrans (asymmetric) | 0.9112 | 0.8306 | 0.0805 |

*At 50 epochs WinTrans converges fastest; the asymmetric CrossScan/WinTrans
design wins at 100 epochs (see RESULTS.md for the convergence analysis).*

---

## License

[MIT](./LICENSE). See `LICENSE` for details.

## Citation

If you use this work, please cite the corresponding paper (add the citation once
published).
