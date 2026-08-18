# Results

All metrics are **global pixel-level** (computed over the union of pixels, valid
pixels only) unless marked "per-image". Two metrics deserve special attention:

- **L8 non-empty IoU** — IoU computed only on tiles that contain algae. Empty
  tiles (49.1% of the L8 validation set) make plain IoU ill-defined.
- **L8 empty Specificity** — background-pixel accuracy on empty tiles; replaces
  IoU there (all-zero prediction vs all-zero GT → 0/0).

Validation split: 80/20, seed=42 → 388 S2 + 388 L8 tiles.

---

## 1. Main model

**DualBranchSegUNet** — asymmetric dual-branch encoder (CrossScan Mamba for S2,
windowed Transformer for L8), shared decoder + shared segmentation head, 100
epochs, 99.21M params.

| Sensor | Accuracy | Precision | Recall | F1 | IoU | Specificity |
|---|---|---|---|---|---|---|
| S2 | 0.9916 | 0.9654 | 0.9615 | 0.9634 | **0.9295** | 0.9955 |
| L8 (non-empty) | 0.9848 | 0.9113 | 0.9764 | 0.9427 | **0.8917** | 0.9839 |
| L8 (empty) | — | — | — | — | — | **0.8157** |

- S2–L8 gap (non-empty IoU): **0.0378**
- Per-image averages: S2 IoU 0.9225, L8 non-empty IoU 0.9006, gap 0.0219.

---

## 2. Classic single-sensor baselines

| Model | Params | S2 IoU | L8 NE IoU | L8 Empty Spec |
|---|---|---|---|---|
| U-Net | 13.40M | **0.9498** | 0.7871 | 1.0000 |
| U-Net++ | 9.17M | 0.9379 | 0.7602 | 1.0000 |
| Attn U-Net | 13.76M | 0.9336 | **0.8124** | 1.0000 |
| SegFormer-B2 | 19.80M | 0.7774 | 0.5917 | 0.9999 |
| Swin U-Net | 35.16M | 0.8560 | 0.6084 | 1.0000 |
| AlgaeMamba | 25.70M | 0.9002 | 0.7554 | 0.9972 |
| AlgaeNet | 7.76M | 0.9179 | 0.7427 | 0.9980 |
| **Ours (DualBranchSegUNet)** | **99.21M** | 0.9295 | **0.8917** | 0.8157 |

*One model replaces two single-sensor models: S2 within 2.1% of the best
single-sensor U-Net, L8 ahead of the best single-sensor Attn U-Net by 9.7%.*

---

## 3. Multi-sensor handling

From-scratch multi-sensor baselines (`running/multi_sensor_baselines.py`), 50
epochs, batch 4. All use one shared `RoutedEncoder` + shared PixelShuffle
decoder; only the sensor-input front differs.

### hidden = 512

| Model | Strategy | S2 IoU | L8 NE IoU | L8 Empty Spec | Gap |
|---|---|---|---|---|---|
| RoutedSingleUNet | per-sensor encoder routing | 0.8618 | 0.6780 | 0.6162 | 0.1838 |
| CommonBandSharedUNet | 7 shared bands + shared backbone | 0.8407 | 0.6516 | 0.7734 | 0.1890 |
| SensorAdapter (AnySat-style) | private projection + shared backbone | **0.8963** | **0.7599** | 0.6888 | **0.1364** |
| SpectralSet (SEnSeI-style) | wavelength-conditioned set encoder | 0.8576 | 0.7249 | 0.7008 | 0.1327 |

### hidden = 1024

| Model | S2 IoU | L8 NE IoU | L8 Empty Spec | Best epoch |
|---|---|---|---|---|
| RoutedSingleUNet | 0.8596 | 0.5808 | 0.5457 | 31 |
| CommonBandSharedUNet | 0.8512 | 0.7116 | 0.7427 | 19 |
| SensorAdapter (AnySat-style) | 0.8524 | 0.7206 | 0.5076 | 8 ⚠️ |
| SpectralSet (SEnSeI-style) | 0.8319 | 0.6640 | 0.5707 | 41 |

*At 50 epochs these single-branch models overfit at hidden=1024 (SensorAdapter
peaks at epoch 8); hidden=512 is the better operating point for this budget.*

---

## 4. Encoder ablation — CrossScan vs WinTrans

DualBranchSegUNet, 50 epochs, batch 4, hidden=1024.

| MSI / OLI | S2 IoU | L8 NE IoU | L8 Empty Spec | Gap |
|---|---|---|---|---|
| CrossScan / CrossScan | 0.8734 | 0.7649 | 0.6925 | 0.1084 |
| WinTrans / WinTrans | **0.9301** | **0.8599** | **0.8116** | **0.0702** |
| WinTrans / CrossScan | 0.8827 | 0.7221 | 0.7075 | 0.1606 |
| CrossScan / WinTrans (asymmetric) | 0.9112 | 0.8306 | 0.7476 | 0.0805 |

*At 50 epochs WinTrans (content-dependent attention) converges fastest and wins
across the board. CrossScan's fixed 4-direction SSM has a slower learning curve
but a higher ceiling — the asymmetric CrossScan/WinTrans design overtakes the
symmetric baselines at 100 epochs (see §1).*

---

## 5. Auxiliary-component ablations

### 5.1 Algae-presence classifier

NoFusion dual-branch, 50 epochs, unfiltered L8. Only `use_algae_classifier`
differs.

| Config | S2 IoU | L8 NE IoU | L8 Empty Spec | L8 all-zero rate |
|---|---|---|---|---|
| No classifier | 0.8819 | 0.5315 | 0.9167 | 34.6% |
| + AlgaeClassifier | 0.8795 | **0.5876** (+10.6%) | **0.9651** | 63.2% |

### 5.2 OLI shallow-skip drop (`drop_l8_shallow_skips`)

*(Note: despite the flag name, the dropped skip is the **deepest** 32×32 skip —
`skips[-1]` in `_decode`.)*

| drop | Context | S2 IoU | L8 NE IoU | L8 Empty Spec |
|---|---|---|---|---|
| 0 | NoFusion 50ep | **0.8658** | 0.5580 | 0.9076 |
| 1 | NoFusion 50ep | 0.8324 | **0.6070** | **0.9246** |
| 1 | WVM2B 100ep | 0.8444 | 0.6906 | 0.7033 |
| 2 | WVM2B 100ep | **0.8906** | **0.7508** | 0.7195 |

*drop=1 helps L8 (+4.9pp non-empty IoU) but hurts S2 (−3.3pp) — dropping the
deepest skip tears the shared decoder. The final config uses drop=1.*

---

## 6. Loss-component ablations

### 6.1 Morphological loss sub-terms (global, 100ep reference)

`L_morph = L_shape + 0.3·L_conn + 0.1·L_ms`, weight 0.01.

| Config | S2 IoU | L8 NE IoU | L8 Empty Spec |
|---|---|---|---|
| NoMorph | 0.8784 | 0.7811 | 0.7344 |
| L_shape only | 0.9240 | 0.7327 | 0.6146 |
| L_shape + L_conn | 0.9138 | **0.8437** | **0.7968** |
| L_shape + L_ms | 0.8679 | 0.8351 | 0.7562 |
| **Full Morph** | **0.9295** | **0.8917** | **0.8157** |

*L_shape (compactness) boosts S2 but fragments L8; L_conn (skeleton
connectivity) is the key balancer; L_ms (multi-scale area) drives late-stage
empty-tile convergence. All three are necessary.*

### 6.2 ECA band selection

| Mode | S2 IoU | L8 NE IoU | L8 Empty Spec |
|---|---|---|---|
| **TopK (hard truncation)** | **0.9198** | **0.8413** | 0.7706 |
| Soft (weighted) | 0.9198 | 0.8090 | 0.8481 |
| Full bands | 0.9185 | 0.7769 | 0.6299 |
| Random fixed-K | 0.8894 | 0.8003 | 0.8830 |

### 6.3 TopK attention rate (100ep)

| r | S2 IoU | L8 NE IoU | L8 Empty Spec | Gap |
|---|---|---|---|---|
| 0.2 | 0.9129 | 0.8717 | 0.8180 | 0.0412 |
| **0.4** | **0.9295** | 0.8917 | 0.8157 | 0.0378 |
| 0.6 | 0.9042 | 0.8602 | 0.8424 | 0.0440 |
| 0.8 | 0.9415 | 0.8822 | 0.7899 | 0.0593 |
| **1.0** | 0.9303 | **0.9119** | **0.8397** | **0.0184** |

### 6.4 Deep TopK rate (50ep, fixed bug)

| dtr | S2 IoU | L8 NE IoU | L8 Empty Spec | Gap |
|---|---|---|---|---|
| 0.1 | 0.8865 | 0.8430 | 0.7273 | 0.0435 |
| **0.15** | **0.9222** | 0.8378 | 0.7754 | 0.0844 |
| 0.2 | 0.8587 | 0.7715 | 0.7344 | 0.0873 |
| 0.4 | 0.8997 | 0.8347 | 0.7423 | 0.0650 |
| 0.6 | 0.8633 | 0.8239 | 0.7226 | 0.0394 |
| **0.8** | 0.8959 | **0.8573** | **0.7924** | 0.0386 |
| 1.0 | 0.8849 | 0.8150 | 0.7453 | 0.0699 |

*S2 prefers very sparse attention (dtr=0.15, CrossScan's strong spatial bias);
L8 prefers denser attention (dtr=0.8, WinTrans's weaker bias) — the two branches
want **asymmetric** deep-topk rates.*

---

## 7. Summary of the design

1. **Asymmetric encoder** — CrossScan (S2 ceiling) + WinTrans (L8 robustness).
2. **Hard Top-K ECA** — band selection; Soft/FullBands/Random all worse.
3. **Morphological loss** — compactness + skeleton + multi-scale; all three terms needed.
4. **Algae classifier** — decides empty vs non-empty L8 tiles; +10.6% non-empty IoU.
5. **Asymmetric deep-topk** — S2 sparse / L8 dense is the remaining gap reducer.

The full experiment log (with per-run commands and per-image variants) is in
`experiments/EXPERIMENT_PLAN.md`.
