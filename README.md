# Mixed-Precision Quantization Framework (W=A Co-Optimization)

Layer-wise mixed-precision quantization with joint Weight=Activation (W=A) constraint. The engine automatically profiles each layer's sensitivity, searches for the optimal bit-width assignment under a target accuracy drop, verifies compliance, and falls back to QAT if needed — all in a single command.

---

## Requirements

- Python 3.9+
- PyTorch 2.0+ with CUDA
- torchvision, timm, tqdm, pandas, numpy

```bash
pip install torch torchvision timm tqdm pandas numpy
```

---

## Repository Structure

```
├── models/                          # Pre-trained checkpoints (not tracked by git)
├── quantization_framework/
│   ├── models/                      # Model definitions (LeViT, ResNet, Swin, VGG)
│   ├── quantization/                # Quantization primitives and activation quantizers
│   ├── evaluation/                  # Data loaders and accuracy evaluation pipeline
│   ├── search/                      # Greedy search algorithm
│   ├── export/                      # Model compression utilities
│   └── experiments/
│       ├── auto_quantize_engine_joint.py   # Main entry point
│       ├── joint_sensitivity.py            # W=A sensitivity profiling
│       ├── joint_search.py                 # Greedy bit-width search
│       ├── verify_wa_constraint.py         # W=A compliance verification
│       ├── validate_config.py              # PTQ accuracy validation
│       └── qat_training.py                 # Quantization-aware training (fallback)
```

---

## Quick Start

Run the full pipeline with a single command:

```bash
python quantization_framework/experiments/auto_quantize_engine_joint.py \
  --model <model_name> \
  --checkpoint <path/to/checkpoint> \
  --dataset <dataset> \
  --bits 2 4 8
```

**Examples:**

```bash
# LeViT on CIFAR-10
python quantization_framework/experiments/auto_quantize_engine_joint.py \
  --model levit \
  --checkpoint models/best3_levit_model_cifar10.pth \
  --dataset cifar10 --bits 2 4 8

# Swin Transformer on CIFAR-100
python quantization_framework/experiments/auto_quantize_engine_joint.py \
  --model swin \
  --checkpoint models/best_swin_model_cifar_changed.pth \
  --dataset cifar100 --bits 2 4 8

# ResNet-18 on GTSRB
python quantization_framework/experiments/auto_quantize_engine_joint.py \
  --model resnet \
  --checkpoint models/road_0.9994904891304348.pth \
  --dataset gtsrb --bits 2 4 8

# VGG-11 (BN) on CIFAR-10
python quantization_framework/experiments/auto_quantize_engine_joint.py \
  --model vgg11_bn \
  --checkpoint models/vgg11_bn.pt \
  --dataset cifar10 --bits 2 4 8
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `--model` | Model: `levit`, `resnet`, `swin`, `vgg11_bn` | required |
| `--checkpoint` | Path to pre-trained `.pth` / `.pt` checkpoint | required |
| `--dataset` | Dataset: `cifar10`, `cifar100`, `gtsrb` | required |
| `--bits` | Candidate bit-widths (space-separated) | required |
| `--target-drop` | Max allowed accuracy drop (%) for the search | `3.0` |
| `--qat-threshold` | Accuracy drop (%) above which QAT is triggered | `5.0` |

### Datasets

- **CIFAR-10 / CIFAR-100**: downloaded automatically to `./data/` on first run.
- **GTSRB**: download manually from [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) and place under `./data/gtsrb/` with `Train/`, `Test/` subdirectories and `Test.csv`.

### Checkpoints

Place model `.pth` / `.pt` files in `models/`. The directory is gitignored.

---

## Pipeline

The engine runs the following steps automatically:

**Step 1 — Joint sensitivity profiling**: each layer is individually quantized at each candidate bit-width with W=A enforced, and the accuracy drop vs FP32 is recorded.

**Step 2 — Greedy search**: layers are sorted from least to most sensitive. Bit-widths are reduced greedily until the estimated cumulative accuracy drop approaches `--target-drop`.

**Step 2.5 — W=A constraint verification**: confirms every layer in the generated config has matching weight and activation bits.

**Step 3 — PTQ gate**: applies the config post-training and measures actual accuracy on the full validation set. If the drop is within `--qat-threshold`, the model is accepted.

**Step 5 — QAT recovery** (only if PTQ fails): fine-tunes the quantized model for up to 15 epochs with early stopping (patience = 5). Best validation checkpoint is saved.

**Step 6 — BOPs calculation**: reports Bit Operations for both FP32 baseline and the quantized model.

### Outputs

| File | Description |
|---|---|
| `<model>_sensitivity_<bits>.csv` | Per-layer W=A sensitivity profile |
| `<model>_config_<bits>_weight.json` | Per-layer weight bit-widths |
| `<model>_config_<bits>_activation.json` | Per-layer activation bit-widths |
| `<model>_config_<bits>.json` | Joint W=A config (reference) |
| `metrics.json` | Final accuracy, BOPs, and W=A compliance summary |

---

## Results

All runs use bits `{2, 4, 8}` with a 3% target accuracy drop.

| Model | Dataset | Params | Baseline | Final Acc | Avg Bits | BOPs Reduction | Outcome |
|---|---|---|---|---|---|---|---|
| VGG-11-BN | CIFAR-10 | 28.1M | 92.77% | 91.11% | 5.45 | **48.78×** | PTQ passed |
| ResNet-18 | GTSRB | 11.2M | 99.51% | 95.34% | 5.71 | **32.28×** | QAT (best val) |
| LeViT | CIFAR-10 | 37.6M | 97.82% | 97.17% | 7.05 | **16.01×** | QAT (best val) |
| Swin-T | CIFAR-100 | 27.6M | 89.65% | 86.23% | 7.62 | **16.28×** | PTQ passed |

### Bit-Width Distributions (W=A pairs)

| Model | W8/A8 | W4/A4 | W2/A2 |
|---|---|---|---|
| VGG-11-BN | 36.4% (4/11) | 63.6% (7/11) | — |
| ResNet-18 | 47.6% (10/21) | 42.9% (9/21) | 9.5% (2/21) |
| LeViT | 77.8% (49/63) | 19.0% (12/63) | 3.2% (2/63) |
| Swin-T | 90.6% (48/53) | 9.4% (5/53) | — |

W=A constraint compliance: **100%** across all models and layers.

---

## Notes

- Stem convolutions and downsampling projection layers are automatically assigned higher bit-widths by the greedy search — these are the most sensitivity-critical layers.
- LeViT and Swin use reduced batch sizes automatically to fit within GPU memory constraints.
- GTSRB and CIFAR inputs are upsampled to 224×224 for transformer-based models; VGG processes at native 32×32.
- QAT uses a cosine-annealed learning rate with early stopping. The best validation checkpoint is saved as `<model>_qat_best.pth`.
