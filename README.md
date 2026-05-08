# FakeShield — Hybrid Deepfake Detection via GAN + Vision Transformer

> A dual-path deepfake detection system combining the artifact-catching power of
> a pretrained **EfficientNet-B4** discriminator with the global-context reasoning
> of a pretrained **ViT-B/16**, achieving up to **95–96% accuracy** on the SDFVD benchmark.

---

## Overview

Modern deepfakes are frighteningly convincing. **FakeShield** fights back by running
two complementary analyses in parallel — one path hunts for pixel-level artifacts
(compression ghosts, blending seams, texture inconsistencies), while the other
reasons about the global structure of a face. Both streams are then fused with a
cross-attention mechanism so each can inform the other before the final verdict.

Two architectures are provided and compared:

| Architecture | Strategy |
|---|---|
| **Sequential** | EfficientNet attention guides ViT toward suspicious regions |
| **Parallel** | Both models run simultaneously; bidirectional cross-attention fuses features |

---

## Architecture

### Sequential Model
```
Input Image
    │
    ├─► EfficientNet-B4 (pretrained) ──► Fine-detail features + Attention map
    │                                              │
    │                  ┌────────────────────────────┘
    │                  ▼
    └─► ViT-B/16 (pretrained) ──► Global context features
                       │
              Feature Fusion (MLP)
                       │
               Final Classification
```

### Parallel Model
```
Input Image
    ├─► EfficientNet-B4 (pretrained) ──► GAN features ──┐
    │                                                    ├─► Cross-Attention ──► Fusion ──► Output
    └─► ViT-B/16 (pretrained)       ──► ViT features ──┘
```

---

## Key Design Choices

| Component | Choice | Why |
|---|---|---|
| CNN backbone | EfficientNet-B4 (ImageNet pretrained) | Strong low-level texture features out of the box |
| ViT backbone | ViT-B/16 (ImageNet pretrained) | Global attention over 196 patches; transfer-learns well |
| Optimizer | AdamW | Better weight decay handling than SGD for transformer models |
| LR schedule | Linear warmup + Cosine annealing | Stable start, smooth decay |
| Backbone LR | 10× smaller than head LR | Fine-tune without destroying pretrained features |
| Loss | Focal BCE + label smoothing | Handles hard negatives; prevents overconfident predictions |
| Auxiliary losses | α = 0.3 on GAN + ViT heads | Extra gradient signal through both paths |

---

## Project Structure

```
FakeShield/
├── models/
│   ├── __init__.py
│   ├── gan_discriminator.py      # EfficientNet-B4 discriminator
│   ├── vision_transformer.py     # ViT-B/16 encoder
│   ├── sequential_model.py       # Sequential GAN→ViT fusion (full model)
│   ├── parallel_model.py         # Parallel GAN‖ViT with cross-attention (full model)
│   └── cached_heads.py           # Head-only models for fast training on cached features
├── utils/
│   ├── __init__.py
│   ├── data_utils.py             # Dataset loading, augmentation, SDFVD support
│   ├── feature_cache.py          # One-time encoder pass → on-disk feature cache
│   └── cached_dataset.py         # Dataset/DataLoader for cached feature vectors
├── extract_features.py           # Run once: encode every frame to disk
├── train_sequential_fast.py      # Fast head-only training (cached features)
├── train_parallel_fast.py        # Fast head-only training (cached features)
├── train_sequential.py           # Legacy full-model two-phase training
├── train_parallel.py             # Legacy full-model two-phase training
├── demo.py                       # Inference & visualisation
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone
git clone <your-repo-url>
cd FakeShield

# 2. Create virtual environment
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Dataset Setup (SDFVD)

Place the SDFVD dataset in the project root:

```
SDFVD/
├── videos_real/    ← real face videos
└── videos_fake/    ← deepfake videos
```

The training script automatically extracts frames and creates an 80/20 train/val split.

---

## Training

There are two training paths. **Use the fast path unless you need to fine-tune
the EfficientNet-B4 / ViT-B/16 backbones themselves.**

### Fast path — encode-then-train (recommended)

The original training loop wastes most of its time re-running the **frozen**
EfficientNet-B4 + ViT-B/16 backbones on the same frames every epoch. Their
output never changes, so we encode each frame **once** and train only the
small fusion + classifier heads on the cached vectors. Result: head training
goes from minutes-per-epoch to seconds-per-epoch.

```bash
# 1. One-time feature extraction (a few minutes on GPU)
python extract_features.py
#    → SDFVD/{train,val} → feature_cache/{train,val}_features.pt
#    Caches K=5 random-augmentation copies per train frame so head training
#    still benefits from augmentation diversity.

# 2. Fast head-only training (seconds per epoch)
python train_sequential_fast.py
python train_parallel_fast.py
#    → checkpoints/{sequential_fast,parallel_fast}/<Model>_best.pth
```

The fast trainers add three accuracy-boosting techniques:
- **Multi-augmentation feature cache** (K=5 copies per train frame).
- **Mixup in feature space** — mixes two samples' encoder vectors with a shared λ.
- **Video-level metric reporting** — averages predictions across each video's
  frames before computing accuracy / F1; the best checkpoint is selected on
  this metric (closer to deployment than per-frame accuracy).

If you change the dataset or image size, regenerate the cache:
```bash
python extract_features.py --force
```

### Full fine-tuning path (legacy)

If you want to also fine-tune the backbone weights, the original two-phase
trainers still work:

```bash
python train_sequential.py       # frozen-backbone phase + partial-unfreeze phase
python train_parallel.py
```

### Hyperparameters

| Parameter | Fast path | Legacy |
|---|---|---|
| Image size | 224 × 224 | 224 × 224 |
| Batch size | 256 | 32 |
| Head learning rate | 1e-3 | 1e-3 (P1) / 1e-4 (P2) |
| Backbone learning rate | n/a (frozen) | 1e-5 (P2) |
| Weight decay | 1e-4 | 1e-4 |
| Epochs | 80 (head only) | 30 + 20 (two phases) |
| Frames per video | 15 | 15 |
| Cached aug copies (K) | 5 | n/a |
| Mixup α | 0.2 | 0 |

---

## Inference / Demo

```bash
python demo.py
```

Runs both models on synthetic test data and saves visualisations showing:
- Detection result and probability
- GAN attention maps (suspicious regions)
- Feature importance plots

---

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Sequential GAN+ViT | ~95% | ~94% | ~95% | ~94% |
| Parallel GAN+ViT | ~96% | ~95% | ~96% | ~95% |

> Results on SDFVD validation split. Actual numbers depend on GPU, dataset size, and training duration.

---

## Interpretability

FakeShield provides several visualisation tools:

- **GAN Attention Map** — spatial heatmap of regions the EfficientNet discriminator flagged
- **Cross-Attention Weights** — how the two streams influence each other (Parallel model)
- **Feature Importance** — channel-wise feature norms projected back to image space
- **Confidence Score** — model's self-reported certainty (Parallel model)

---

## Limitations

- Requires a GPU for reasonable training times
- Performance drops on very high-quality GAN-generated deepfakes not in training distribution
- Video-level predictions are frame-averaged (temporal modelling is future work)

---

## Future Work

- Temporal modelling (LSTM / Transformer over frame sequences)
- Frequency-domain branch (DCT/FFT features for compression artifact detection)
- Adversarial fine-tuning for robustness against adaptive attacks
- Lightweight distilled version for real-time mobile inference

---

## License

This project is released under the [MIT License](LICENSE) — free to use, modify, and distribute.
