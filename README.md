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
│   ├── sequential_model.py       # Sequential GAN→ViT fusion
│   └── parallel_model.py         # Parallel GAN‖ViT with cross-attention
├── utils/
│   ├── __init__.py
│   └── data_utils.py             # Dataset loading, augmentation, SDFVD support
├── train.py                      # Training script
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

```bash
python train.py
```

This will:
- Load SDFVD, extract frames, build dataloaders
- Train both Sequential and Parallel models for up to 50 epochs
- Save best checkpoints to `checkpoints/sequential/` and `checkpoints/parallel/`
- Plot training curves to `sequential_training_history.png` / `parallel_training_history.png`

### Hyperparameters (defaults)

| Parameter | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 8 |
| Head learning rate | 1e-4 |
| Backbone learning rate | 1e-5 |
| Weight decay | 1e-4 |
| Epochs | 50 |
| Frames per video | 15 |

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

For research and educational purposes only. Use responsibly.
