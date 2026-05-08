# FakeShield — Hybrid Deepfake Detection via GAN + Vision Transformer

> A two-stream deepfake detection system that fuses an **EfficientNet-B4**
> CNN backbone (artifact-sensitive low-level features) with a pretrained
> **Vision Transformer** (global context). Two fusion designs are compared
> — Sequential and Parallel cross-attention — under stratified 5-fold cross-
> validation on SDFVD. Best result: **94.30% ± 0.70% video accuracy** with
> the Parallel head on DINOv2-B/14 features.

---

## Headline numbers (5-fold stratified CV on SDFVD)

| Method | Backbone (ViT side) | Frame Acc | Video Acc |
|---|---|---|---|
| Linear probe (concat baseline) | ViT-B/16 | 91.19% ± 1.29% | — |
| Sequential GAN+ViT | ViT-B/16 | 90.88% ± 1.35% | 92.52% ± 1.49% |
| Parallel GAN+ViT | ViT-B/16 | 90.94% ± 0.64% | 91.93% ± 1.14% |
| Sequential GAN+ViT | DINOv2-B/14 | 92.96% ± 0.51% | 93.11% ± 0.62% |
| **Parallel GAN+ViT** | **DINOv2-B/14** | **93.27% ± 0.38%** | **94.30% ± 0.70%** |
| Ensemble (Sequential + Parallel) | DINOv2-B/14 | 93.40% ± 0.44% | 93.70% ± 0.97% |

All numbers are **mean ± standard deviation across 5 folds** (stratified, balanced
real/fake per fold, seed 42). The Parallel architecture has the tightest std,
indicating consistent generalisation.

---

## Why this approach

A single CNN learns sharp local artifact features but struggles with global
geometric inconsistencies; a single ViT captures global structure but
under-attends to fine compression seams and blending edges. FakeShield runs
**both** in parallel and lets them inform each other through cross-attention
before a final fused classification.

Two architectures are implemented and compared:

| Architecture | Strategy |
|---|---|
| **Sequential** | Concatenate GAN + ViT features, then 3-layer MLP fusion + auxiliary heads on each branch |
| **Parallel** | Bidirectional cross-attention: GAN features query ViT features and vice versa, then fusion |

---

## Architecture diagrams

### Sequential head (`SequentialHead`)
```
gan_raw  (1792-d, frozen EffNet-B4 avgpool)        vit_raw (768-d, frozen ViT)
   │                                                  │
   ▼                                                  ▼
GAN projector → 512-d  +  aux logit               ViT projector → 384-d  +  aux logit
                       \\                           /
                        Concat → 896-d
                                │
                  LayerNorm + GELU MLP (512→256)
                                │
                          Linear → 1 logit
```

### Parallel head (`ParallelHead`)
```
gan_raw                                vit_raw
   │                                      │
   ▼                                      ▼
GAN projector (512-d)                 ViT projector (384-d)
   │                                      │
   └──────► Bidirectional Cross-Attention ◄─┘
            (GAN ⇄ ViT, attention_dim=256)
   │                                      │
   ▼                                      ▼
   gan_enhanced (512-d)         vit_enhanced (384-d)
                       \\        /
                        Concat → 896-d
                                │
                  LayerNorm + GELU MLP (512→256)
                                │
                          Linear → 1 logit
```

The two backbones (EfficientNet-B4 + a chosen ViT — defaults to torchvision
ViT-B/16; can be swapped to DINOv2 ViT-S/14, B/14, L/14, or g/14) are **always
frozen**. Only the head trains. Their per-frame outputs are encoded once, cached
to disk, and reused every epoch — making each CV fold finish in seconds.

---

## Project layout

```
FakeShield/
├── extract_features.py          # One-time: encode every frame to disk
├── train_sequential_fast.py     # 5-fold CV training of SequentialHead
├── train_parallel_fast.py       # 5-fold CV training of ParallelHead
├── train_ensemble_fast.py       # 5-fold CV co-training of both heads + ensemble eval
├── demo.py                      # Inference on a single image (TTA + ensemble + abstention)
├── models/
│   ├── cached_heads.py          # SequentialHead / ParallelHead (head-only models)
│   ├── sequential_model.py      # Legacy full-backbone Sequential model
│   ├── parallel_model.py        # Legacy full-backbone Parallel model
│   ├── gan_discriminator.py
│   └── vision_transformer.py
├── utils/
│   ├── feature_cache.py         # Frozen-encoder pass + on-disk cache
│   ├── cached_dataset.py        # Datasets / DataLoaders for cached features
│   └── data_utils.py            # SDFVD frame extraction (legacy path)
├── checkpoints/
│   ├── sequential_fast/SequentialHead_best.pth
│   ├── parallel_fast/ParallelHead_best.pth
│   └── ensemble_fast/Ensemble_best.pth
├── feature_cache/               # Cached encoder features (ViT-B/16)
├── feature_cache_dinov2/        # Cached encoder features (DINOv2 backbone)
├── SDFVD/
│   ├── videos_real/             # raw real videos
│   ├── videos_fake/             # raw deepfake videos
│   ├── train/{real,fake}/       # per-class video subsets (auto-created)
│   └── val/{real,fake}/         # per-class video subsets (auto-created)
├── requirements.txt
├── README.md
└── report.txt
```

---

## Installation

```bash
git clone <your-repo-url>
cd "Research Paper"

python -m venv env
# Windows:
env\Scripts\activate
# Linux / macOS:
source env/bin/activate

pip install -r requirements.txt
```

Tested on Windows 11 with Python 3.10, PyTorch 2.x, CUDA 11.8, and an
NVIDIA GeForce GTX 1650 (4 GB VRAM).

---

## Dataset setup (SDFVD)

The first run extracts frames from raw `videos_real/` and `videos_fake/`
folders into `SDFVD/{train,val}/extracted_frames/{real,fake}/`. After that,
all training operates on the cached feature files and never re-decodes video.

If you start from a fresh SDFVD download:

```
SDFVD/
├── videos_real/
│   ├── v1.mp4
│   └── ...
└── videos_fake/
    ├── vs1.mp4
    └── ...
```

The legacy frame-extraction path (run once via `python -c "from utils.data_utils
import _create_train_val_split, DeepfakeDataset; _create_train_val_split('./SDFVD');
DeepfakeDataset('./SDFVD/train'); DeepfakeDataset('./SDFVD/val')"`) writes the
extracted JPGs into the per-class folders before feature caching.

---

## Pipeline overview

The full training cycle has three independent stages:

1. **Cache features once** — `extract_features.py` runs the frozen backbones
   over every frame and saves the resulting (1792-d EffNet, *D*-d ViT) vectors.
2. **Train heads with 5-fold CV** — `train_*_fast.py` reads the cache, splits
   the combined train+val pool into 5 stratified folds, trains a fresh head per
   fold, and reports mean ± std accuracy.
3. **Run inference** — `demo.py` loads the best checkpoint, encodes a single
   image via the same frozen backbones, applies 10-crop TTA + (optionally)
   ensemble averaging, and prints `P(fake)` with a 3-state decision band.

---

## Quick start (3 commands)

```bash
# 1.  Cache features once  (~3 hr on a GTX 1650 for K=5 augmentations)
python extract_features.py

# 2.  Train all three models  (~10 minutes total — features are cached)
python train_sequential_fast.py
python train_parallel_fast.py
python train_ensemble_fast.py

# 3.  Test on an image
python demo.py "path/to/image.jpg" --model ensemble
```

Each command is detailed below.

---

## Step 1 — Feature extraction

```bash
python extract_features.py [options]
```

| Flag | Default | Meaning |
|---|---|---|
| `--data-dir` | `./SDFVD` | Dataset root containing `train/` and `val/` |
| `--cache-dir` | `./feature_cache` | Where the `train_features.pt` / `val_features.pt` files are saved |
| `--vit-backbone` | `vit_b_16` | Choose `vit_b_16`, `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `dinov2_vitg14` |
| `--img-size` | `224` | Encoder input resolution |
| `--batch-size` | `64` | Encoding batch size (drop to 16–32 for DINOv2 ViT-L/14 on 4 GB GPUs) |
| `--num-aug` | `5` | Number of augmented copies cached per train frame (val uses 1) |
| `--force` | off | Rebuild even if signature matches |

**Use a separate cache directory per backbone** so they don't overwrite each other:

```bash
# ViT-B/16 (default)
python extract_features.py --cache-dir ./feature_cache

# DINOv2 ViT-B/14 (drop-in replacement, +2 pts CV accuracy)
python extract_features.py --vit-backbone dinov2_vitb14 --cache-dir ./feature_cache_dinov2

# DINOv2 ViT-L/14 (larger, slower; potentially +1 more pt)
python extract_features.py --vit-backbone dinov2_vitl14 --cache-dir ./feature_cache_dinov2 --batch-size 16
```

The DINOv2 weights are downloaded once via `torch.hub` from facebookresearch/dinov2
and cached to `~/.cache/torch/hub/`.

---

## Step 2 — Training

All three trainers share the same CLI surface:

| Flag | Default | Meaning |
|---|---|---|
| `--cache-dir` | `./feature_cache` | Which feature cache to read |
| `--folds` | `5` | Number of stratified CV folds |
| `--epochs` | `40` | Maximum epochs per fold |
| `--batch-size` | `128` | Training batch size |
| `--lr` | `3e-4` | AdamW learning rate |
| `--weight-decay` | `5e-3` | AdamW weight decay |
| `--patience` | `15` | Early-stop patience (epochs without improvement) |
| `--dropout` | `0.3` | Head dropout |
| `--seed` | `42` | RNG seed (controls fold split) |

```bash
# Sequential head only
python train_sequential_fast.py --cache-dir ./feature_cache_dinov2

# Parallel head only
python train_parallel_fast.py --cache-dir ./feature_cache_dinov2

# Ensemble: trains both heads per fold, evaluates them as an ensemble
python train_ensemble_fast.py --cache-dir ./feature_cache_dinov2
```

Outputs:
- Per-fold val accuracy / F1 / video-level accuracy
- Cross-validation mean ± std (frame and video)
- Best fold's checkpoint saved to `checkpoints/{sequential,parallel,ensemble}_fast/`

The ensemble trainer also writes its trained Sequential and Parallel models
into the standalone checkpoint paths, so `demo.py` can load any of them.

### Loss & training details

- **Loss**: `BCEWithLogitsLoss` on the main output + auxiliary BCE on the
  GAN-branch and ViT-branch logits with weight α = 0.1.
- **Label smoothing** ε = 0.1 (smoothed targets passed to BCE-with-logits).
- **Sampler**: `WeightedRandomSampler` for class balance (the SDFVD splits
  are balanced, but we keep this in case a future dataset isn't).
- **Scheduler**: `CosineAnnealingLR` over the full max-epoch budget.
- **Eval-time TTA**: every val frame is scored at all K=5 cached augmentations
  and the sigmoid probabilities are averaged before thresholding.
- **Best-checkpoint selection**: highest *frame* val accuracy per fold (lower
  variance than 22-video-level F1 on SDFVD).

---

## Step 3 — Inference

```bash
python demo.py "C:\path\to\image.jpg" [--model {sequential|parallel|ensemble|auto}] [--no-tta]
```

| Flag | Default | Meaning |
|---|---|---|
| (positional) `image` | — | Path to the image to classify |
| `--model` | `auto` | `sequential`, `parallel`, `ensemble`, or `auto` (uses ensemble if both checkpoints exist) |
| `--no-tta` | off | Skip 10-crop TTA (faster, slightly less robust) |
| `--threshold-low` | `0.35` | P(fake) below this → REAL |
| `--threshold-high` | `0.65` | P(fake) above this → FAKE |
| `--seq-ckpt` | `checkpoints/sequential_fast/SequentialHead_best.pth` | Sequential checkpoint |
| `--par-ckpt` | `checkpoints/parallel_fast/ParallelHead_best.pth` | Parallel checkpoint |

What the demo does in order:

1. Reads the checkpoint metadata to find which backbone (e.g. `dinov2_vitb14`)
   and which feature dimensions were used during training.
2. Loads the matching frozen backbones (EfficientNet-B4 + the chosen ViT).
3. Preprocesses the input image into 10 TTA views (4 corners + center, plus
   their horizontal flips), or 1 view with `--no-tta`.
4. Encodes all views through the frozen backbones.
5. Runs the head(s) on every view's features.
6. Averages sigmoid probabilities across views and (for ensemble) across heads.
7. Prints `P(fake)` mean and range, plus a 3-state decision: `REAL` /
   `UNCERTAIN` / `FAKE`.

Example output:
```
Device: cuda
Mode  : ensemble  (TTA: 10-crop)
  sequential  CV (5-fold) mean ± std: 93.40% ± 0.44%
    parallel  CV (5-fold) mean ± std: 93.40% ± 0.44%
  Backbone: EfficientNet-B4 + dinov2_vitb14

Image: C:\path\to\image.jpg
  Views encoded   : 10
  P(fake) mean    :  20.92%
  P(fake) range   :   3.89% —  63.28%
  Decision band   : REAL < 35%  |  UNCERTAIN  |  FAKE > 65%
  →  REAL
```

The `UNCERTAIN` zone is intentional. On out-of-distribution images (random
photos, screenshots, AI-generated images that don't match SDFVD's
manipulation style), the model abstains rather than confidently mis-classifying.

---

## Reproducing the headline result

```bash
# 1. Extract features with DINOv2 ViT-B/14 (one-time, ~3 hr on GTX 1650)
python extract_features.py --vit-backbone dinov2_vitb14 --cache-dir ./feature_cache_dinov2

# 2. Train all three (~10 minutes total)
python train_sequential_fast.py --cache-dir ./feature_cache_dinov2
python train_parallel_fast.py   --cache-dir ./feature_cache_dinov2
python train_ensemble_fast.py   --cache-dir ./feature_cache_dinov2
```

Expected output (seed=42):
- Sequential: ~92.96% ± 0.51% frame, ~93.11% ± 0.62% video
- Parallel:   ~93.27% ± 0.38% frame, ~94.30% ± 0.70% video  ← **best video**
- Ensemble:   ~93.40% ± 0.44% frame, ~93.70% ± 0.97% video  ← **best frame**

---

## Hyperparameters reference

| Parameter | Value | Notes |
|---|---|---|
| Image size | 224 × 224 | Standard for both backbones |
| Optimiser | AdamW | β₁=0.9, β₂=0.999 |
| Learning rate (head) | 3e-4 | Default for all three trainers |
| Weight decay | 5e-3 | Heavy regularisation; small dataset |
| Mixup α | 0 (disabled) | K=5 cached augmentations already provide variance |
| Aux-loss weight α | 0.1 | Auxiliary GAN/ViT branch losses |
| Label smoothing ε | 0.1 | Symmetric (ε/2 on each class) |
| Dropout | 0.3 | In all head MLP layers |
| Batch size | 128 | Heads are tiny; bigger batch is fine |
| Epochs (max) | 40 | Per fold |
| Early-stop patience | 15 | Frame-level val acc, no improvement |
| LR schedule | Cosine | T_max = epochs, η_min = 1e-6 |
| K (cached aug copies) | 5 | Train side; val uses K=1 |
| Eval-time TTA | All K augs averaged | Free since features are cached |

---

## Limitations

- **SDFVD is small** (~106 unique videos, 1,590 frames after extraction). The
  reported standard deviations are tight, but a single random train/val split
  has high variance — that's why we report 5-fold CV.
- **Single-domain training**. Models trained on SDFVD generalise well to
  SDFVD-like videos and poorly to images outside that distribution (UI
  screenshots, text-to-image AI art, casual phone photos in unseen settings).
  The `UNCERTAIN` decision band exposes this honestly rather than hiding it.
- **Frame-level model**. Temporal coherence between consecutive frames is not
  modelled. Video-level accuracy is computed by averaging per-frame
  predictions.
- **Backbone is frozen**. We do not fine-tune EfficientNet-B4 or the ViT — both
  for speed and to avoid overfitting on a corpus this small.

---

## Future work

- **Cross-dataset evaluation** — train on FaceForensics++ or Celeb-DF, test
  zero-shot on SDFVD. The feature cache + head trainers are dataset-agnostic;
  dropping in a second cache reuses the entire pipeline.
- **Temporal model** — LSTM or temporal Transformer over per-frame features.
- **Frequency-domain branch** — DCT/FFT features as a third stream alongside
  GAN and ViT, for compression-artifact detection.
- **DINOv2 ViT-L/14 or ViT-g/14** — strictly larger backbones; modest expected
  lift but at significant extraction-time cost.
- **End-to-end fine-tuning** — unfreeze the last few backbone blocks for the
  final epochs of training, weighed against the overfitting risk.

---

## License

MIT. See [LICENSE](LICENSE).
