"""
demo.py — Run trained Sequential / Parallel cached-head models on an image.

Three robustness techniques on top of the basic forward pass:

  1. Test-Time Augmentation (TTA):
       The image is preprocessed into 10 views — center crop + 4 corner crops,
       plus their horizontal flips — and predictions are averaged at the logit
       level. This smooths overconfident out-of-distribution calls, which is
       important because the training set (SDFVD, ~106 videos) doesn't cover
       the broad distribution of arbitrary photos / screenshots.

  2. Ensemble:
       When both Sequential and Parallel checkpoints are available, both heads
       run and their logits are averaged. Standard cheap robustness gain.

  3. Confidence band (abstention zone):
       Three-state decision instead of a hard 0.5 threshold. P(fake) below 0.35
       is REAL, above 0.65 is FAKE, between is UNCERTAIN. Lets the model say
       "I don't know" rather than guessing on OOD inputs.

Usage:
    python demo.py path/to/image.jpg
    python demo.py path/to/image.jpg --model parallel
    python demo.py path/to/image.jpg --model sequential
    python demo.py path/to/image.jpg --no-tta
    python demo.py path/to/image.jpg --threshold 0.5
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torchvision.models as tv_models
from PIL import Image
from torchvision import transforms

from models.cached_heads import SequentialHead, ParallelHead


IMG_SIZE = 224
RESIZE_FOR_TTA = 256                 # ImageNet TenCrop convention
SEQ_CKPT = 'checkpoints/sequential_fast/SequentialHead_best.pth'
PAR_CKPT = 'checkpoints/parallel_fast/ParallelHead_best.pth'
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def load_backbones(device: torch.device):
    try:
        from torchvision.models import EfficientNet_B4_Weights, ViT_B_16_Weights
        eff = tv_models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        vit = tv_models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    except (AttributeError, ImportError):
        eff = tv_models.efficientnet_b4(pretrained=True)
        vit = tv_models.vit_b_16(pretrained=True)

    eff_features = eff.features.to(device).eval()
    eff_pool = eff.avgpool.to(device).eval()
    vit.heads = nn.Identity()
    vit = vit.to(device).eval()
    for m in (eff_features, eff_pool, vit):
        for p in m.parameters():
            p.requires_grad = False
    return eff_features, eff_pool, vit


def preprocess(image_path: str, use_tta: bool) -> torch.Tensor:
    """Returns a (V, 3, IMG_SIZE, IMG_SIZE) tensor.  V = 10 with TTA, 1 without."""
    img = Image.open(image_path).convert('RGB')
    if use_tta:
        transform = transforms.Compose([
            transforms.Resize(RESIZE_FOR_TTA),
            transforms.TenCrop(IMG_SIZE),
            transforms.Lambda(lambda crops: torch.stack([
                NORMALIZE(transforms.functional.to_tensor(c)) for c in crops
            ])),
        ])
        return transform(img)                                          # (10, 3, 224, 224)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        NORMALIZE,
    ])
    return transform(img).unsqueeze(0)                                 # (1, 3, 224, 224)


@torch.no_grad()
def encode(views: torch.Tensor, eff_features, eff_pool, vit):
    """Encode V views into (V, 1792) GAN features and (V, 768) ViT features."""
    spatial = eff_features(views)
    gan = eff_pool(spatial).flatten(1)
    cls = vit(views)
    return gan, cls


def load_head(model_type: str, ckpt_path: str, device: torch.device):
    head = (SequentialHead(dropout=0.0) if model_type == 'sequential'
            else ParallelHead(dropout=0.0)).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    head.load_state_dict(ckpt['model_state_dict'])
    head.eval()
    return head, ckpt


def head_logits(head: nn.Module, gan: torch.Tensor, vit: torch.Tensor) -> torch.Tensor:
    out = head(gan, vit)
    return out['output'].squeeze(-1)        # (V,)


def decide(prob: float, low: float, high: float) -> str:
    if prob < low:
        return 'REAL'
    if prob > high:
        return 'FAKE'
    return 'UNCERTAIN'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Path to image to classify')
    parser.add_argument('--model', choices=['sequential', 'parallel', 'ensemble'],
                        default='auto', nargs='?',
                        help="Which head to use. 'auto' (default) ensembles both if "
                             "both checkpoints exist, else falls back to whichever "
                             "is available.")
    parser.add_argument('--no-tta', action='store_true',
                        help='Disable 10-crop test-time augmentation.')
    parser.add_argument('--threshold-low', type=float, default=0.35,
                        help='P(fake) below this → REAL')
    parser.add_argument('--threshold-high', type=float, default=0.65,
                        help='P(fake) above this → FAKE')
    parser.add_argument('--seq-ckpt', default=SEQ_CKPT)
    parser.add_argument('--par-ckpt', default=PAR_CKPT)
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        sys.exit(1)

    seq_avail = os.path.exists(args.seq_ckpt)
    par_avail = os.path.exists(args.par_ckpt)
    if args.model == 'auto':
        if seq_avail and par_avail:
            args.model = 'ensemble'
        elif seq_avail:
            args.model = 'sequential'
        elif par_avail:
            args.model = 'parallel'
        else:
            print("No checkpoints found. Train first:")
            print("  python train_sequential_fast.py")
            print("  python train_parallel_fast.py")
            sys.exit(1)
    if args.model == 'ensemble' and not (seq_avail and par_avail):
        print("--model ensemble requires both checkpoints.")
        sys.exit(1)
    if args.model == 'sequential' and not seq_avail:
        print(f"Sequential checkpoint missing: {args.seq_ckpt}")
        sys.exit(1)
    if args.model == 'parallel' and not par_avail:
        print(f"Parallel checkpoint missing: {args.par_ckpt}")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Mode  : {args.model}{'  (TTA off)' if args.no_tta else '  (TTA: 10-crop)'}")

    heads = []
    if args.model in ('sequential', 'ensemble'):
        h, c = load_head('sequential', args.seq_ckpt, device)
        heads.append(('sequential', h, c))
    if args.model in ('parallel', 'ensemble'):
        h, c = load_head('parallel', args.par_ckpt, device)
        heads.append(('parallel', h, c))

    for name, _h, c in heads:
        cv_mean, cv_std = c.get('cv_mean_acc'), c.get('cv_std_acc')
        if cv_mean is not None:
            print(f"  {name:>10s}  CV (5-fold) mean ± std: "
                  f"{cv_mean*100:.2f}% ± {cv_std*100:.2f}%")

    eff_features, eff_pool, vit = load_backbones(device)
    views = preprocess(args.image, use_tta=not args.no_tta).to(device)
    n_views = views.shape[0]
    gan_feat, vit_feat = encode(views, eff_features, eff_pool, vit)

    with torch.no_grad():
        all_logits = torch.stack([
            head_logits(h, gan_feat, vit_feat) for _name, h, _c in heads
        ], dim=0)                                                   # (n_heads, n_views)
    probs = torch.sigmoid(all_logits)                               # (n_heads, n_views)
    mean_prob = probs.mean().item()
    min_prob = probs.min().item()
    max_prob = probs.max().item()
    label = decide(mean_prob, args.threshold_low, args.threshold_high)

    print(f"\nImage: {args.image}")
    print(f"  Views encoded   : {n_views}")
    print(f"  P(fake) mean    : {mean_prob*100:6.2f}%")
    print(f"  P(fake) range   : {min_prob*100:6.2f}% — {max_prob*100:6.2f}%")
    print(f"  Decision band   : REAL < {args.threshold_low*100:.0f}%  |  "
          f"UNCERTAIN  |  FAKE > {args.threshold_high*100:.0f}%")
    print(f"  →  {label}")
    if label == 'UNCERTAIN':
        print(f"     (model is not confident on this image — "
              f"likely outside the SDFVD training distribution)")


if __name__ == '__main__':
    main()
