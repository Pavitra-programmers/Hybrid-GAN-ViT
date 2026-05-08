"""
extract_features.py — One-time encoder pass to build the feature cache.

Runs frozen EfficientNet-B4 + ViT-B/16 over every frame in SDFVD/{train,val}
and saves the resulting feature vectors to feature_cache/.

Re-run with --force if you change the dataset, image size, or num_aug.
"""

import argparse
import os
import sys

import torch

from utils.feature_cache import extract_and_cache_features


def main():
    parser = argparse.ArgumentParser(description="Cache frozen-encoder features.")
    parser.add_argument('--data-dir', default='./SDFVD', help='Dataset root (with train/val).')
    parser.add_argument('--cache-dir', default='./feature_cache',
                        help='Where to save the .pt cache files. Tip: use a different '
                             'directory per backbone (e.g. ./feature_cache_dinov2).')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='For DINOv2 ViT-L/14 on 4 GB GPUs, try 16 or 32.')
    parser.add_argument('--num-aug', type=int, default=5,
                        help='Number of augmented copies per train frame (val uses 1).')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--force', action='store_true', help='Rebuild even if signature matches.')
    parser.add_argument('--vit-backbone', default='vit_b_16',
                        choices=['vit_b_16',
                                 'dinov2_vits14', 'dinov2_vitb14',
                                 'dinov2_vitl14', 'dinov2_vitg14'],
                        help='Which ViT backbone to use. DINOv2 variants are downloaded '
                             'via torch.hub on first run (vitl14 is ~1.1 GB).')
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Error: dataset not found at {args.data_dir}")
        sys.exit(1)

    train_split = os.path.join(args.data_dir, 'train')
    val_split = os.path.join(args.data_dir, 'val')
    if not os.path.isdir(train_split) or not os.path.isdir(val_split):
        print(f"Error: expected {train_split} and {val_split} to exist.")
        print("Run train_sequential.py once (it auto-creates the split) or "
              "create the split manually.")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*65}")
    print("  FEATURE CACHE EXTRACTION")
    print(f"{'='*65}")
    print(f"  Device     : {device}")
    if torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Cache dir  : {args.cache_dir}")
    print(f"  ViT backbone: {args.vit_backbone}")
    print(f"  Image size : {args.img_size}")
    print(f"  Train aug  : {args.num_aug} copies per frame")
    print(f"  Force      : {args.force}")
    print()

    extract_and_cache_features(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_aug=args.num_aug,
        num_workers=args.num_workers,
        device=device,
        force=args.force,
        vit_backbone=args.vit_backbone,
    )

    print(f"\n{'='*65}")
    print("  EXTRACTION COMPLETE — you can now run:")
    print("    python train_sequential_fast.py")
    print("    python train_parallel_fast.py")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
