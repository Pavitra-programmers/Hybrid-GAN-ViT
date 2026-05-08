"""
feature_cache.py — Extract and cache frozen-backbone features for fast training.

The pretrained EfficientNet-B4 and ViT-B/16 backbones are frozen during Phase 1.
Their output for a given frame never changes, so re-running them every epoch is
wasted compute. This module encodes each frame ONCE into:

    - GAN raw features  : 1792-dim (EfficientNet-B4 avgpool output)
    - ViT raw features  :  768-dim (ViT-B/16 CLS token)

Features for K random augmentations per frame are stored, so head-only training
still benefits from augmentation diversity without re-running the encoders.

The cache is a single .pt file:
    {
        'gan_feats':  Tensor [N, K, 1792]  float16
        'vit_feats':  Tensor [N, K,  768]  float16
        'labels':     Tensor [N]           uint8   (0 = real, 1 = fake)
        'paths':      List[str] length N
        'video_ids':  List[str] length N   (basename without _frame_XXXXX)
        'split':      str ('train' or 'val')
        'signature':  str (hash of dataset state for cache invalidation)
        'meta':       dict (img_size, num_aug, model versions)
    }
"""

import hashlib
import os
from typing import Optional

import torch
import torchvision.models as tv_models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


GAN_FEAT_DIM = 1792   # EfficientNet-B4 avgpool output
DEFAULT_NUM_AUG = 5   # augmented copies per frame in the cache

# Output dim of the supported ViT backbones (CLS token).
VIT_BACKBONE_DIMS = {
    'vit_b_16':       768,    # torchvision ViT-B/16 (ImageNet1K-pretrained)
    'dinov2_vits14':  384,    # DINOv2 ViT-S/14
    'dinov2_vitb14':  768,    # DINOv2 ViT-B/14
    'dinov2_vitl14':  1024,   # DINOv2 ViT-L/14  ← strongest, ~1.1 GB download
    'dinov2_vitg14':  1536,   # DINOv2 ViT-g/14  ← largest, ~5 GB download
}


def _load_frozen_encoders(device: torch.device, vit_backbone: str = 'vit_b_16'):
    """Load EfficientNet-B4 + the requested ViT backbone, frozen, in eval mode.

    Returns (eff_features, eff_pool, vit, vit_dim).  `vit` is a callable that
    takes a (B, 3, 224, 224) tensor and returns a (B, vit_dim) CLS embedding.
    """
    try:
        from torchvision.models import EfficientNet_B4_Weights
        eff = tv_models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    except (AttributeError, ImportError):
        eff = tv_models.efficientnet_b4(pretrained=True)
    eff_features = eff.features.to(device).eval()
    eff_pool = eff.avgpool.to(device).eval()
    for p in list(eff_features.parameters()) + list(eff_pool.parameters()):
        p.requires_grad = False

    if vit_backbone == 'vit_b_16':
        try:
            from torchvision.models import ViT_B_16_Weights
            vit = tv_models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        except (AttributeError, ImportError):
            vit = tv_models.vit_b_16(pretrained=True)
        vit.heads = torch.nn.Identity()
        vit = vit.to(device).eval()
    elif vit_backbone.startswith('dinov2_'):
        # torch.hub fetches from facebookresearch/dinov2; first call downloads
        # weights (cached in ~/.cache/torch/hub).
        vit = torch.hub.load('facebookresearch/dinov2', vit_backbone,
                             trust_repo=True, verbose=False)
        vit = vit.to(device).eval()
    else:
        raise ValueError(f"Unknown vit_backbone: {vit_backbone!r}. "
                         f"Supported: {list(VIT_BACKBONE_DIMS)}")

    for p in vit.parameters():
        p.requires_grad = False

    if vit_backbone not in VIT_BACKBONE_DIMS:
        raise ValueError(f"Unknown vit_backbone: {vit_backbone!r}")
    return eff_features, eff_pool, vit, VIT_BACKBONE_DIMS[vit_backbone]


def _get_train_transform(img_size: int):
    """Augmentation pipeline mirrors data_utils.get_transforms(is_training=True)."""
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _get_val_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class _ImageListDataset(Dataset):
    """Reads images from a list of (path, label) tuples and applies a single transform."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label, idx


def _collect_samples(split_dir: str):
    """Walk split_dir/{real,fake}/*.jpg|png and return [(path, label, video_id), ...].

    Also looks under split_dir/extracted_frames/{real,fake} — that's where the
    legacy DeepfakeDataset writes frames after extracting them from videos.
    """
    samples = []
    frames_dir = os.path.join(split_dir, 'extracted_frames')
    for label_name, label_idx in [('real', 0), ('fake', 1)]:
        # Prefer extracted_frames/{label} if it has images (matches legacy layout).
        candidates = [
            os.path.join(frames_dir, label_name),
            os.path.join(split_dir, label_name),
        ]
        for sub in candidates:
            if not os.path.isdir(sub):
                continue
            images = [
                f for f in sorted(os.listdir(sub))
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            if not images:
                continue
            for fname in images:
                path = os.path.join(sub, fname)
                video_id = fname.rsplit('_frame_', 1)[0]
                samples.append((path, label_idx, video_id))
            break  # found images in this candidate; don't double-count
    return samples


def _signature_from_samples(samples, num_aug: int, img_size: int,
                            vit_backbone: str = 'vit_b_16') -> str:
    h = hashlib.sha1()
    h.update(f'aug={num_aug};size={img_size};n={len(samples)};vit={vit_backbone}'.encode())
    for path, label, vid in samples[:512]:   # truncated; fast and stable
        st = os.stat(path) if os.path.exists(path) else None
        size = st.st_size if st else 0
        h.update(f'{os.path.basename(path)}|{label}|{size};'.encode())
    return h.hexdigest()


@torch.no_grad()
def _encode_dataloader(loader, eff_features, eff_pool, vit, device,
                       gan_buf, vit_buf, aug_idx, desc):
    """Run one full pass; write features into [:, aug_idx, :] of the buffers."""
    for images, _labels, indices in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)

        spatial = eff_features(images)               # (B, 1792, h, w)
        pooled = eff_pool(spatial).flatten(1)        # (B, 1792)
        gan = pooled.float().cpu()

        cls = vit(images).float().cpu()              # (B, 768)

        for i, idx in enumerate(indices.tolist()):
            gan_buf[idx, aug_idx] = gan[i].half()
            vit_buf[idx, aug_idx] = cls[i].half()


def extract_and_cache_features(
    data_dir: str,
    cache_dir: str,
    img_size: int = 224,
    batch_size: int = 64,
    num_aug: int = DEFAULT_NUM_AUG,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
    force: bool = False,
    vit_backbone: str = 'vit_b_16',
) -> dict:
    """
    Walk data_dir/{train,val}/{real,fake}/*.{jpg,png}, encode every frame with
    frozen EfficientNet-B4 + the chosen ViT backbone, and cache the features.

    Train split: caches num_aug random-augmentation copies per frame.
    Val split:   caches a single deterministic (resized + normalized) copy.

    vit_backbone: 'vit_b_16' (default) or 'dinov2_vits14' / 'dinov2_vitb14' /
                  'dinov2_vitl14' / 'dinov2_vitg14'.

    Returns:
        dict with 'train_cache' and 'val_cache' file paths.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cache_dir, exist_ok=True)

    eff_features, eff_pool, vit, vit_dim = _load_frozen_encoders(device, vit_backbone)

    out = {}
    for split, n_aug, transform_fn in [
        ('train', num_aug, _get_train_transform),
        ('val', 1, _get_val_transform),
    ]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            print(f"[feature_cache] Skipping '{split}' — directory not found at {split_dir}")
            continue

        samples = _collect_samples(split_dir)
        if not samples:
            print(f"[feature_cache] No images found under {split_dir}; skipping.")
            continue

        cache_path = os.path.join(cache_dir, f'{split}_features.pt')
        signature = _signature_from_samples(samples, n_aug, img_size, vit_backbone)

        if os.path.exists(cache_path) and not force:
            try:
                existing = torch.load(cache_path, map_location='cpu', weights_only=False)
                if existing.get('signature') == signature:
                    n_real = int((existing['labels'] == 0).sum())
                    n_fake = int((existing['labels'] == 1).sum())
                    print(f"[feature_cache] Reusing cache: {cache_path}  "
                          f"(N={len(existing['labels'])}, K={existing['gan_feats'].shape[1]}, "
                          f"real={n_real}, fake={n_fake})")
                    out[f'{split}_cache'] = cache_path
                    continue
            except Exception as e:
                print(f"[feature_cache] Existing cache unreadable ({e}); rebuilding.")

        n = len(samples)
        gan_buf = torch.zeros((n, n_aug, GAN_FEAT_DIM), dtype=torch.float16)
        vit_buf = torch.zeros((n, n_aug, vit_dim), dtype=torch.float16)
        labels = torch.tensor([s[1] for s in samples], dtype=torch.uint8)
        paths = [s[0] for s in samples]
        video_ids = [s[2] for s in samples]

        flat = [(p, l) for p, l, _ in samples]

        for k in range(n_aug):
            tfm = transform_fn(img_size)
            ds = _ImageListDataset(flat, tfm)
            loader = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=device.type == 'cuda',
            )
            desc = f'  Encoding {split} aug {k+1}/{n_aug}'
            _encode_dataloader(loader, eff_features, eff_pool, vit, device,
                               gan_buf, vit_buf, k, desc)

        cache = {
            'gan_feats': gan_buf,
            'vit_feats': vit_buf,
            'labels': labels,
            'paths': paths,
            'video_ids': video_ids,
            'split': split,
            'signature': signature,
            'meta': {
                'img_size': img_size,
                'num_aug': n_aug,
                'gan_dim': GAN_FEAT_DIM,
                'vit_dim': vit_dim,
                'vit_backbone': vit_backbone,
            },
        }
        torch.save(cache, cache_path)
        n_real = int((labels == 0).sum())
        n_fake = int((labels == 1).sum())
        size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print(f"[feature_cache] Saved: {cache_path}  "
              f"(N={n}, K={n_aug}, real={n_real}, fake={n_fake}, {size_mb:.1f} MB)")
        out[f'{split}_cache'] = cache_path

    return out


def load_cached_features(cache_path: str) -> dict:
    """Load a feature cache file produced by extract_and_cache_features."""
    return torch.load(cache_path, map_location='cpu', weights_only=False)
