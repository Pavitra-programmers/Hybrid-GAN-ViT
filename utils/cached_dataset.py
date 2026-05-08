"""
cached_dataset.py — Datasets / DataLoaders for cached encoder features.

Two entry points:
  - create_cached_dataloaders(train_cache, val_cache, ...)
        Original held-out split (used by the simple trainer).
  - merge_caches_for_kfold(train_cache, val_cache) + make_kfold_loaders(...)
        For 5-fold CV across the *combined* train+val data — required for
        small datasets like SDFVD where any single train/val split is high
        variance.

For the train side (random_aug=True), if the underlying cache has K>1
augmented copies per frame, __getitem__ picks one at random each call.
For the val side, k=0 (deterministic).
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .feature_cache import load_cached_features


class CachedFeaturesDataset(Dataset):
    """Dataset over a single cache file."""

    def __init__(self, cache_path: str, random_aug: bool = True):
        cache = load_cached_features(cache_path)
        self.gan_feats = cache['gan_feats']        # (N, K, 1792) float16
        self.vit_feats = cache['vit_feats']        # (N, K,  768) float16
        self.labels = cache['labels'].long()       # (N,)
        self.paths = cache['paths']
        self.video_ids = cache['video_ids']
        self.num_aug = self.gan_feats.shape[1]
        self.random_aug = random_aug and self.num_aug > 1

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        k = torch.randint(0, self.num_aug, (1,)).item() if self.random_aug else 0
        gan = self.gan_feats[idx, k].float()
        vit = self.vit_feats[idx, k].float()
        return gan, vit, self.labels[idx].float(), idx


class IndexedFeaturesDataset(Dataset):
    """Dataset that wraps pre-merged tensors and slices by an index list.

    Used for k-fold CV where we keep one in-memory representation of all data
    and expose only fold-specific indices to each loader.
    """

    def __init__(self, gan_feats, vit_feats, labels, video_ids, paths,
                 indices, random_aug: bool):
        self.gan_feats = gan_feats     # (N_total, K, 1792) float16
        self.vit_feats = vit_feats     # (N_total, K,  768) float16
        self.labels = labels           # (N_total,) long
        self.video_ids = video_ids     # list[str], length N_total
        self.paths = paths             # list[str], length N_total
        self.indices = list(indices)
        self.num_aug = gan_feats.shape[1]
        self.random_aug = random_aug and self.num_aug > 1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        k = torch.randint(0, self.num_aug, (1,)).item() if self.random_aug else 0
        gan = self.gan_feats[idx, k].float()
        vit = self.vit_feats[idx, k].float()
        return gan, vit, self.labels[idx].float(), idx


def merge_caches_for_kfold(train_cache_path: str, val_cache_path: str) -> dict:
    """Load both caches and concatenate them for k-fold CV.

    Aug dimension K is normalized to max(K_train, K_val) by tiling the smaller
    along the K axis. Concretely: if train has K=5 and val has K=1, the val
    block is replicated 5× so the final tensor is rectangular and any frame
    can be sampled uniformly across augmentations.

    Returns:
        dict with keys: gan_feats, vit_feats, labels, paths, video_ids,
                        n_train, n_val, num_aug
    """
    train = load_cached_features(train_cache_path)
    val = load_cached_features(val_cache_path)

    K = max(train['gan_feats'].shape[1], val['gan_feats'].shape[1])

    def _pad_aug(t: torch.Tensor, target_K: int) -> torch.Tensor:
        cur = t.shape[1]
        if cur == target_K:
            return t
        repeats = target_K // cur
        remainder = target_K - repeats * cur
        parts = [t] * repeats + ([t[:, :remainder]] if remainder else [])
        return torch.cat(parts, dim=1)

    gan = torch.cat([_pad_aug(train['gan_feats'], K),
                     _pad_aug(val['gan_feats'], K)], dim=0)
    vit = torch.cat([_pad_aug(train['vit_feats'], K),
                     _pad_aug(val['vit_feats'], K)], dim=0)
    labels = torch.cat([train['labels'], val['labels']], dim=0).long()
    paths = list(train['paths']) + list(val['paths'])
    video_ids = list(train['video_ids']) + list(val['video_ids'])

    train_meta = train.get('meta', {})
    val_meta = val.get('meta', {})
    return {
        'gan_feats': gan,
        'vit_feats': vit,
        'labels': labels,
        'paths': paths,
        'video_ids': video_ids,
        'n_train': len(train['labels']),
        'n_val': len(val['labels']),
        'num_aug': K,
        'vit_backbone': train_meta.get('vit_backbone',
                                       val_meta.get('vit_backbone', 'vit_b_16')),
        'gan_dim': train_meta.get('gan_dim', gan.shape[-1]),
        'vit_dim': train_meta.get('vit_dim', vit.shape[-1]),
    }


def make_kfold_loaders(merged: dict, train_idx, val_idx,
                       batch_size: int = 128, num_workers: int = 0,
                       use_weighted_sampler: bool = True):
    """Build train/val loaders for one k-fold split."""
    train_ds = IndexedFeaturesDataset(
        merged['gan_feats'], merged['vit_feats'], merged['labels'],
        merged['video_ids'], merged['paths'],
        train_idx, random_aug=True,
    )
    val_ds = IndexedFeaturesDataset(
        merged['gan_feats'], merged['vit_feats'], merged['labels'],
        merged['video_ids'], merged['paths'],
        val_idx, random_aug=False,
    )

    if use_weighted_sampler and len(train_ds) > 0:
        fold_labels = merged['labels'][list(train_idx)]
        weights = _balanced_weights(fold_labels)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return train_loader, val_loader, train_ds, val_ds


def _balanced_weights(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=2).float()
    total = counts.sum()
    class_w = total / (2.0 * counts.clamp(min=1.0))
    return class_w[labels]


def create_cached_dataloaders(train_cache: str, val_cache: str,
                              batch_size: int = 256,
                              num_workers: int = 0,
                              use_weighted_sampler: bool = True):
    """Original held-out split. Kept for backwards compatibility."""
    train_ds = CachedFeaturesDataset(train_cache, random_aug=True)
    val_ds = CachedFeaturesDataset(val_cache, random_aug=False)

    if use_weighted_sampler and len(train_ds) > 0:
        weights = _balanced_weights(train_ds.labels)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    n_real_t = int((train_ds.labels == 0).sum())
    n_fake_t = int((train_ds.labels == 1).sum())
    print(f"  Train cache: {len(train_ds):,} frames "
          f"(real={n_real_t}, fake={n_fake_t}, K={train_ds.num_aug})")
    print(f"  Val cache:   {len(val_ds):,} frames "
          f"(real={int((val_ds.labels == 0).sum())}, "
          f"fake={int((val_ds.labels == 1).sum())})")

    return train_loader, val_loader, train_ds, val_ds
