"""
train_parallel_fast.py — Stratified 5-fold CV training of ParallelHead on
cached EfficientNet-B4 + ViT-B/16 features.

Mirror of train_sequential_fast.py but trains the bidirectional cross-attention
ParallelHead. See that file for the full rationale on why this dataset needs
k-fold CV rather than a single held-out split.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

from models.cached_heads import ParallelHead
from utils.cached_dataset import merge_caches_for_kfold, make_kfold_loaders


SAVE_DIR = 'checkpoints/parallel_fast'


def smoothed_bce_loss(preds: dict, targets: torch.Tensor,
                      smoothing: float = 0.1, alpha: float = 0.1) -> torch.Tensor:
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)
    smooth = targets * (1 - smoothing) + smoothing * 0.5

    main = F.binary_cross_entropy_with_logits(preds['output'].float(), smooth)
    if alpha <= 0:
        return main
    aux = F.binary_cross_entropy_with_logits(preds['gan_classification'].float(), smooth) \
        + F.binary_cross_entropy_with_logits(preds['vit_classification'].float(), smooth)
    return main + alpha * aux


@torch.no_grad()
def evaluate(model, val_ds, device, batch_size: int = 128):
    """TTA evaluation: average sigmoid probabilities over all K cached augmentations
    per val frame before thresholding. Loss is still computed on k=0 for monitoring."""
    model.eval()
    K = val_ds.num_aug
    indices = list(val_ds.indices)
    N = len(indices)
    labels_arr = np.array([val_ds.labels[i].item() for i in indices], dtype=np.float32)
    video_ids = [val_ds.video_ids[i] for i in indices]

    sum_probs = np.zeros(N, dtype=np.float64)
    losses = []

    for k in range(K):
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_global = indices[start:end]
            gan = val_ds.gan_feats[batch_global, k].float().to(device, non_blocking=True)
            vit = val_ds.vit_feats[batch_global, k].float().to(device, non_blocking=True)
            out = model(gan, vit)
            probs = torch.sigmoid(out['output']).squeeze(-1).float().cpu().numpy()
            sum_probs[start:end] += probs
            if k == 0:
                lbl = torch.from_numpy(labels_arr[start:end]).to(device)
                losses.append(smoothed_bce_loss(out, lbl).item())

    avg_probs = sum_probs / K
    preds = (avg_probs >= 0.5).astype(int)

    has_both = len(np.unique(labels_arr)) > 1
    acc = (preds == labels_arr).mean() if N else 0.0
    prec = precision_score(labels_arr, preds, zero_division=0) if has_both else 0.0
    rec = recall_score(labels_arr, preds, zero_division=0) if has_both else 0.0
    f1 = f1_score(labels_arr, preds, zero_division=0) if has_both else 0.0

    by_video = {}
    for prob, lbl, v in zip(avg_probs, labels_arr, video_ids):
        by_video.setdefault(v, {'probs': [], 'label': lbl})['probs'].append(prob)
    if by_video:
        v_probs = np.array([np.mean(d['probs']) for d in by_video.values()])
        v_labels = np.array([d['label'] for d in by_video.values()])
        v_preds = (v_probs >= 0.5).astype(int)
        v_acc = (v_preds == v_labels).mean()
        v_f1 = f1_score(v_labels, v_preds, zero_division=0) if len(np.unique(v_labels)) > 1 else 0.0
    else:
        v_acc = v_f1 = 0.0

    return float(np.mean(losses)) if losses else 0.0, acc, prec, rec, f1, v_acc, v_f1


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    losses, correct, total = [], 0, 0
    for gan, vit, lbl, _ in loader:
        gan = gan.to(device, non_blocking=True)
        vit = vit.to(device, non_blocking=True)
        lbl = lbl.to(device, non_blocking=True)

        out = model(gan, vit)
        loss = smoothed_bce_loss(out, lbl)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        with torch.no_grad():
            pred = (torch.sigmoid(out['output']).squeeze(1) >= 0.5).float()
            correct += (pred == lbl).sum().item()
            total += lbl.size(0)
    return float(np.mean(losses)) if losses else 0.0, correct / total if total else 0.0


def train_one_fold(merged, train_idx, val_idx, fold_num, total_folds,
                   args, device):
    train_loader, val_loader, train_ds, val_ds = make_kfold_loaders(
        merged, train_idx, val_idx,
        batch_size=args.batch_size, num_workers=0,
        use_weighted_sampler=True,
    )
    n_real_t = int((merged['labels'][list(train_idx)] == 0).sum())
    n_fake_t = int((merged['labels'][list(train_idx)] == 1).sum())
    n_real_v = int((merged['labels'][list(val_idx)] == 0).sum())
    n_fake_v = int((merged['labels'][list(val_idx)] == 1).sum())
    print(f"\n  ─── Fold {fold_num}/{total_folds}  "
          f"(train: {len(train_idx)}  real={n_real_t} fake={n_fake_t};  "
          f"val: {len(val_idx)}  real={n_real_v} fake={n_fake_v}) ───")

    model = ParallelHead(dropout=args.dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=1e-6)

    best_acc = 0.0
    best_state = None
    best_metrics = None
    patience = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        vl_loss, vl_acc, prec, rec, f1, v_acc, v_f1 = evaluate(
            model, val_ds, device, batch_size=args.batch_size,
        )
        scheduler.step()

        marker = ''
        if vl_acc > best_acc:
            best_acc = vl_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = dict(val_loss=vl_loss, val_acc=vl_acc, val_f1=f1,
                                val_precision=prec, val_recall=rec,
                                val_video_acc=v_acc, val_video_f1=v_f1,
                                epoch=epoch)
            patience = 0
            marker = '  ✓'
        else:
            patience += 1

        if epoch == 1 or epoch % 5 == 0 or marker or patience >= args.patience:
            print(f"    Ep {epoch:>3}/{args.epochs}  ({time.time()-t0:.1f}s)  "
                  f"TrLoss={tr_loss:.4f} TrAcc={tr_acc*100:5.1f}%  "
                  f"VlLoss={vl_loss:.4f} VlAcc={vl_acc*100:5.1f}%  "
                  f"F1={f1*100:5.1f}%  [Vid Acc={v_acc*100:5.1f}%]{marker}")

        if patience >= args.patience:
            print(f"    Early stop  (best fold val_acc={best_acc*100:.2f}%)")
            break

    return best_acc, best_state, best_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', default='./feature_cache')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=5e-3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_cache = os.path.join(args.cache_dir, 'train_features.pt')
    val_cache = os.path.join(args.cache_dir, 'val_features.pt')
    if not os.path.exists(train_cache) or not os.path.exists(val_cache):
        print(f"Feature cache not found in {args.cache_dir}.")
        print("Run:  python extract_features.py")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*70}")
    print(f"  PARALLEL HEAD — {args.folds}-Fold Stratified CV")
    print(f"{'='*70}")
    print(f"  Device     : {device}")
    print(f"  Cache dir  : {args.cache_dir}")
    print(f"  Epochs     : {args.epochs}    Batch: {args.batch_size}")
    print(f"  LR         : {args.lr:.0e}    WD: {args.weight_decay}    Dropout: {args.dropout}")
    print(f"  Folds      : {args.folds}")

    merged = merge_caches_for_kfold(train_cache, val_cache)
    n_total = len(merged['labels'])
    print(f"  Total      : {n_total} frames "
          f"(real={int((merged['labels']==0).sum())}, "
          f"fake={int((merged['labels']==1).sum())}, "
          f"K augs={merged['num_aug']})")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    labels_np = merged['labels'].numpy()

    fold_accs, fold_f1s, fold_video_accs = [], [], []
    best_global_acc = 0.0
    best_payload = None

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n_total), labels_np), start=1):
        fold_acc, fold_state, fold_metrics = train_one_fold(
            merged, tr_idx, va_idx, fold, args.folds, args, device,
        )
        fold_accs.append(fold_acc)
        fold_f1s.append(fold_metrics['val_f1'])
        fold_video_accs.append(fold_metrics['val_video_acc'])
        if fold_acc > best_global_acc:
            best_global_acc = fold_acc
            best_payload = {
                'fold': fold,
                'fold_metrics': fold_metrics,
                'model_state_dict': fold_state,
                'all_fold_accs': None,
                'all_fold_f1s': None,
                'cv_mean_acc': None,
                'cv_std_acc': None,
                'args': vars(args),
            }

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    mean_f1 = float(np.mean(fold_f1s))
    std_f1 = float(np.std(fold_f1s))
    mean_v_acc = float(np.mean(fold_video_accs))
    std_v_acc = float(np.std(fold_video_accs))

    if best_payload is not None:
        best_payload['all_fold_accs'] = [float(a) for a in fold_accs]
        best_payload['all_fold_f1s'] = [float(a) for a in fold_f1s]
        best_payload['cv_mean_acc'] = mean_acc
        best_payload['cv_std_acc'] = std_acc
        os.makedirs(SAVE_DIR, exist_ok=True)
        torch.save(best_payload, os.path.join(SAVE_DIR, 'ParallelHead_best.pth'))

    print(f"\n{'='*70}")
    print(f"  CROSS-VALIDATION RESULTS (ParallelHead)")
    print(f"{'='*70}")
    for i, (a, f1, va) in enumerate(zip(fold_accs, fold_f1s, fold_video_accs), start=1):
        print(f"    Fold {i}:  val_acc={a*100:6.2f}%   val_f1={f1*100:6.2f}%   video_acc={va*100:6.2f}%")
    print(f"  ───────────────────────────────────────────────────────────────────")
    print(f"    Mean ± Std (frame acc):  {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"    Mean ± Std (frame F1) :  {mean_f1*100:.2f}% ± {std_f1*100:.2f}%")
    print(f"    Mean ± Std (video acc):  {mean_v_acc*100:.2f}% ± {std_v_acc*100:.2f}%")
    print(f"    Best fold val_acc     :  {best_global_acc*100:.2f}%  (saved)")
    print(f"  Checkpoint : {SAVE_DIR}/ParallelHead_best.pth")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
