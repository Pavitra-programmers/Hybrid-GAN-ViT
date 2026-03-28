"""
train_sequential.py — Two-phase training for SequentialGANViT

Phase 1 (frozen backbone):
  Both EfficientNet-B4 and ViT-B/16 backbones are completely frozen.
  Only the projection heads, fusion MLP, and classifier are trained.
  This prevents catastrophic forgetting of ImageNet features on small datasets.
  ≈ 2–3 M trainable params vs 105 M total.

Phase 2 (partial unfreeze):
  Last 2 EfficientNet blocks + last 4 ViT transformer layers are unfrozen.
  Backbone fine-tuned at 10× lower LR than new layers.
  Runs only if Phase 1 val_F1 > PHASE2_MIN_F1 threshold.

Additional improvements over the original train.py:
  - WeightedRandomSampler  → balanced batches despite class imbalance
  - Gradient accumulation  → effective batch size = 32 on 4 GB GPU
  - Mixed precision (AMP)  → faster training, lower memory
  - Plain BCE + label smoothing  → more stable than focal loss on small data
  - Early stopping on F1 (not accuracy) → robust to class imbalance
  - num_workers=0  → Windows-safe multiprocessing
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from models.sequential_model import SequentialGANViT
from utils.data_utils import create_dataloaders

# ── Hyperparameters ──────────────────────────────────────────────────────────
IMG_SIZE          = 224
BATCH_SIZE        = 8          # physical batch size (GPU memory constraint)
ACCUM_STEPS       = 4          # gradient accumulation → effective BS = 32
PHASE1_EPOCHS     = 30         # frozen-backbone phase
PHASE2_EPOCHS     = 20         # partial-unfreeze phase
PHASE2_MIN_F1     = 0.30       # only enter phase 2 if phase 1 F1 exceeds this
PHASE1_LR         = 1e-3       # LR for new layers in phase 1
PHASE2_HEAD_LR    = 1e-4       # LR for new layers in phase 2
PHASE2_BACKBONE_LR= 1e-5       # LR for unfrozen backbone layers in phase 2
WEIGHT_DECAY      = 1e-4
LABEL_SMOOTHING   = 0.1
EARLY_STOP_PATIENCE = 10       # patience in epochs (measured on val_F1)
FRAMES_PER_VIDEO  = 15
SAVE_DIR          = 'checkpoints/sequential'
SDFVD_PATH        = './SDFVD'


# ── Loss ─────────────────────────────────────────────────────────────────────
class SmoothedBCE(nn.Module):
    """Binary cross-entropy with label smoothing and auxiliary branch losses."""

    def __init__(self, smoothing: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.smoothing = smoothing
        self.alpha = alpha
        self.bce = nn.BCELoss()

    def forward(self, predictions: dict, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        smooth = targets * (1 - self.smoothing) + self.smoothing * 0.5

        eps = 1e-7
        main_loss = self.bce(
            predictions['output'].clamp(eps, 1 - eps), smooth)

        if self.alpha > 0:
            gan_loss = self.bce(
                predictions['gan_classification'].clamp(eps, 1 - eps), smooth)
            vit_loss = self.bce(
                predictions['vit_classification'].clamp(eps, 1 - eps), smooth)
            return main_loss + self.alpha * (gan_loss + vit_loss)
        return main_loss


# ── Trainer ──────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, model: SequentialGANViT, device: torch.device,
                 head_lr: float, backbone_lr: float = 0.0,
                 weight_decay: float = WEIGHT_DECAY):
        self.model  = model.to(device)
        self.device = device
        self.scaler = GradScaler(enabled=device.type == 'cuda')
        self.loss_fn = SmoothedBCE(smoothing=LABEL_SMOOTHING, alpha=0.2)

        # Parameter groups: backbone (if any unfrozen) + new layers
        backbone_ids = set()
        for name in ['gan_discriminator.backbone_features',
                     'gan_discriminator.backbone_pool',
                     'vision_transformer.encoder']:
            for pname, p in model.named_parameters():
                if pname.startswith(name):
                    backbone_ids.add(id(p))

        backbone_params = [p for p in model.parameters()
                           if id(p) in backbone_ids and p.requires_grad]
        new_params      = [p for p in model.parameters()
                           if id(p) not in backbone_ids and p.requires_grad]

        param_groups = [{'params': new_params, 'lr': head_lr}]
        if backbone_params and backbone_lr > 0:
            param_groups.append({'params': backbone_params, 'lr': backbone_lr})

        self.optimizer = optim.AdamW(
            param_groups, weight_decay=weight_decay, betas=(0.9, 0.999))

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in model.parameters())
        print(f"  Trainable params: {n_trainable:,} / {n_total:,} "
              f"({100*n_trainable/n_total:.1f}%)")

        self.history = {k: [] for k in
                        ['train_loss', 'val_loss', 'train_acc',
                         'val_acc', 'val_f1', 'val_precision', 'val_recall']}
        self.best_f1      = 0.0
        self.patience_ctr = 0

    def _set_scheduler(self, num_epochs: int):
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, num_epochs), eta_min=1e-6)

    # ── one epoch ─────────────────────────────────────────────────────────────
    def train_epoch(self, loader) -> tuple:
        self.model.train()
        total_loss = 0.0
        correct = total = 0
        self.optimizer.zero_grad()

        bar = tqdm(loader, desc="  Train", leave=False)
        for step, (images, labels) in enumerate(bar, 1):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.device.type == 'cuda'):
                preds = self.model(images)
                loss  = self.loss_fn(preds, labels) / ACCUM_STEPS

            self.scaler.scale(loss).backward()

            if step % ACCUM_STEPS == 0 or step == len(loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * ACCUM_STEPS
            out = preds['output'].squeeze(1) if preds['output'].dim() > 1 else preds['output']
            predicted = (out > 0.5).float()
            correct  += (predicted == labels).sum().item()
            total    += labels.size(0)
            bar.set_postfix(loss=f"{loss.item()*ACCUM_STEPS:.4f}",
                            acc=f"{100*correct/total:.1f}%")

        return total_loss / len(loader), correct / total if total else 0.0

    @torch.no_grad()
    def val_epoch(self, loader) -> tuple:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels in tqdm(loader, desc="  Val  ", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.device.type == 'cuda'):
                preds = self.model(images)
                loss  = self.loss_fn(preds, labels)

            total_loss += loss.item()
            out = preds['output'].squeeze(1) if preds['output'].dim() > 1 else preds['output']
            predicted = (out > 0.5).float()
            all_preds.extend(predicted.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        acc  = (all_preds == all_labels).mean() if len(all_preds) else 0.0
        has_both = len(np.unique(all_labels)) > 1

        prec = precision_score(all_labels, all_preds, zero_division=0) if has_both else 0.0
        rec  = recall_score(all_labels, all_preds, zero_division=0)    if has_both else 0.0
        f1   = f1_score(all_labels, all_preds, zero_division=0)        if has_both else 0.0

        return total_loss / len(loader), acc, prec, rec, f1

    # ── full phase ────────────────────────────────────────────────────────────
    def run_phase(self, train_loader, val_loader, num_epochs: int,
                  save_dir: str, phase_label: str) -> float:
        """Train for num_epochs, save best checkpoint. Returns best val_F1."""
        os.makedirs(save_dir, exist_ok=True)
        self._set_scheduler(num_epochs)

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = self.train_epoch(train_loader)
            vl_loss, vl_acc, prec, rec, f1 = self.val_epoch(val_loader)
            self.scheduler.step()

            self.history['train_loss'].append(tr_loss)
            self.history['val_loss'].append(vl_loss)
            self.history['train_acc'].append(tr_acc)
            self.history['val_acc'].append(vl_acc)
            self.history['val_f1'].append(f1)
            self.history['val_precision'].append(prec)
            self.history['val_recall'].append(rec)

            lr = self.optimizer.param_groups[0]['lr']
            print(f"  {phase_label} Ep {epoch:>3}/{num_epochs} "
                  f"({time.time()-t0:.0f}s)  "
                  f"TrLoss={tr_loss:.4f} TrAcc={tr_acc*100:.1f}%  "
                  f"VlLoss={vl_loss:.4f} VlAcc={vl_acc*100:.1f}%  "
                  f"F1={f1*100:.1f}%  P={prec*100:.1f}%  R={rec*100:.1f}%  "
                  f"LR={lr:.1e}")

            if f1 > self.best_f1:
                self.best_f1 = f1
                self.patience_ctr = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': vl_loss, 'val_acc': vl_acc,
                    'val_precision': prec, 'val_recall': rec, 'val_f1': f1,
                }, os.path.join(save_dir, 'SequentialGANViT_best.pth'))
                print(f"    ✓ Best checkpoint saved  (F1={f1*100:.2f}%)")
            else:
                self.patience_ctr += 1
                if self.patience_ctr >= EARLY_STOP_PATIENCE:
                    print(f"    Early stop (patience={EARLY_STOP_PATIENCE})  "
                          f"Best F1={self.best_f1*100:.2f}%")
                    break

        return self.best_f1

    def plot(self, save_path: str):
        ep = range(1, len(self.history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(ep, self.history['train_loss'], label='Train')
        axes[0].plot(ep, self.history['val_loss'],   label='Val')
        axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(ep, [x*100 for x in self.history['train_acc']], label='Train')
        axes[1].plot(ep, [x*100 for x in self.history['val_acc']],   label='Val')
        axes[1].set_title('Accuracy (%)'); axes[1].legend(); axes[1].grid(alpha=0.3)

        axes[2].plot(ep, [x*100 for x in self.history['val_f1']],        label='F1')
        axes[2].plot(ep, [x*100 for x in self.history['val_precision']], label='Precision')
        axes[2].plot(ep, [x*100 for x in self.history['val_recall']],    label='Recall')
        axes[2].set_title('Val Metrics (%)'); axes[2].legend(); axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved → {save_path}")
        plt.close()


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*65}")
    print("  SEQUENTIAL GAN+ViT — Two-Phase Training")
    print(f"{'='*65}")
    print(f"  Device : {device}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  Effective batch size: {BATCH_SIZE * ACCUM_STEPS}  "
          f"(physical {BATCH_SIZE} × {ACCUM_STEPS} accum steps)")

    if not os.path.exists(SDFVD_PATH):
        print(f"\nError: dataset not found at {SDFVD_PATH}")
        return

    print(f"\nLoading dataset from {SDFVD_PATH} ...")
    train_loader, val_loader = create_dataloaders(
        data_dir=SDFVD_PATH,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        num_workers=0,          # 0 is safest on Windows
        train_split=0.8,
        frames_per_video=FRAMES_PER_VIDEO,
        use_weighted_sampler=True,
    )
    print(f"  Train: {len(train_loader.dataset):,}  |  Val: {len(val_loader.dataset):,}")

    model = SequentialGANViT(
        img_size=IMG_SIZE, gan_feature_dim=512,
        vit_embed_dim=768, num_heads=12, num_layers=12, dropout=0.2)

    # ── Phase 1: Frozen backbone ───────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  PHASE 1 — Frozen backbones (head-only training)")
    print(f"{'─'*65}")
    model.gan_discriminator.freeze_backbone()
    model.vision_transformer.freeze_backbone()

    trainer = Trainer(model, device, head_lr=PHASE1_LR)
    best_f1_p1 = trainer.run_phase(
        train_loader, val_loader,
        num_epochs=PHASE1_EPOCHS,
        save_dir=SAVE_DIR,
        phase_label="P1"
    )

    # ── Phase 2: Partial unfreeze ──────────────────────────────────────────
    if best_f1_p1 >= PHASE2_MIN_F1:
        print(f"\n{'─'*65}")
        print(f"  PHASE 2 — Partial backbone unfreeze  (P1 F1={best_f1_p1*100:.1f}%)")
        print(f"{'─'*65}")

        # Load best phase-1 weights before continuing
        ckpt = torch.load(os.path.join(SAVE_DIR, 'SequentialGANViT_best.pth'),
                          map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)

        model.gan_discriminator.unfreeze_top_blocks(num_blocks=2)
        model.vision_transformer.unfreeze_top_blocks(num_blocks=4)

        trainer2 = Trainer(model, device,
                           head_lr=PHASE2_HEAD_LR,
                           backbone_lr=PHASE2_BACKBONE_LR)
        trainer2.history = {k: v[:] for k, v in trainer.history.items()}
        trainer2.best_f1 = trainer.best_f1
        trainer2.patience_ctr = 0

        trainer2.run_phase(
            train_loader, val_loader,
            num_epochs=PHASE2_EPOCHS,
            save_dir=SAVE_DIR,
            phase_label="P2"
        )
        trainer2.plot('sequential_training_history.png')
        final_f1 = trainer2.best_f1
    else:
        print(f"\n  Phase 1 F1 ({best_f1_p1*100:.1f}%) below threshold "
              f"({PHASE2_MIN_F1*100:.0f}%) — skipping Phase 2.")
        trainer.plot('sequential_training_history.png')
        final_f1 = best_f1_p1

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best Val F1 : {final_f1*100:.2f}%")
    print(f"  Checkpoint  : {SAVE_DIR}/SequentialGANViT_best.pth")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
