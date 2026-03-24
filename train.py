import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

# Import our models
from models.sequential_model import SequentialGANViT
from models.parallel_model import ParallelGANViT
from utils.data_utils import create_dataloaders, get_transforms


def _get_param_groups(model, backbone_lr, head_lr):
    """
    Return two parameter groups:
      - pretrained backbone parameters  → smaller LR (backbone_lr)
      - new fusion / classification layers → larger LR (head_lr)
    Fine-tuning pretrained weights at a lower rate prevents destroying
    the ImageNet representations while allowing the new heads to adapt quickly.
    """
    backbone_ids = set()
    for submodule_name in ['gan_discriminator.backbone_features',
                           'gan_discriminator.backbone_pool',
                           'vision_transformer.encoder']:
        for name, param in model.named_parameters():
            if name.startswith(submodule_name):
                backbone_ids.add(id(param))

    backbone_params = [p for p in model.parameters() if id(p) in backbone_ids]
    new_params      = [p for p in model.parameters() if id(p) not in backbone_ids]

    return [
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': new_params,      'lr': head_lr},
    ]


class Trainer:
    """Trainer for deepfake detection models with pretrained backbones."""

    def __init__(self, model, device, learning_rate=1e-4, weight_decay=1e-4, gradient_clip=1.0):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip

        # Differential learning rates: backbone gets 10× smaller LR
        param_groups = _get_param_groups(model, backbone_lr=learning_rate * 0.1,
                                         head_lr=learning_rate)

        # AdamW is the standard choice for transformer-based models
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Cosine annealing schedule (set after num_epochs is known in train())
        self.scheduler = None
        self.warmup_epochs = 3

        # Training history
        self.train_losses      = []
        self.val_losses        = []
        self.train_accuracies  = []
        self.val_accuracies    = []
        self.val_precisions    = []
        self.val_recalls       = []
        self.val_f1_scores     = []

        # Early stopping
        self.best_val_acc     = 0.0
        self.patience_counter = 0
        self.early_stop_patience = 15

    def _warmup_lr(self, epoch):
        """Linear warmup for the first warmup_epochs epochs."""
        scale = (epoch + 1) / self.warmup_epochs
        for i, group in enumerate(self.optimizer.param_groups):
            if i == 0:
                group['lr'] = self.learning_rate * 0.1 * scale
            else:
                group['lr'] = self.learning_rate * scale

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct    = 0
        total      = 0

        progress_bar = tqdm(train_loader, desc="Training")

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(images)
            loss = self.model.compute_loss(predictions, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: invalid loss at batch {batch_idx}, skipping.")
                continue

            loss.backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()

            total_loss += loss.item()
            outputs = predictions['output']
            outputs_sq = outputs.squeeze(1) if outputs.dim() > 1 else outputs
            predicted = (outputs_sq > 0.5).float()
            correct  += (predicted == labels).sum().item()
            total    += labels.size(0)

            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc':  f'{100 * correct / total:.2f}%'
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss   = 0
        all_preds    = []
        all_labels   = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                predictions = self.model(images)
                loss = self.model.compute_loss(predictions, labels)
                total_loss += loss.item()

                outputs = predictions['output']
                outputs_sq = outputs.squeeze(1) if outputs.dim() > 1 else outputs
                predicted = (outputs_sq > 0.5).float()

                all_preds.extend(predicted.cpu().numpy().flatten().tolist())
                all_labels.extend(labels.cpu().numpy().flatten().tolist())

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)

        avg_loss = total_loss / len(val_loader)
        accuracy = (all_preds == all_labels).mean() if len(all_preds) > 0 else 0.0

        if len(all_preds) > 0 and len(np.unique(all_labels)) > 1:
            precision = precision_score(all_labels, all_preds, zero_division=0)
            recall    = recall_score(all_labels, all_preds, zero_division=0)
            f1        = f1_score(all_labels, all_preds, zero_division=0)
        else:
            precision = recall = f1 = 0.0

        return avg_loss, accuracy, precision, recall, f1

    def train(self, train_loader, val_loader, num_epochs=50, save_dir='checkpoints'):
        """Main training loop."""
        os.makedirs(save_dir, exist_ok=True)

        # Cosine annealing after warmup
        cosine_epochs = max(1, num_epochs - self.warmup_epochs)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cosine_epochs, eta_min=1e-6
        )

        print(f"Starting training for {num_epochs} epochs — {self.model.__class__.__name__}")
        print(f"Device: {self.device}  |  Base LR: {self.learning_rate}  |  Backbone LR: {self.learning_rate * 0.1}")
        print("-" * 60)

        for epoch in range(num_epochs):
            start = time.time()

            # Learning rate schedule
            if epoch < self.warmup_epochs:
                self._warmup_lr(epoch)
            else:
                self.scheduler.step()

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_p, val_r, val_f1 = self.validate(val_loader)

            # History
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.val_precisions.append(val_p)
            self.val_recalls.append(val_r)
            self.val_f1_scores.append(val_f1)

            elapsed = time.time() - start
            current_lr = self.optimizer.param_groups[1]['lr']  # head LR
            print(f"Epoch {epoch+1:>3}/{num_epochs} ({elapsed:.1f}s)  "
                  f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc*100:.2f}%  "
                  f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc*100:.2f}%  "
                  f"F1: {val_f1*100:.2f}%  LR: {current_lr:.2e}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc   = val_acc
                self.patience_counter = 0
                model_name = f"{self.model.__class__.__name__}_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc':  val_acc,
                    'val_precision': val_p,
                    'val_recall':    val_r,
                    'val_f1':        val_f1
                }, os.path.join(save_dir, model_name))
                print(f"  ✓ Best model saved — Val Acc: {val_acc*100:.2f}%")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}. "
                          f"Best Val Acc: {self.best_val_acc*100:.2f}%")
                    break

            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                ckpt_name = f"{self.model.__class__.__name__}_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_acc': val_acc
                }, os.path.join(save_dir, ckpt_name))

    def plot_training_history(self, save_path='training_history.png'):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(self.train_losses, label='Train', color='blue')
        axes[0, 0].plot(self.val_losses,   label='Val',   color='red')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot([x*100 for x in self.train_accuracies], label='Train', color='blue')
        axes[0, 1].plot([x*100 for x in self.val_accuracies],   label='Val',   color='red')
        axes[0, 1].set_title('Accuracy (%)')
        axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot([x*100 for x in self.val_precisions], label='Precision', color='green')
        axes[1, 0].plot([x*100 for x in self.val_recalls],    label='Recall',    color='orange')
        axes[1, 0].plot([x*100 for x in self.val_f1_scores],  label='F1',        color='purple')
        axes[1, 0].set_title('Validation Metrics (%)')
        axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot([x*100 for x in self.val_accuracies], label='Accuracy', color='red')
        axes[1, 1].plot([x*100 for x in self.val_f1_scores],  label='F1',       color='purple')
        axes[1, 1].set_title('Accuracy vs F1 (%)')
        axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
        plt.close()


def main():
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Hyperparameters
    img_size         = 224
    batch_size       = 8         # Keep small for stability on limited GPU RAM
    num_epochs       = 50
    learning_rate    = 1e-4      # Head LR; backbone will get 1e-5
    frames_per_video = 15

    sdfvd_path = './SDFVD'
    if not os.path.exists(sdfvd_path):
        print(f"Error: SDFVD dataset not found at {sdfvd_path}")
        return

    print(f"\nLoading dataset from {sdfvd_path}...")
    try:
        train_loader, val_loader = create_dataloaders(
            data_dir=sdfvd_path,
            batch_size=batch_size,
            img_size=img_size,
            num_workers=2,
            train_split=0.8,
            frames_per_video=frames_per_video
        )
        print(f"Train: {len(train_loader.dataset)} samples  |  Val: {len(val_loader.dataset)} samples")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return

    # ── Sequential Model ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TRAINING SEQUENTIAL GAN+ViT MODEL (EfficientNet-B4 + ViT-B/16)")
    print("="*60)

    sequential_model = SequentialGANViT(
        img_size=img_size,
        gan_feature_dim=512,
        vit_embed_dim=768,
        num_heads=12,
        num_layers=12,
        dropout=0.1
    )

    n_params = sum(p.numel() for p in sequential_model.parameters())
    print(f"Parameters: {n_params:,}")

    seq_trainer = Trainer(
        model=sequential_model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        gradient_clip=1.0
    )
    seq_trainer.train(train_loader, val_loader, num_epochs, 'checkpoints/sequential')
    seq_trainer.plot_training_history('sequential_training_history.png')

    # ── Parallel Model ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TRAINING PARALLEL GAN+ViT MODEL (EfficientNet-B4 + ViT-B/16)")
    print("="*60)

    parallel_model = ParallelGANViT(
        img_size=img_size,
        gan_feature_dim=512,
        vit_embed_dim=768,
        num_heads=12,
        num_layers=12,
        dropout=0.1
    )

    n_params = sum(p.numel() for p in parallel_model.parameters())
    print(f"Parameters: {n_params:,}")

    par_trainer = Trainer(
        model=parallel_model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        gradient_clip=1.0
    )
    par_trainer.train(train_loader, val_loader, num_epochs, 'checkpoints/parallel')
    par_trainer.plot_training_history('parallel_training_history.png')

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Sequential Accuracy : {seq_trainer.best_val_acc*100:.2f}%")
    print(f"Best Parallel Accuracy   : {par_trainer.best_val_acc*100:.2f}%")

    for name, trainer in [("Sequential", seq_trainer), ("Parallel", par_trainer)]:
        if trainer.val_f1_scores:
            print(f"\n{name} — Final epoch metrics:")
            print(f"  Accuracy  : {trainer.val_accuracies[-1]*100:.2f}%")
            print(f"  Precision : {trainer.val_precisions[-1]*100:.2f}%")
            print(f"  Recall    : {trainer.val_recalls[-1]*100:.2f}%")
            print(f"  F1 Score  : {trainer.val_f1_scores[-1]*100:.2f}%")


if __name__ == "__main__":
    main()
