import torch
import torch.nn as nn
import torch.nn.functional as F
from .gan_discriminator import GANDiscriminator
from .vision_transformer import VisionTransformer


class SequentialGANViT(nn.Module):
    """
    Sequential GAN + ViT Model for Deepfake Detection.

    Stage 1: EfficientNet-B4 (GAN Discriminator) — fine details, low-level artifacts.
    Stage 2: ViT-B/16 (Vision Transformer) — global context, structural patterns.
    Both use pretrained ImageNet weights for strong feature representations.
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 gan_feature_dim=512, vit_embed_dim=768, num_heads=12,
                 num_layers=12, dropout=0.1):
        super(SequentialGANViT, self).__init__()

        # GAN Discriminator — pretrained EfficientNet-B4 backbone
        self.gan_discriminator = GANDiscriminator(
            input_channels=in_channels,
            feature_dim=gan_feature_dim
        )

        # Vision Transformer — pretrained ViT-B/16 backbone
        self.vision_transformer = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=vit_embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

        # Feature fusion: gan_feature_dim + vit_embed_dim // 2 = 512 + 384 = 896
        fusion_input_dim = gan_feature_dim + vit_embed_dim // 2
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Initialise only the new (non-pretrained) layers
        self._initialize_new_layers()

    def _initialize_new_layers(self):
        """Initialise fusion and classification layers. Pretrained backbone weights are kept."""
        for m in [self.feature_fusion, self.classifier]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)
                elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    nn.init.constant_(layer.weight, 1.0)
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        """
        Forward pass through the sequential model.

        Args:
            x: Input image tensor (batch_size, channels, height, width)

        Returns:
            dict with keys: output, gan_features, vit_features,
                            gan_classification, vit_classification, attention_map
        """
        # Stage 1: EfficientNet discriminator — fine detail analysis
        gan_features, gan_classification = self.gan_discriminator(x)

        # Stage 2: ViT — global context on the original image
        # NOTE: We feed the original (normalised) image directly to ViT.
        # Modifying normalised pixels with an attention mask produces out-of-distribution
        # inputs for the pretrained ViT, hurting performance.
        vit_features, vit_classification = self.vision_transformer(x)

        # Attention map for interpretability (not used in forward computation)
        gan_attention = self.gan_discriminator.get_attention_map(x)

        # Feature fusion
        combined_features = torch.cat([gan_features, vit_features], dim=1)
        fused_features = self.feature_fusion(combined_features)

        # Final classification
        final_output = self.classifier(fused_features)

        return {
            'output': final_output,
            'gan_features': gan_features,
            'vit_features': vit_features,
            'gan_classification': gan_classification,
            'vit_classification': vit_classification,
            'attention_map': gan_attention
        }

    def get_interpretability_maps(self, x):
        """Get interpretability maps for model explanation."""
        gan_attention = self.gan_discriminator.get_attention_map(x)
        vit_attention_maps = self.vision_transformer.get_attention_maps(x)

        with torch.no_grad():
            gan_features, _ = self.gan_discriminator(x)
            vit_features, _ = self.vision_transformer(x)

            feature_importance = torch.norm(gan_features, dim=1, keepdim=True)
            feature_importance = F.interpolate(
                feature_importance.unsqueeze(2),
                size=x.shape[2:], mode='bilinear', align_corners=False
            )

        return {
            'gan_attention': gan_attention,
            'vit_attention_maps': vit_attention_maps,
            'feature_importance': feature_importance
        }

    def compute_loss(self, predictions, targets, alpha=0.3):
        """
        Compute combined loss for training.

        Args:
            predictions: Model predictions dictionary
            targets: Ground truth labels
            alpha: Weight for auxiliary GAN / ViT losses (default 0.3)

        Returns:
            total_loss: Combined loss value
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        # Apply label smoothing to targets for better calibration
        smooth_eps = 0.1
        smooth_targets = targets * (1.0 - smooth_eps) + smooth_eps * 0.5

        eps = 1e-7

        # Primary loss — final fused output with focal weighting
        output = predictions['output']
        output_clamped = torch.clamp(output, eps, 1 - eps)
        bce = F.binary_cross_entropy(output_clamped, smooth_targets, reduction='none')
        pt = torch.where(targets == 1, output, 1 - output)
        focal_weight = (1 - pt) ** 2.0
        final_loss = (focal_weight * bce).mean()

        if torch.isnan(final_loss) or torch.isinf(final_loss):
            final_loss = torch.tensor(0.693, device=output.device)

        # Auxiliary losses from intermediate classifiers
        if alpha > 0:
            gan_loss = F.binary_cross_entropy(
                torch.clamp(predictions['gan_classification'], eps, 1 - eps),
                smooth_targets, reduction='mean'
            )
            vit_loss = F.binary_cross_entropy(
                torch.clamp(predictions['vit_classification'], eps, 1 - eps),
                smooth_targets, reduction='mean'
            )
            total_loss = final_loss + alpha * (gan_loss + vit_loss)
        else:
            total_loss = final_loss

        return total_loss
