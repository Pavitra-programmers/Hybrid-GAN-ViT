import torch
import torch.nn as nn
import torch.nn.functional as F
from .gan_discriminator import GANDiscriminator
from .vision_transformer import VisionTransformer


class ParallelGANViT(nn.Module):
    """
    Parallel GAN + ViT Model for Deepfake Detection.

    Both EfficientNet-B4 (GAN path) and ViT-B/16 (ViT path) process the image
    simultaneously, then bidirectional cross-attention fuses the features.
    Both backbones use pretrained ImageNet weights.
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 gan_feature_dim=512, vit_embed_dim=768, num_heads=12,
                 num_layers=12, dropout=0.1):
        super(ParallelGANViT, self).__init__()

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

        # vit_feature_dim is embed_dim // 2 = 384 (set inside VisionTransformer)
        vit_feature_dim = vit_embed_dim // 2   # 384

        # Cross-attention mechanism for bidirectional feature interaction
        self.cross_attention = CrossAttention(
            gan_feature_dim=gan_feature_dim,
            vit_feature_dim=vit_feature_dim,
            attention_dim=256
        )

        # Feature fusion: 512 + 384 = 896
        fusion_input_dim = gan_feature_dim + vit_feature_dim
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

        # Confidence estimator — defined here so variable names are in scope
        self.confidence_estimator = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Initialise only the new (non-pretrained) layers
        self._initialize_new_layers()

    def _initialize_new_layers(self):
        """Initialise fusion, classifier, and confidence estimator layers."""
        new_modules = [self.feature_fusion, self.classifier,
                       self.confidence_estimator, self.cross_attention]
        for module in new_modules:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)
                elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    nn.init.constant_(layer.weight, 1.0)
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        """
        Forward pass through the parallel model.

        Args:
            x: Input image tensor (batch_size, channels, height, width)

        Returns:
            dict with all predictions and intermediate features
        """
        # Parallel processing: both paths see the original image
        gan_features, gan_classification = self.gan_discriminator(x)
        vit_features, vit_classification = self.vision_transformer(x)

        # Bidirectional cross-attention
        enhanced_gan, enhanced_vit, attention_weights = self.cross_attention(
            gan_features, vit_features
        )

        # Feature fusion
        combined_features = torch.cat([enhanced_gan, enhanced_vit], dim=1)
        fused_features = self.feature_fusion(combined_features)

        # Final classification
        final_output = self.classifier(fused_features)

        # Confidence estimation
        confidence = self.confidence_estimator(combined_features)

        return {
            'output': final_output,
            'gan_features': gan_features,
            'vit_features': vit_features,
            'enhanced_gan_features': enhanced_gan,
            'enhanced_vit_features': enhanced_vit,
            'gan_classification': gan_classification,
            'vit_classification': vit_classification,
            'confidence': confidence,
            'attention_weights': attention_weights
        }

    def get_interpretability_maps(self, x):
        """Get interpretability maps for model explanation."""
        gan_attention = self.gan_discriminator.get_attention_map(x)
        vit_attention_maps = self.vision_transformer.get_attention_maps(x)

        with torch.no_grad():
            gan_features, _ = self.gan_discriminator(x)
            vit_features, _ = self.vision_transformer(x)
            _, _, cross_attention = self.cross_attention(gan_features, vit_features)

            gan_importance = torch.norm(gan_features, dim=1, keepdim=True)
            vit_importance = torch.norm(vit_features, dim=1, keepdim=True)

            gan_importance = F.interpolate(
                gan_importance.unsqueeze(2), size=x.shape[2:],
                mode='bilinear', align_corners=False
            )
            vit_importance = F.interpolate(
                vit_importance.unsqueeze(2), size=x.shape[2:],
                mode='bilinear', align_corners=False
            )

        return {
            'gan_attention': gan_attention,
            'vit_attention_maps': vit_attention_maps,
            'cross_attention': cross_attention,
            'gan_importance': gan_importance,
            'vit_importance': vit_importance
        }

    def compute_loss(self, predictions, targets, alpha=0.3, beta=0.1):
        """
        Compute combined loss for training.

        Args:
            predictions: Model predictions dictionary
            targets: Ground truth labels
            alpha: Weight for auxiliary GAN / ViT losses (default 0.3)
            beta: Weight for confidence loss (default 0.1)

        Returns:
            total_loss: Combined loss value
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        # Label smoothing
        smooth_eps = 0.1
        smooth_targets = targets * (1.0 - smooth_eps) + smooth_eps * 0.5

        eps = 1e-7
        output = predictions['output']
        output_clamped = torch.clamp(output, eps, 1 - eps)

        # Focal loss for primary output
        bce = F.binary_cross_entropy(output_clamped, smooth_targets, reduction='none')
        pt = torch.where(targets == 1, output, 1 - output)
        focal_weight = (1 - pt) ** 2.0
        final_loss = (focal_weight * bce).mean()

        if torch.isnan(final_loss) or torch.isinf(final_loss):
            final_loss = torch.tensor(0.693, device=output.device)

        if alpha > 0 or beta > 0:
            gan_loss = F.binary_cross_entropy(
                torch.clamp(predictions['gan_classification'], eps, 1 - eps),
                smooth_targets, reduction='mean'
            )
            vit_loss = F.binary_cross_entropy(
                torch.clamp(predictions['vit_classification'], eps, 1 - eps),
                smooth_targets, reduction='mean'
            )
            confidence_loss = F.binary_cross_entropy(
                torch.clamp(predictions['confidence'], eps, 1 - eps),
                smooth_targets, reduction='mean'
            )
            total_loss = (final_loss
                          + alpha * (gan_loss + vit_loss)
                          + beta * confidence_loss)
        else:
            total_loss = final_loss

        return total_loss


class CrossAttention(nn.Module):
    """
    Bidirectional cross-attention between GAN (EfficientNet) and ViT features.
    """

    def __init__(self, gan_feature_dim, vit_feature_dim, attention_dim):
        super(CrossAttention, self).__init__()

        self.attention_dim = attention_dim

        # GAN → attention space
        self.gan_query = nn.Linear(gan_feature_dim, attention_dim)
        self.gan_key   = nn.Linear(gan_feature_dim, attention_dim)
        self.gan_value = nn.Linear(gan_feature_dim, attention_dim)

        # ViT → attention space
        self.vit_query = nn.Linear(vit_feature_dim, attention_dim)
        self.vit_key   = nn.Linear(vit_feature_dim, attention_dim)
        self.vit_value = nn.Linear(vit_feature_dim, attention_dim)

        # Output projections back to original dims
        self.gan_output = nn.Linear(attention_dim, gan_feature_dim)
        self.vit_output = nn.Linear(attention_dim, vit_feature_dim)

        # Layer normalization
        self.gan_norm = nn.LayerNorm(gan_feature_dim)
        self.vit_norm = nn.LayerNorm(vit_feature_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, gan_features, vit_features):
        """
        Args:
            gan_features: (B, gan_feature_dim)
            vit_features: (B, vit_feature_dim)

        Returns:
            enhanced_gan, enhanced_vit, attention_weights dict
        """
        # Add sequence dimension: (B, 1, dim)
        gan_q = self.gan_query(gan_features).unsqueeze(1)
        vit_k = self.vit_key(vit_features).unsqueeze(1)
        vit_v = self.vit_value(vit_features).unsqueeze(1)

        vit_q = self.vit_query(vit_features).unsqueeze(1)
        gan_k = self.gan_key(gan_features).unsqueeze(1)
        gan_v = self.gan_value(gan_features).unsqueeze(1)

        # GAN attends to ViT
        scale = self.attention_dim ** 0.5
        gan_to_vit_scores   = torch.matmul(gan_q, vit_k.transpose(-2, -1)) / scale
        gan_to_vit_weights  = self.dropout(F.softmax(gan_to_vit_scores, dim=-1))
        gan_enhanced        = torch.matmul(gan_to_vit_weights, vit_v).squeeze(1)

        # ViT attends to GAN
        vit_to_gan_scores   = torch.matmul(vit_q, gan_k.transpose(-2, -1)) / scale
        vit_to_gan_weights  = self.dropout(F.softmax(vit_to_gan_scores, dim=-1))
        vit_enhanced        = torch.matmul(vit_to_gan_weights, gan_v).squeeze(1)

        # Residual + LayerNorm
        enhanced_gan = self.gan_norm(gan_features + self.gan_output(gan_enhanced))
        enhanced_vit = self.vit_norm(vit_features + self.vit_output(vit_enhanced))

        attention_weights = {
            'gan_to_vit': gan_to_vit_weights,
            'vit_to_gan': vit_to_gan_weights
        }

        return enhanced_gan, enhanced_vit, attention_weights
