import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GANDiscriminator(nn.Module):
    """
    GAN Discriminator using pretrained EfficientNet-B4 backbone for detecting
    fine details and low-level clues in deepfake images.
    Pretrained ImageNet weights provide strong feature extraction from the start.
    """

    def __init__(self, input_channels=3, feature_dim=512):
        super(GANDiscriminator, self).__init__()

        # Load pretrained EfficientNet-B4 backbone
        try:
            from torchvision.models import EfficientNet_B4_Weights
            efficientnet = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        except (AttributeError, ImportError):
            efficientnet = models.efficientnet_b4(pretrained=True)

        # Use EfficientNet feature extractor (output: 1792 channels after avgpool)
        self.backbone_features = efficientnet.features   # Spatial feature maps
        self.backbone_pool = efficientnet.avgpool         # AdaptiveAvgPool2d(1, 1)
        # EfficientNet-B4 outputs 1792 features after pooling
        backbone_out_dim = 1792

        # Feature extraction layers to project to feature_dim
        self.feature_extractor = nn.Sequential(
            nn.Linear(backbone_out_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # Binary classification head (real vs fake)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the discriminator.

        Args:
            x: Input image tensor of shape (batch_size, channels, height, width)

        Returns:
            features: Extracted features (batch_size, feature_dim) for fusion with ViT
            classification: Binary classification output (batch_size, 1)
        """
        # Extract spatial features via pretrained EfficientNet backbone
        spatial_features = self.backbone_features(x)        # (B, 1792, H', W')
        pooled = self.backbone_pool(spatial_features)        # (B, 1792, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)             # (B, 1792)

        # Project to feature_dim
        features = self.feature_extractor(pooled)            # (B, feature_dim)

        # Classification
        classification = self.classifier(features)           # (B, 1)

        return features, classification

    def freeze_backbone(self):
        """Freeze EfficientNet backbone — only train projection/classifier heads."""
        for param in self.backbone_features.parameters():
            param.requires_grad = False
        for param in self.backbone_pool.parameters():
            param.requires_grad = False

    def unfreeze_top_blocks(self, num_blocks: int = 2):
        """Unfreeze the last num_blocks feature blocks for fine-tuning phase 2."""
        blocks = list(self.backbone_features.children())
        for block in blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    def get_attention_map(self, x):
        """
        Generate attention map highlighting suspicious regions using
        the spatial activation maps from the EfficientNet backbone.

        Args:
            x: Input image tensor of shape (B, C, H, W)

        Returns:
            attention_map: Attention map of shape (B, 1, H', W')
        """
        with torch.no_grad():
            # Get spatial feature maps from the last conv block
            spatial_features = self.backbone_features(x)    # (B, 1792, H', W')

        # Average across channels to get a single spatial attention map
        attention = spatial_features.mean(dim=1, keepdim=True)  # (B, 1, H', W')

        # Normalize attention map via softmax over spatial dims
        b, c, h, w = attention.shape
        attn_flat = attention.view(b, c, h * w)
        attn_flat = F.softmax(attn_flat, dim=2)
        attention = attn_flat.view(b, c, h, w)

        return attention
