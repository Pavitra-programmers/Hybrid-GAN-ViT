import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VisionTransformer(nn.Module):
    """
    Vision Transformer using pretrained ViT-B/16 backbone for global context analysis.
    Pretrained ImageNet weights dramatically improve feature quality over training from scratch.
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768,
                 num_heads=12, num_layers=12, mlp_ratio=4.0, dropout=0.1, num_classes=1):
        super(VisionTransformer, self).__init__()

        # Load pretrained ViT-B/16 backbone
        try:
            from torchvision.models import ViT_B_16_Weights
            vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        except (AttributeError, ImportError):
            vit = models.vit_b_16(pretrained=True)

        # Remove the classification head — we will replace it
        # ViT-B/16 outputs 768-dim CLS token features
        vit.heads = nn.Identity()
        self.encoder = vit  # Output shape after forward: (B, 768)

        vit_out_dim = 768   # ViT-B/16 hidden dim

        # Feature extraction for fusion (project to embed_dim // 2)
        self.feature_extractor = nn.Sequential(
            nn.Linear(vit_out_dim, embed_dim // 2),   # 768 -> 384
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim // 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the Vision Transformer.

        Args:
            x: Input image tensor of shape (batch_size, channels, height, width)

        Returns:
            features: Extracted features (batch_size, embed_dim // 2) for fusion
            classification: Binary classification output (batch_size, 1)
        """
        # Extract CLS token features via pretrained ViT-B/16
        cls_features = self.encoder(x)               # (B, 768)

        # Project to fusion-compatible dimension
        features = self.feature_extractor(cls_features)  # (B, embed_dim // 2)

        # Classification
        classification = self.classifier(features)       # (B, 1)

        return features, classification

    def get_attention_maps(self, x):
        """
        Compatibility stub — returns empty list.
        Attention map extraction from the pretrained ViT encoder requires
        registering forward hooks; for the main training pipeline this is not needed.
        """
        return []
