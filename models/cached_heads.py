"""
cached_heads.py — Head-only models that consume cached encoder features.

Same fusion + classifier architecture as SequentialGANViT / ParallelGANViT, but
inputs are pre-encoded (gan_raw_feat: 1792-dim, vit_raw_feat: 768-dim) so the
EfficientNet-B4 and ViT-B/16 backbones are never executed during training.

Output dict keys mirror the originals so loss / metric code is reusable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GANProjector(nn.Module):
    """Projects raw EfficientNet-B4 1792-dim feature → gan_feature_dim, with aux head.

    Outputs raw logits (no Sigmoid) — pair with BCEWithLogitsLoss for stability.
    """

    def __init__(self, raw_dim: int = 1792, feature_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(raw_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, gan_raw):
        feats = self.feature_extractor(gan_raw)
        return feats, self.classifier(feats)


class _ViTProjector(nn.Module):
    """Projects raw ViT-B/16 768-dim CLS → embed_dim/2, with aux head (logits)."""

    def __init__(self, raw_dim: int = 768, embed_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        out = embed_dim // 2
        self.feature_extractor = nn.Sequential(
            nn.Linear(raw_dim, out),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(out, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, vit_raw):
        feats = self.feature_extractor(vit_raw)
        return feats, self.classifier(feats)


class _CrossAttention(nn.Module):
    """Bidirectional cross-attention — mirrors models.parallel_model.CrossAttention."""

    def __init__(self, gan_feature_dim, vit_feature_dim, attention_dim=256):
        super().__init__()
        self.attention_dim = attention_dim
        self.gan_q = nn.Linear(gan_feature_dim, attention_dim)
        self.gan_k = nn.Linear(gan_feature_dim, attention_dim)
        self.gan_v = nn.Linear(gan_feature_dim, attention_dim)
        self.vit_q = nn.Linear(vit_feature_dim, attention_dim)
        self.vit_k = nn.Linear(vit_feature_dim, attention_dim)
        self.vit_v = nn.Linear(vit_feature_dim, attention_dim)
        self.gan_out = nn.Linear(attention_dim, gan_feature_dim)
        self.vit_out = nn.Linear(attention_dim, vit_feature_dim)
        self.gan_norm = nn.LayerNorm(gan_feature_dim)
        self.vit_norm = nn.LayerNorm(vit_feature_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, gan_f, vit_f):
        gq = self.gan_q(gan_f).unsqueeze(1)
        vk = self.vit_k(vit_f).unsqueeze(1)
        vv = self.vit_v(vit_f).unsqueeze(1)
        vq = self.vit_q(vit_f).unsqueeze(1)
        gk = self.gan_k(gan_f).unsqueeze(1)
        gv = self.gan_v(gan_f).unsqueeze(1)

        scale = self.attention_dim ** 0.5
        g2v = self.dropout(F.softmax(torch.matmul(gq, vk.transpose(-2, -1)) / scale, dim=-1))
        gan_enh = torch.matmul(g2v, vv).squeeze(1)
        v2g = self.dropout(F.softmax(torch.matmul(vq, gk.transpose(-2, -1)) / scale, dim=-1))
        vit_enh = torch.matmul(v2g, gv).squeeze(1)

        return (
            self.gan_norm(gan_f + self.gan_out(gan_enh)),
            self.vit_norm(vit_f + self.vit_out(vit_enh)),
        )


def _init_new_layers(*modules):
    """Xavier-uniform for Linear (well-suited for GELU/sigmoid downstream)."""
    for m in modules:
        for layer in m.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, (nn.BatchNorm1d, nn.LayerNorm)):
                if layer.weight is not None:
                    nn.init.constant_(layer.weight, 1.0)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)


class SequentialHead(nn.Module):
    """Head-only Sequential GAN+ViT model. Inputs: cached (gan_raw, vit_raw).

    LayerNorm replaces BatchNorm1d so eval-time stats don't depend on training-batch
    statistics — important when train uses mixup and val does not.
    Output is raw logits; pair with BCEWithLogitsLoss.
    """

    def __init__(self, gan_raw_dim=1792, vit_raw_dim=768,
                 gan_feature_dim=512, vit_embed_dim=768, dropout=0.3):
        super().__init__()
        self.gan_proj = _GANProjector(gan_raw_dim, gan_feature_dim, dropout)
        self.vit_proj = _ViTProjector(vit_raw_dim, vit_embed_dim, dropout)

        vit_feat_dim = vit_embed_dim // 2
        fusion_in = gan_feature_dim + vit_feat_dim
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_in, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout * 0.5),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )
        _init_new_layers(self.gan_proj, self.vit_proj, self.feature_fusion, self.classifier)

    def forward(self, gan_raw, vit_raw):
        gan_f, gan_cls = self.gan_proj(gan_raw)
        vit_f, vit_cls = self.vit_proj(vit_raw)
        fused = self.feature_fusion(torch.cat([gan_f, vit_f], dim=1))
        return {
            'output': self.classifier(fused),
            'gan_features': gan_f,
            'vit_features': vit_f,
            'gan_classification': gan_cls,
            'vit_classification': vit_cls,
        }


class ParallelHead(nn.Module):
    """Head-only Parallel GAN+ViT model with bidirectional cross-attention.

    LayerNorm replaces BatchNorm1d. Output is raw logits.
    """

    def __init__(self, gan_raw_dim=1792, vit_raw_dim=768,
                 gan_feature_dim=512, vit_embed_dim=768, dropout=0.3):
        super().__init__()
        self.gan_proj = _GANProjector(gan_raw_dim, gan_feature_dim, dropout)
        self.vit_proj = _ViTProjector(vit_raw_dim, vit_embed_dim, dropout)

        vit_feat_dim = vit_embed_dim // 2
        self.cross_attn = _CrossAttention(gan_feature_dim, vit_feat_dim, attention_dim=256)

        fusion_in = gan_feature_dim + vit_feat_dim
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_in, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout * 0.5),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )
        _init_new_layers(self.gan_proj, self.vit_proj, self.cross_attn,
                         self.feature_fusion, self.classifier)

    def forward(self, gan_raw, vit_raw):
        gan_f, gan_cls = self.gan_proj(gan_raw)
        vit_f, vit_cls = self.vit_proj(vit_raw)
        gan_enh, vit_enh = self.cross_attn(gan_f, vit_f)
        combined = torch.cat([gan_enh, vit_enh], dim=1)
        fused = self.feature_fusion(combined)
        return {
            'output': self.classifier(fused),
            'gan_features': gan_f,
            'vit_features': vit_f,
            'gan_classification': gan_cls,
            'vit_classification': vit_cls,
        }
