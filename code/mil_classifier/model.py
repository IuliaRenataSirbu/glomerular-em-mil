# model.py
# ─────────────────────────────────────────────────────────────────────────────
# Defines two model classes and one factory function:
#
#   AttentionMIL    – gated attention pooling module (Ilse et al. 2018).
#                     Learns a per-image importance weight and returns a
#                     weighted-average feature vector for the patient bag.
#                     Only used by MILClassifier.
#
#   ImageClassifier – ResNet50 backbone + classification head. No MIL module.
#                     Used for TRAINING_MODE = "image" (Approach 1).
#                     Standard supervised image classification — every image
#                     is an independent sample with its patient's label.
#
#   MILClassifier   – ResNet50 backbone + AttentionMIL + classification head.
#                     Used for TRAINING_MODE = "patient".
#
#   build_model()   – factory function: reads cfg.TRAINING_MODE and returns
#                     the correct model instance. This is the only function
#                     that main.py needs to import from here.
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights  # pretrained ResNet50 from torchvision

from config import cfg


# ─────────────────────────────────────────────────────────────────────────────
# SHARED BACKBONE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_backbone(pretrained: bool):
    """
    Loads ResNet50 and removes its ImageNet classification head.
    Returns (backbone, embed_dim) where embed_dim = 2048.
    Both model classes use this identical backbone setup.
    """
    weights   = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None    # ImageNet pretrained weights
    backbone  = resnet50(weights=weights)                                  # full ResNet50 from torchvision
    embed_dim = backbone.fc.in_features                                    # 2048 — size of the penultimate feature vector
    backbone.fc = nn.Identity()                                            # replace 1000-class head with identity; output is now [B, 2048]
    return backbone, embed_dim


# ─────────────────────────────────────────────────────────────────────────────
# GATED ATTENTION MIL  (Ilse et al. 2018)
# ─────────────────────────────────────────────────────────────────────────────

class AttentionMIL(nn.Module):
    """
    Gated attention pooling. Given a bag of N image feature vectors, produces
    one weighted-average vector representing the entire patient, along with
    per-image importance weights that can be used for interpretability.

    Architecture (two parallel branches whose outputs are multiplied):
        V branch : Linear → tanh   — encodes the feature content
        U branch : Linear → sigmoid — acts as a gate controlling information flow
        w layer  : Linear → scalar — scores each image after gating
        softmax  : normalises scores to sum to 1 across the bag
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        """
        Args:
            embed_dim  : input feature dimension (2048 for ResNet50)
            hidden_dim : internal attention network width (default 128)
        """
        super().__init__()
        self.V = nn.Linear(embed_dim, hidden_dim)               # tanh branch:    [embed_dim] → [hidden_dim]
        self.U = nn.Linear(embed_dim, hidden_dim)               # sigmoid branch: [embed_dim] → [hidden_dim]
        self.w = nn.Linear(hidden_dim, 1)                       # score layer:    [hidden_dim] → scalar

    def forward(self, H: torch.Tensor):
        """
        Args:
            H : [N, embed_dim] — one feature vector per image in the bag

        Returns:
            pooled : [embed_dim] — attention-weighted patient feature vector
            A      : [N]         — per-image attention weights (sum to 1)
        """
        A = torch.tanh(self.V(H)) * torch.sigmoid(self.U(H))   # gated activation: [N, hidden_dim]
        A = self.w(A).squeeze(-1)                               # raw attention logit per image: [N]
        A = torch.softmax(A, dim=0)                             # normalise across bag: [N], sums to 1
        pooled = (A.unsqueeze(-1) * H).sum(0)                   # weighted sum: [embed_dim]
        return pooled, A                                        # patient vector + per-image weights


# ─────────────────────────────────────────────────────────────────────────────
# APPROACH 1 MODEL — IMAGE CLASSIFIER
# Used when TRAINING_MODE = "image"
# ─────────────────────────────────────────────────────────────────────────────

class ImageClassifier(nn.Module):
    """
    Pure image-level classifier. No MIL, no attention module.

    Architecture:
        ResNet50 backbone → 2048-d feature → Dropout(0.3) → Linear(2048, num_classes)

    Every image is classified independently. The patient label is propagated
    to each image during training. At no point are images from the same patient
    aggregated — this is standard supervised image classification.

    This model has no forward_patient() or aggregate() method. Calling those
    would raise an AttributeError, which is intentional — it prevents
    accidentally running MIL-style inference on an image-level model.
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone, embed_dim = _build_backbone(pretrained)             # ResNet50, embed_dim = 2048
        self.head = nn.Sequential(
            nn.Dropout(0.3),                                               # 30% dropout to reduce overfitting
            nn.Linear(embed_dim, num_classes),                             # [2048] → [num_classes]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a batch of images.
        x       : [B, 3, 224, 224]
        returns : [B, 2048]
        """
        return self.backbone(x)                                            # fc=Identity, so output is [B, 2048]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass for image-level training and validation.
        x       : [B, 3, 224, 224]
        returns : [B, num_classes] logits
        """
        return self.head(self.encode(x))                                   # [B, 2048] → [B, num_classes]


# ─────────────────────────────────────────────────────────────────────────────
# APPROACH 2 MODEL — MIL CLASSIFIER
# Used when TRAINING_MODE = "patient"
# ─────────────────────────────────────────────────────────────────────────────

class MILClassifier(nn.Module):
    """
    Full MIL pipeline for TRAINING_MODE="patient".

    Architecture:
        ResNet50 backbone → 2048-d features per image
        → MIL aggregation (mean / max / attention) → single 2048-d patient vector
        → Dropout(0.3) → Linear(2048, num_classes)
    """

    def __init__(self, num_classes: int, mil_mode: str, pretrained: bool = True,
                 tab_dim: int = 0, tab_proj_dim: int = 64):
        """
        Args:
            num_classes  : number of output classes
            mil_mode     : "mean", "max", or "attention"
            pretrained   : whether to load ImageNet weights for the backbone
            tab_dim      : number of tabular marker features (0 = no fusion)
            tab_proj_dim : projection layer output dim before concatenation
        """
        super().__init__()

        # ── Backbone ─────────────────────────────────────────────────────────
        self.backbone, embed_dim = _build_backbone(pretrained)             # ResNet50, embed_dim = 2048
        self.embed_dim           = embed_dim                               # stored for reference

        # ── MIL Aggregation ───────────────────────────────────────────────────
        self.mil_mode = mil_mode                                           # which aggregation strategy to use
        if mil_mode == "attention":
            self.attention = AttentionMIL(embed_dim)                       # only instantiated for attention mode

        # ── Tabular Fusion ────────────────────────────────────────────────────
        self.tab_dim = tab_dim
        if tab_dim > 0:
            self.tab_proj = nn.Sequential(
                nn.Linear(tab_dim, tab_proj_dim),
                nn.ReLU(),
            )
            head_in = embed_dim + tab_proj_dim
        else:
            head_in = embed_dim

        # ── Classification Head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(head_in, num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a batch of images using the backbone.
        x       : [B, 3, 224, 224]
        returns : [B, 2048]
        """
        return self.backbone(x)                                            # [B, 2048]

    def aggregate(self, features: torch.Tensor, tab_feats=None):
        """
        Aggregate a bag of image features into one patient-level prediction.

        Args:
            features  : [N_imgs, 2048] — one feature vector per image in the bag
            tab_feats : [tab_dim] float tensor or None — tabular marker vector

        Returns:
            logits : [num_classes]   — patient-level class scores (before softmax)
            attn   : [N_imgs] or None — per-image attention weights (attention mode only)
        """
        if self.mil_mode == "mean":
            pooled = features.mean(0)
            attn   = None
        elif self.mil_mode == "max":
            pooled, _ = features.max(0)
            attn      = None
        elif self.mil_mode == "attention":
            pooled, attn = self.attention(features)
        else:
            raise ValueError(
                f"Unknown MIL_MODE '{self.mil_mode}'. Choose: 'mean', 'max', 'attention'."
            )

        if self.tab_dim > 0:
            if tab_feats is None:
                tab_feats = pooled.new_zeros(self.tab_dim)                 # zero vector if not provided
            tab_proj = self.tab_proj(tab_feats)
            pooled   = torch.cat([pooled, tab_proj], dim=-1)

        logits = self.head(pooled)
        return logits, attn

    def forward_image_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Image-level forward pass — no MIL aggregation.
        Each image is classified independently using the patient label.

        x       : [B, 3, 224, 224]
        returns : [B, num_classes] logits
        """
        feats  = self.encode(x)                                            # [B, 2048]
        logits = self.head(feats)                                          # [B, num_classes]
        return logits                                                      # no aggregation step

    def forward_patient(self, imgs: torch.Tensor):
        """
        Patient-level forward pass — full MIL aggregation.
        Used during training and validation in TRAINING_MODE="patient".

        imgs    : [N_imgs, 3, 224, 224] — all images for one patient
        returns : logits [num_classes], attn [N_imgs] or None
        """
        feats        = self.encode(imgs)                                   # [N_imgs, 2048]
        logits, attn = self.aggregate(feats)                               # [num_classes], [N_imgs] or None
        return logits, attn


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_model(tab_dim: int = 0) -> nn.Module:
    """
    Reads cfg.TRAINING_MODE and cfg.BACKBONE and returns the correct model.

    BACKBONE = "resnet50"    → ResNet50-based ImageClassifier / MILClassifier
    BACKBONE = "dinov2_vits14" → DINOv2MILClassifier loaded from cfg.VIT_CHECKPOINT
    tab_dim: number of tabular marker features (0 = no mid-level fusion)
    """
    backbone = getattr(cfg, "BACKBONE", "resnet50")

    # ── DINOv2 ViT-S/14 backbone (lightly_train exported weights) ───────────
    if backbone == "dinov2_vits14":
        from vit_backbone import build_dinov2_model
        freeze = getattr(cfg, "FREEZE_BACKBONE", False)
        ckpt   = getattr(cfg, "VIT_CHECKPOINT",  None)
        if ckpt is None:
            raise ValueError("cfg.VIT_CHECKPOINT must be set when BACKBONE='dinov2_vits14'")
        model = build_dinov2_model(
            checkpoint_path = cfg.VIT_CHECKPOINT,
            num_classes     = cfg.NUM_CLASSES,
            mil_mode        = cfg.MIL_MODE,
            freeze_backbone = freeze,
            device          = cfg.DEVICE,
            tab_dim         = tab_dim,
            tab_proj_dim    = cfg.TAB_PROJ_DIM,
        )
        freeze_str = "frozen" if freeze else "fine-tuned"
        print(f"  Model: DINOv2MILClassifier  (ViT-S/14, backbone {freeze_str})")
        print(f"         num_classes={cfg.NUM_CLASSES}, mil_mode={cfg.MIL_MODE}")
        print(f"         checkpoint={cfg.VIT_CHECKPOINT}")
        return model

    # ── ResNet50 backbone (default) ───────────────────────────────────────────
    if cfg.TRAINING_MODE == "image":
        model = ImageClassifier(
            num_classes = cfg.NUM_CLASSES,                                 # 7 (or 6 if Normal is removed)
            pretrained  = cfg.PRETRAINED,                                  # ImageNet weights
        )
        print(f"  Model: ImageClassifier  (ResNet50, backbone + head, no MIL)")
        print(f"         num_classes={cfg.NUM_CLASSES}, pretrained={cfg.PRETRAINED}")

    elif cfg.TRAINING_MODE == "patient":
        model = MILClassifier(
            num_classes  = cfg.NUM_CLASSES,
            mil_mode     = cfg.MIL_MODE,
            pretrained   = cfg.PRETRAINED,
            tab_dim      = tab_dim,
            tab_proj_dim = cfg.TAB_PROJ_DIM,
        )
        print(f"  Model: MILClassifier  (ResNet50, full MIL)")
        print(f"         num_classes={cfg.NUM_CLASSES}, mil_mode={cfg.MIL_MODE}, pretrained={cfg.PRETRAINED}")

    else:
        raise ValueError(
            f"Unknown TRAINING_MODE '{cfg.TRAINING_MODE}'. Choose: 'image', 'patient'."
        )

    return model