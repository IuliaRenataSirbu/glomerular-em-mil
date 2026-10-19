# vit_backbone.py
# ─────────────────────────────────────────────────────────────────────────────
# Defines DINOv2MILClassifier and build_dinov2_model.
# Imported by model.py when cfg.BACKBONE = "dinov2_vits14".
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 ViT-S/14  (loaded via torch.hub — no architecture re-implementation)
# ─────────────────────────────────────────────────────────────────────────────

class DINOv2MILClassifier(nn.Module):
    """
    DINOv2 ViT-S/14 MIL classifier.

    Interface:
        encode(x)              → [B, embed_dim]
        aggregate(feats)       → logits, attn
        forward_image_batch(x) → [B, num_classes]
        forward_patient(imgs)  → logits, attn
    """

    def __init__(self, backbone: nn.Module, embed_dim: int,
                 num_classes: int, mil_mode: str = "attention",
                 tab_dim: int = 0, tab_proj_dim: int = 64):
        super().__init__()
        self.backbone  = backbone
        self.mil_mode  = mil_mode
        self.embed_dim = embed_dim
        self.tab_dim   = tab_dim

        if mil_mode == "attention":
            from model import AttentionMIL
            self.attention = AttentionMIL(embed_dim)
        elif mil_mode not in ("mean", "max"):
            raise ValueError(f"Unknown mil_mode '{mil_mode}'")

        if tab_dim > 0:
            self.tab_proj = nn.Sequential(nn.Linear(tab_dim, tab_proj_dim), nn.ReLU())
            head_in = embed_dim + tab_proj_dim
        else:
            head_in = embed_dim

        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(head_in, num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W] → [B, embed_dim]  (CLS token from DINOv2)"""
        if not next(self.backbone.parameters()).requires_grad:
            with torch.no_grad():
                return self.backbone(x)     # DINOv2 hub model returns CLS token directly
        return self.backbone(x)

    def aggregate(self, features: torch.Tensor, tab_feats=None):
        """features: [N, embed_dim] → logits [num_classes], attn [N] or None"""
        if self.mil_mode == "mean":
            pooled, attn = features.mean(0), None
        elif self.mil_mode == "max":
            pooled, attn = features.max(0)[0], None
        elif self.mil_mode == "attention":
            pooled, attn = self.attention(features)

        if self.tab_dim > 0 and tab_feats is not None:
            pooled = torch.cat([pooled, self.tab_proj(tab_feats)], dim=-1)

        logits = self.head(pooled)
        return logits, attn

    def forward_image_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Image-level forward: [B, 3, H, W] → [B, num_classes]"""
        return self.head(self.encode(x))

    def forward_patient(self, imgs: torch.Tensor):
        """Patient-level forward: [N, 3, H, W] → logits, attn"""
        feats        = self.encode(imgs)
        logits, attn = self.aggregate(feats.float())
        return logits, attn


def build_dinov2_model(checkpoint_path,
                       num_classes: int,
                       mil_mode: str = "attention",
                       freeze_backbone: bool = False,
                       device: str = "cpu",
                       tab_dim: int = 0,
                       tab_proj_dim: int = 64) -> DINOv2MILClassifier:
    """
    Loads DINOv2 ViT-S/14 from torch.hub, applies exported lightly_train weights,
    and wraps it in DINOv2MILClassifier.

    Args:
        checkpoint_path : path to the exported .pt file from lightly_train,
                          or one of the keyword strings:
                          "facebook" — official DINOv2 weights pretrained by
                                       Facebook on LVD-142M via SSL (torch.hub)
                          "random"   — random initialisation (no pretraining)
                          None       — same as "random"
        num_classes     : number of output classes
        mil_mode        : "attention" | "mean" | "max"
        freeze_backbone : True = frozen feature extractor, False = fine-tune
        device          : torch device string

    Returns:
        DINOv2MILClassifier ready for training or evaluation
    """
    print(f"\nLoading DINOv2 ViT-S/14 backbone:")
    print(f"  Checkpoint : {checkpoint_path}")

    # ── Load DINOv2 architecture from torch.hub ───────────────────────────────
    # This pulls the official Facebook Research DINOv2 implementation.
    # No architecture re-implementation needed — hub handles it.
    # ── Resolve init mode ─────────────────────────────────────────────────────
    if checkpoint_path in (None, "random"):
        init_mode = "random"
    elif str(checkpoint_path) == "facebook":
        init_mode = "facebook"
    else:
        init_mode = "checkpoint"

    try:
        backbone = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            pretrained=(init_mode == "facebook"),  # True = Facebook SSL weights (LVD-142M)
            verbose=False,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load DINOv2 from torch.hub: {e}\n"
            f"Ensure internet access or that the hub cache is populated. "
            f"Alternatively, set TORCH_HOME to a directory containing the hub cache."
        ) from e

    embed_dim = backbone.embed_dim   # 384 for ViT-S

    # ── Load exported lightly_train weights ───────────────────────────────────
    if init_mode == "facebook":
        init_label = "Facebook SSL (LVD-142M via torch.hub)"
        print(f"  Weights loaded  : official DINOv2 pretrained weights (LVD-142M SSL)")
    elif init_mode == "checkpoint":
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # lightly_train.export(part="model") usually gives a flat state dict,
        # but handle both wrapped and unwrapped forms defensively
        if isinstance(ckpt, dict):
            state = (ckpt.get("state_dict")
                     or ckpt.get("model_state")
                     or ckpt.get("model")
                     or ckpt)
        else:
            state = ckpt

        # Strip any remaining prefix (e.g. "backbone.", "model.") if present
        cleaned = {}
        for k, v in state.items():
            for prefix in ("backbone.", "model.backbone.", "model."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            cleaned[k] = v

        missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
        print(f"  Weights loaded  : {len(cleaned) - len(missing)} / {len(cleaned)} keys matched")
        if missing:
            print(f"  Missing keys    : {len(missing)}  (e.g. {missing[:3]})")
        if unexpected:
            print(f"  Unexpected keys : {len(unexpected)}  (e.g. {unexpected[:3]})")
        init_label = f"lightly_train export ({Path(str(checkpoint_path)).name})"
    else:  # random
        init_label = "random (no pretraining)"
        print(f"  No checkpoint — random initialisation")

    # ── Freeze or unfreeze backbone ───────────────────────────────────────────
    for p in backbone.parameters():
        p.requires_grad = not freeze_backbone
    if freeze_backbone:
        backbone.eval()

    backbone = backbone.to(device)

    # ── Build classifier ──────────────────────────────────────────────────────
    model = DINOv2MILClassifier(
        backbone     = backbone,
        embed_dim    = embed_dim,
        num_classes  = num_classes,
        mil_mode     = mil_mode,
        tab_dim      = tab_dim,
        tab_proj_dim = tab_proj_dim,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    total     = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"  Architecture    : DINOv2 ViT-S/14  (embed_dim={embed_dim})")
    print(f"  Init            : {init_label}")
    print(f"  Backbone frozen : {freeze_backbone}")
    print(f"  MIL mode        : {mil_mode}")
    print(f"  Total params    : {total:.1f}M  |  Trainable: {trainable:.1f}M\n")

    return model


