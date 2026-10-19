"""
explain.py
─────────────────────────────────────────────────────────────────────────────
Patient-level explainability for trained EM3 MIL models.  Four analysis
levels, all inference-only (no retraining):

  1  — Image importance   : max-pool importance score — which images drive
                            the patient-level prediction
  2  — Patch attention    : DINOv2 last-layer attention heatmap overlaid on
                            the top-K highest-importance images
  3b — Image vs tab split : gradient × feature norm — what fraction of the
                            final logit came from the image branch vs the
                            marker branch
  3a — Marker sensitivity : per-marker zero-out ablation ΔP — how much each
                            known marker shifts the predicted probabilities

Levels 3a and 3b are skipped automatically when TAB_FUSION=False.
Level 2 requires BACKBONE=dinov2_vits14 (skipped otherwise).

Usage:
    python explain.py \\
        --run-dir  checkpoints/<run_name>/  \\
        --folds    /path/to/folds.json      \\
        [--split   test]                   \\   # test | val | train
        [--fold    0]                      \\   # fold index for val/train
        [--top-k   5]                      \\   # images for attention maps
        [--levels  1 2 3a 3b]              \\   # subset to run
        [--out     explain_results/]       \\   # output directory
        [--n-patients 20]                  \\   # debug: limit patients
        [--encode-batch-size 32]               # backbone batch size
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap


# ─────────────────────────────────────────────────────────────────────────────
# EARLY CONFIG APPLY  (must run before dataset.py import builds CLASS_TO_IDX)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_config_early():
    """Apply YAML overrides to cfg before any local module is imported."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--config",  default=None)
    args, _ = parser.parse_known_args()

    config_path = args.config
    if config_path is None and args.run_dir:
        run_dir   = Path(args.run_dir)
        yml_files = sorted(run_dir.glob("*.yml"))
        if yml_files:
            config_path = str(yml_files[0])

    if config_path:
        from config import cfg as _cfg
        with open(config_path) as f:
            overrides = yaml.safe_load(f) or {}
        for k, v in overrides.items():
            if hasattr(_cfg, k):
                setattr(_cfg, k, v)
        print(f"  Config applied early from: {config_path}")

_apply_config_early()

sys.path.insert(0, str(Path(__file__).parent))

from config  import cfg
from dataset import (build_dataframe_v2,
                     load_tabular_features, get_transforms,
                     IDX_TO_CLASS, CLASS_TO_IDX)
from model   import build_model


# ─────────────────────────────────────────────────────────────────────────────
# DISEASE SHORT NAMES  (for plot axis labels)
# ─────────────────────────────────────────────────────────────────────────────

_SHORT_NAME_MAP: dict = {
    "IgA nephropathy":                               "IgAN",
    "Alport syndrome/thin membrane disease":          "Alport/TBMD",
    "Alport syndrome/thin basement membrane disease": "Alport/TBMD",
    "Alport/thin basement membrane disease":          "Alport/TBMD",
    "Membranous nephropathy":                         "MN",
    "Diabetic kidney disease":                        "DN",
    "Diabetic nephropathy":                           "DN",
    "Minimal change disease (MCD)":                   "MCD",
    "Minimal change disease":                         "MCD",
    "Lupus nephritis":                                "LN",
    "Normal":                                         "Normal",
}


def _short_name(name: str) -> str:
    """Return the clinical abbreviation for a disease class name.
    Falls back to the full name if the name is not in the lookup table."""
    return _SHORT_NAME_MAP.get(name, name)


_MARKER_DISPLAY = {
    "IgAbin": "IgA", "IgGbin": "IgG", "IgMbin": "IgM",
    "C1qbin": "C1q", "C3cbin": "C3c",
    "Kbin": "Kappa", "lambdabin": "Lambda", "Fbbin": "Fibrinogen",
}


def _clean_marker_name(name: str) -> str:
    """Strip '_corrected' suffix/infix from marker column names for display.

    E.g. 'C3_corrected' → 'C3',  'IgA_corrected' → 'IgA'.
    """
    cleaned = name.replace("_corrected", "").replace("corrected_", "").strip("_")
    return _MARKER_DISPLAY.get(cleaned, cleaned)


# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(path: Path, model, device):
    ckpt  = torch.load(path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    epoch = ckpt.get("epoch", -1)
    print(f"  Checkpoint : {path}")
    print(f"  Saved at epoch {epoch}")
    val_m = ckpt.get("val_metrics", {})
    if val_m:
        for k, v in val_m.items():
            if k != "loss" and not isinstance(v, dict):
                print(f"    val/{k:<12}: {v:.4f}")
    return model


def build_split_df(full_df, fold_data: dict, split: str, fold: int):
    """Return the sub-DataFrame for the requested split."""
    if split == "test":
        test_pids = fold_data.get("test")
        if test_pids is None:
            test_pids = fold_data["folds"].get(str(fold), {}).get("test", [])
        if not test_pids:
            raise ValueError(
                "No 'test' key in folds.json. Use --split val or check your folds file."
            )
        patient_set = set(map(str, test_pids))
        label = "test"
    elif split == "val":
        patient_set = set(map(str, fold_data["folds"][str(fold)]["val"]))
        label = f"val (fold {fold})"
    elif split == "train":
        patient_set = set(map(str, fold_data["folds"][str(fold)]["train"]))
        label = f"train (fold {fold})"
    else:
        raise ValueError(f"Unknown split '{split}'. Choose: test, val, train.")

    split_df = full_df[full_df["patient_id"].astype(str).isin(patient_set)].reset_index(drop=True)
    n_pats   = split_df["patient_id"].nunique()
    print(f"\n  Split: {label}  |  {n_pats} patients  |  {len(split_df)} images")
    return split_df


def get_display_transform():
    """
    Spatial-only transform (no ToTensor, no Normalize) that produces a PIL
    image of the same size the model sees.  Used for attention overlay display.
    """
    import torchvision.transforms as T
    aug_mode = getattr(cfg, "AUG_MODE", "v2")
    size     = cfg.IMG_SIZE + 32  # 256 for IMG_SIZE=224
    if aug_mode in ("v1", "v2s"):
        return T.Compose([T.Resize((size, size)), T.CenterCrop(cfg.IMG_SIZE)])
    else:  # v1a, v2
        return T.Compose([T.Resize(size), T.CenterCrop(cfg.IMG_SIZE)])


def encode_patient_images(model, img_paths: list, transform, device,
                           batch_size: int = 32):
    """
    Load images from disk and encode with the backbone.

    Returns:
        H           : [N, D] CPU tensor (N = successfully loaded images)
        valid_paths : list of paths that loaded successfully (same order as H)
    """
    tensors, valid_paths = [], []
    for p in img_paths:
        try:
            t = transform(Image.open(p).convert("RGB"))  # [3, H, W]
            tensors.append(t)
            valid_paths.append(str(p))
        except Exception as e:
            print(f"    WARNING: failed to load {p}: {e}")

    if not tensors:
        return None, []

    all_feats = []
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i : i + batch_size]).to(device)
        with torch.no_grad():
            feats = model.encode(batch)   # [B, D]
        all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0), valid_paths   # [N, D], [N]


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 1 — IMAGE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def compute_image_importance(H: torch.Tensor) -> torch.Tensor:
    """
    Max-pool importance: I_k = (1/D) Σ_d 𝟙[argmax_j H[j,d] == k]

    For each feature dimension d, find which image k had the highest activation.
    I_k is the fraction of dimensions 'won' by image k.  Values sum to 1.

    Args:
        H : [N, D] — N images × D-dim features, on CPU

    Returns:
        importance : [N] float tensor, sums to 1.0
    """
    N, D    = H.shape
    winners = H.argmax(dim=0)           # [D] — winner image per feature dim
    return torch.bincount(winners, minlength=N).float() / D


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 2 — DINOv2 ATTENTION MAPS
# ─────────────────────────────────────────────────────────────────────────────

def get_dinov2_attention(model, img_path: str, transform, device,
                         patch_size: int = 14) -> np.ndarray:
    """
    Extract DINOv2 last-layer CLS attention for one image.

    Tries two approaches in order:
      1. backbone.get_last_self_attention() if it exists *and* returns the
         expected [B, n_heads, N+1, N+1] attention tensor.
      2. Forward hook on the last block's qkv projection — robust fallback
         that works regardless of DINOv2 hub version or fused-attention mode.

    In both cases we average the CLS→patch attention row over all heads and
    bilinearly upsample to the model input resolution.

    Returns:
        attn_map : [H, W] float32 array normalised to [0, 1], or None on error
    """
    backbone = model.backbone

    try:
        tensor = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        _, _, H, W = tensor.shape
    except Exception as e:
        print(f"    WARNING: cannot load {img_path}: {e}")
        return None

    # ── Helper: reshape raw attention → upsampled map ────────────────────────
    def _to_map(attn_cls_np):
        """attn_cls_np: [N_patches] float32 → [H, W] normalised float32"""
        nh, nw = H // patch_size, W // patch_size
        arr = torch.from_numpy(attn_cls_np).float()
        out = F.interpolate(
            arr.reshape(1, 1, nh, nw),
            size=(H, W), mode="bilinear", align_corners=False,
        )[0, 0].numpy()
        a_min, a_max = out.min(), out.max()
        if a_max > a_min:
            out = (out - a_min) / (a_max - a_min)
        return out

    # ── Approach 1: get_last_self_attention ───────────────────────────────────
    if hasattr(backbone, "get_last_self_attention"):
        try:
            with torch.no_grad():
                attn = backbone.get_last_self_attention(tensor)
            # Validate: must be [B, n_heads, N+1, N+1] — square last two dims
            if attn.ndim == 4 and attn.shape[2] == attn.shape[3]:
                attn_cls = attn[0, :, 0, 1:].mean(0).cpu().numpy()  # [N_patches]
                return _to_map(attn_cls)
            # Wrong shape (e.g. hub version that returns block output) — fall through
        except Exception:
            pass  # Fall through to hook approach

    # ── Approach 2: forward hook on last block's qkv ─────────────────────────
    # This works regardless of DINOv2 version or whether fused attention is used,
    # because we intercept the qkv linear output before any attention kernel runs.
    blocks = getattr(backbone, "blocks", None)
    if blocks is None or len(blocks) == 0:
        return None
    last_attn = getattr(blocks[-1], "attn", None)
    if last_attn is None:
        return None
    qkv_layer = getattr(last_attn, "qkv", None)
    if qkv_layer is None:
        return None

    n_heads  = last_attn.num_heads
    captured = {}

    def _hook(module, inp, out):
        # out: [B, N, 3 * n_heads * head_dim]  (output of the qkv Linear)
        B, N, D3 = out.shape
        head_dim = D3 // (3 * n_heads)
        # Reshape and permute to get separate Q, K, V per head
        qkv = out.reshape(B, N, 3, n_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]          # [B, n_heads, N, head_dim] each
        # Apply per-head layer norms if present (DINOv2 registers-norm variant)
        if hasattr(last_attn, "q_norm"):
            q = last_attn.q_norm(q)
        if hasattr(last_attn, "k_norm"):
            k = last_attn.k_norm(k)
        scale = head_dim ** -0.5
        attn  = (q * scale) @ k.transpose(-2, -1)   # [B, n_heads, N, N]
        attn  = attn.softmax(dim=-1)
        captured["attn"] = attn.detach().cpu()

    handle = qkv_layer.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            backbone(tensor)          # full forward triggers hook on last block
    except Exception as e:
        print(f"    WARNING: backbone forward failed during attention hook: {e}")
        return None
    finally:
        handle.remove()

    if "attn" not in captured:
        return None

    attn = captured["attn"]                     # [1, n_heads, N+1, N+1]
    attn_cls = attn[0, :, 0, 1:].mean(0).numpy()   # [N_patches]
    return _to_map(attn_cls)


def save_attention_overlay(img_path: str, attn_map: np.ndarray, out_path: Path,
                            display_transform, importance_score: float,
                            gt_class: str = "",
                            alpha: float = 0.5):
    """
    Save a 3-panel figure: original image | attention heatmap | overlay.
    display_transform: spatial-only PIL transform (no ToTensor/Normalize).
    gt_class: ground-truth disease label shown in the figure title.
    """
    try:
        pil_disp = display_transform(Image.open(img_path).convert("RGB"))
        img_arr  = np.array(pil_disp)                      # [H, W, 3] uint8
        heatmap  = (cm.jet(attn_map)[:, :, :3] * 255).astype(np.uint8)
        overlay  = (alpha * heatmap + (1 - alpha) * img_arr).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, arr, title in zip(axes,
                                   [img_arr, cm.jet(attn_map), overlay],
                                   ["Original", "Attention", "Overlay"]):
            ax.imshow(arr)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        gt_str   = f"  |  GT: {_short_name(gt_class)}" if gt_class else ""
        sup_text = f"{Path(img_path).stem}{gt_str}  (importance={importance_score:.4f})"
        plt.suptitle(sup_text, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"    WARNING: save_attention_overlay failed for {img_path}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 3b — IMAGE vs TAB CONTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_contribution_split(model, H: torch.Tensor, tab_feats: torch.Tensor,
                                pred_class: int, device) -> dict:
    """
    Branch contribution split: image branch vs IF/marker branch.

    PRIMARY — Exact logit decomposition (no gradients needed):
        Since head = Dropout(eval=identity) + Linear, the predicted-class logit
        decomposes exactly as:
            ℓ_ĉ = W_ĉ[:D] · p_img  +  W_ĉ[D:] · p_tab  +  b_ĉ
        We report the two signed dot products and their absolute-value fractions.
        Signed values show whether a branch promoted (+) or suppressed (−) class ĉ.

    DEPRECATED — Gradient × feature sum (kept for comparison):
        Σ |∂ℓ_ĉ/∂p[i] · p[i]|  over each branch.  Has a dimensionality bias
        (image branch has D >> P dims) and is not exactly additive to the logit.

    Returns dict with both sets of values, or None if model has no tab_proj.
    """
    if not hasattr(model, "tab_proj") or tab_feats is None:
        return None

    H, tab_feats = H.to(device), tab_feats.to(device)

    with torch.no_grad():
        image_pool   = H.max(0)[0]               # [D]
        tab_proj_out = model.tab_proj(tab_feats)  # [P]

    # ── PRIMARY: exact logit decomposition ───────────────────────────────────
    # Find the classification Linear (last nn.Linear in head)
    linear = None
    for m in reversed(list(model.head.modules())):
        if isinstance(m, torch.nn.Linear):
            linear = m
            break

    logit_img = logit_tab = image_frac = tab_frac = None
    if linear is not None:
        with torch.no_grad():
            W   = linear.weight[pred_class]   # [D+P]
            D   = image_pool.shape[0]
            logit_img = float((W[:D] * image_pool).sum().item())    # signed
            logit_tab = float((W[D:] * tab_proj_out).sum().item())  # signed
        total_abs = abs(logit_img) + abs(logit_tab) + 1e-12
        image_frac = abs(logit_img) / total_abs
        tab_frac   = abs(logit_tab) / total_abs

    # ── DEPRECATED: gradient × feature sum ───────────────────────────────────
    image_pool_g = image_pool.detach().requires_grad_(True)
    tab_proj_g   = tab_proj_out.detach().requires_grad_(True)
    pooled = torch.cat([image_pool_g, tab_proj_g], dim=-1)
    logit_head = model.head(pooled)[pred_class]
    logit_head.backward()
    dep_c_img = (image_pool_g.grad * image_pool_g.detach()).abs().sum().item()
    dep_c_tab = (tab_proj_g.grad   * tab_proj_g.detach()).abs().sum().item()
    dep_total = dep_c_img + dep_c_tab + 1e-12

    return {
        # Primary: logit decomposition
        "image_frac":  image_frac,
        "tab_frac":    tab_frac,
        "logit_img":   logit_img,   # signed dot product W_img · p_img
        "logit_tab":   logit_tab,   # signed dot product W_tab · p_tab
        # Deprecated: gradient × feature sum
        "dep_image_frac": dep_c_img / dep_total,
        "dep_tab_frac":   dep_c_tab / dep_total,
        "dep_image_abs":  dep_c_img,
        "dep_tab_abs":    dep_c_tab,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 3a — MARKER SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_marker_sensitivity(model, H: torch.Tensor, tab_feats: torch.Tensor,
                                marker_names: list, n_classes: int,
                                device) -> dict:
    """
    For each marker i where tab_feats[i] != 0 (i.e. known):
        ΔP[i, c] = P(c | full markers) − P(c | m_i = 0, rest unchanged)

    Patients with unknown marker (0) are skipped for that marker, so the result
    reflects true informativeness without dilution by missing-data patients.

    Returns:
        {marker_name: {"delta_p": [float × n_classes] or None,
                       "original_value": float,
                       "status": "known" | "unknown"}}
    or None if model has no tab_proj.
    """
    if not hasattr(model, "tab_proj") or tab_feats is None:
        return None

    H, tab_feats = H.to(device), tab_feats.to(device)

    with torch.no_grad():
        logits_full, _ = model.aggregate(H, tab_feats)
        probs_full     = F.softmax(logits_full, dim=-1).cpu().numpy()  # [C]

    baseline_pred = int(probs_full.argmax())

    result = {}
    for i, name in enumerate(marker_names):
        orig = float(tab_feats[i].item())
        if orig == 0.0:
            result[name] = {"delta_p": None, "original_value": orig, "status": "unknown",
                            "baseline_pred": baseline_pred, "ablated_pred": None}
            continue

        ablated    = tab_feats.clone()
        ablated[i] = 0.0
        with torch.no_grad():
            logits_abl, _ = model.aggregate(H, ablated)
            probs_abl      = F.softmax(logits_abl, dim=-1).cpu().numpy()

        result[name] = {
            "delta_p":        (probs_full - probs_abl).tolist(),  # positive = knowing marker raised P(c)
            "original_value": orig,
            "status":         "known",
            "baseline_pred":  baseline_pred,
            "ablated_pred":   int(probs_abl.argmax()),
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE MARKER SENSITIVITY ACROSS PATIENTS
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_sensitivity(patient_results: list, marker_names: list,
                           class_names: list, out_dir: Path) -> dict:
    """
    Average signed and absolute ΔP across all patients where each marker
    is known, then save JSON + two visualisations.

    patient_results : list of (true_label, sensitivity_dict) one per patient
    """
    n_markers = len(marker_names)
    n_classes  = len(class_names)

    bucket     = defaultdict(lambda: defaultdict(list))  # all known: [m_name][c_idx] → [ΔP]
    bucket_pos = defaultdict(lambda: defaultdict(list))  # marker = +1
    bucket_neg = defaultdict(lambda: defaultdict(list))  # marker = −1
    # delta-accuracy: grouped by true class — [m_name][true_c] → [baseline_correct - ablated_correct]
    acc_bucket = defaultdict(lambda: defaultdict(list))
    n_known    = defaultdict(int)
    n_pos      = defaultdict(int)
    n_neg      = defaultdict(int)

    for true_label, sens in patient_results:
        if sens is None:
            continue
        for m_name, info in sens.items():
            if info["status"] != "known":
                continue
            n_known[m_name] += 1
            orig = info["original_value"]
            for c_idx, dp in enumerate(info["delta_p"]):
                bucket[m_name][c_idx].append(dp)
                if orig > 0:
                    bucket_pos[m_name][c_idx].append(dp)
                else:
                    bucket_neg[m_name][c_idx].append(dp)
            if orig > 0:
                n_pos[m_name] += 1
            else:
                n_neg[m_name] += 1
            # delta accuracy: was the prediction correct before/after ablation?
            bp = info.get("baseline_pred")
            ap = info.get("ablated_pred")
            if bp is not None and ap is not None:
                baseline_correct = int(bp == true_label)
                ablated_correct  = int(ap == true_label)
                acc_bucket[m_name][true_label].append(baseline_correct - ablated_correct)

    signed_mat      = np.full((n_markers, n_classes), np.nan)
    abs_mat         = np.full((n_markers, n_classes), np.nan)
    signed_pos_mat  = np.full((n_markers, n_classes), np.nan)
    signed_neg_mat  = np.full((n_markers, n_classes), np.nan)
    n_known_arr     = np.zeros(n_markers, dtype=int)
    n_pos_arr       = np.zeros(n_markers, dtype=int)
    n_neg_arr       = np.zeros(n_markers, dtype=int)

    for i, m in enumerate(marker_names):
        n_known_arr[i] = n_known.get(m, 0)
        n_pos_arr[i]   = n_pos.get(m, 0)
        n_neg_arr[i]   = n_neg.get(m, 0)
        for c in range(n_classes):
            vals = bucket[m][c]
            if vals:
                signed_mat[i, c] = float(np.mean(vals))
                abs_mat[i, c]    = float(np.mean(np.abs(vals)))
            vals_pos = bucket_pos[m][c]
            if vals_pos:
                signed_pos_mat[i, c] = float(np.mean(vals_pos))
            vals_neg = bucket_neg[m][c]
            if vals_neg:
                signed_neg_mat[i, c] = float(np.mean(vals_neg))

    # delta-accuracy matrix: rows = markers, cols = true classes
    delta_acc_mat = np.full((n_markers, n_classes), np.nan)
    for i, m in enumerate(marker_names):
        for c in range(n_classes):
            vals = acc_bucket[m][c]
            if vals:
                delta_acc_mat[i, c] = float(np.mean(vals))

    overall_imp = np.nanmean(abs_mat, axis=1)   # [n_markers]

    # ── JSON ──────────────────────────────────────────────────────────────────
    result_json = {}
    for i, m in enumerate(marker_names):
        result_json[m] = {
            "n_known":            int(n_known_arr[i]),
            "n_positive":         int(n_pos_arr[i]),
            "n_negative":         int(n_neg_arr[i]),
            "overall_importance": (float(overall_imp[i])
                                   if not np.isnan(overall_imp[i]) else None),
            "signed_delta_p": {class_names[c]: float(signed_mat[i, c])
                               for c in range(n_classes)
                               if not np.isnan(signed_mat[i, c])},
            "abs_delta_p":    {class_names[c]: float(abs_mat[i, c])
                               for c in range(n_classes)
                               if not np.isnan(abs_mat[i, c])},
            # Stratified by marker value — shows directional effects separately
            "signed_delta_p_pos": {class_names[c]: float(signed_pos_mat[i, c])
                                   for c in range(n_classes)
                                   if not np.isnan(signed_pos_mat[i, c])},
            "signed_delta_p_neg": {class_names[c]: float(signed_neg_mat[i, c])
                                   for c in range(n_classes)
                                   if not np.isnan(signed_neg_mat[i, c])},
            # Delta accuracy: grouped by true class (positive = marker improved accuracy)
            "delta_acc_by_true_class": {class_names[c]: float(delta_acc_mat[i, c])
                                        for c in range(n_classes)
                                        if not np.isnan(delta_acc_mat[i, c])},
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "marker_sensitivity.json"
    with open(json_path, "w") as f:
        json.dump(result_json, f, indent=2)
    print(f"  Saved → {json_path}")

    # ── Signed heatmap (n_markers × n_classes) ────────────────────────────────
    row_labels  = [f"{_clean_marker_name(m)}  (n={n_known_arr[i]})"
                   for i, m in enumerate(marker_names)]
    short_class = [_short_name(c) for c in class_names]
    _plot_sensitivity_heatmap(
        mat=signed_mat, row_labels=row_labels, col_labels=short_class,
        title="Marker Sensitivity: Mean signed ΔP\n(positive = knowing marker raises P(class))",
        out_path=out_dir / "marker_sensitivity_signed.png",
    )

    # ── Stratified signed heatmap (+1 vs −1 patients, interleaved rows) ────────
    strat_mat        = np.full((2 * n_markers, n_classes), np.nan)
    strat_row_labels = []
    for i, m in enumerate(marker_names):
        strat_mat[2 * i]     = signed_pos_mat[i]
        strat_mat[2 * i + 1] = signed_neg_mat[i]
        short = _clean_marker_name(m)
        strat_row_labels.append(f"{short} (+)  (n={n_pos_arr[i]})")
        strat_row_labels.append(f"{short} (−)  (n={n_neg_arr[i]})")
    _plot_sensitivity_heatmap(
        mat=strat_mat, row_labels=strat_row_labels, col_labels=short_class,
        title="Marker Sensitivity by Value: Mean signed ΔP\n(+ = marker-positive patients; − = marker-negative patients)",
        out_path=out_dir / "marker_sensitivity_signed_stratified.png",
    )

    # ── Delta-accuracy heatmap (grouped by true class) ───────────────────────
    # Row labels include the count of known patients (same as signed heatmap)
    _plot_sensitivity_heatmap(
        mat=delta_acc_mat, row_labels=row_labels, col_labels=short_class,
        title="Marker Sensitivity: Mean ΔAccuracy by true class\n"
              "(positive = knowing marker improved correct prediction rate)",
        out_path=out_dir / "marker_sensitivity_delta_acc.png",
        cbar_label="ΔAccuracy",
    )

    # ── Absolute bar chart (overall per-marker importance) ────────────────────
    bar_labels = [f"{_clean_marker_name(m)}\n(n={n_known_arr[i]})"
                  for i, m in enumerate(marker_names)]
    _plot_importance_bar(
        values=overall_imp, labels=bar_labels,
        title="Marker Overall Importance: Mean |ΔP| across classes",
        out_path=out_dir / "marker_sensitivity_abs.png",
    )

    return result_json


def _draw_sensitivity_heatmap(ax, fig, mat, row_labels, col_labels,
                               cbar_label="ΔP", fs=12):
    """Draw sensitivity heatmap onto an existing axes."""
    n_rows, n_cols = mat.shape
    vmax = float(np.nanmax(np.abs(mat))) if not np.all(np.isnan(mat)) else 0.1
    cmap_trunc = LinearSegmentedColormap.from_list(
        "RdBu_r_trunc", plt.cm.RdBu_r(np.linspace(0.15, 0.85, 256))
    )
    im = ax.imshow(mat, cmap=cmap_trunc, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=fs)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=fs)
    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:+.3f}", ha="center", va="center",
                        fontsize=fs - 2, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=fs - 2)
    cbar.set_label(cbar_label, fontsize=fs)
    ax.set_facecolor("white")


def _plot_sensitivity_heatmap(mat, row_labels, col_labels, title, out_path,
                              cbar_label="ΔP"):
    fs = 12
    n_rows, n_cols = mat.shape
    fig, ax = plt.subplots(figsize=(max(8, n_cols * 1.6), max(5, n_rows * 0.75)))
    _draw_sensitivity_heatmap(ax, fig, mat, row_labels, col_labels,
                              cbar_label=cbar_label, fs=fs)
    ax.set_title(title, fontsize=fs + 2, fontweight="bold")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def replot_sensitivity_from_json(json_path: Path, out_dir: Path,
                                 class_names: list):
    """Regenerate all marker-sensitivity plots from a saved marker_sensitivity.json.

    No model required — reads the pre-computed matrices from JSON and calls the
    same plot functions used by aggregate_sensitivity().
    """
    with open(json_path) as f:
        data = json.load(f)

    marker_names = list(data.keys())
    n_markers    = len(marker_names)
    n_classes    = len(class_names)
    short_class  = [_short_name(c) for c in class_names]

    signed_mat     = np.full((n_markers, n_classes), np.nan)
    abs_mat        = np.full((n_markers, n_classes), np.nan)
    signed_pos_mat = np.full((n_markers, n_classes), np.nan)
    signed_neg_mat = np.full((n_markers, n_classes), np.nan)
    delta_acc_mat  = np.full((n_markers, n_classes), np.nan)
    n_known_arr    = np.zeros(n_markers, dtype=int)
    n_pos_arr      = np.zeros(n_markers, dtype=int)
    n_neg_arr      = np.zeros(n_markers, dtype=int)

    for i, m in enumerate(marker_names):
        d = data[m]
        n_known_arr[i] = d.get("n_known", 0)
        n_pos_arr[i]   = d.get("n_positive", 0)
        n_neg_arr[i]   = d.get("n_negative", 0)
        for c_idx, cn in enumerate(class_names):
            v = d.get("signed_delta_p",     {}).get(cn)
            if v is not None: signed_mat[i, c_idx] = v
            v = d.get("abs_delta_p",        {}).get(cn)
            if v is not None: abs_mat[i, c_idx] = v
            v = d.get("signed_delta_p_pos", {}).get(cn)
            if v is not None: signed_pos_mat[i, c_idx] = v
            v = d.get("signed_delta_p_neg", {}).get(cn)
            if v is not None: signed_neg_mat[i, c_idx] = v
            v = d.get("delta_acc_by_true_class", {}).get(cn)
            if v is not None: delta_acc_mat[i, c_idx] = v

    overall_imp = np.nanmean(abs_mat, axis=1)

    row_labels = [f"{_clean_marker_name(m)}  (n={n_known_arr[i]})"
                  for i, m in enumerate(marker_names)]

    _plot_sensitivity_heatmap(
        mat=signed_mat, row_labels=row_labels, col_labels=short_class,
        title="Marker Sensitivity: Mean signed ΔP\n(positive = knowing marker raises P(class))",
        out_path=out_dir / "marker_sensitivity_signed.png",
    )

    strat_mat        = np.full((2 * n_markers, n_classes), np.nan)
    strat_row_labels = []
    for i, m in enumerate(marker_names):
        strat_mat[2 * i]     = signed_pos_mat[i]
        strat_mat[2 * i + 1] = signed_neg_mat[i]
        short = _clean_marker_name(m)
        strat_row_labels.append(f"{short} (+)  (n={n_pos_arr[i]})")
        strat_row_labels.append(f"{short} (−)  (n={n_neg_arr[i]})")
    _plot_sensitivity_heatmap(
        mat=strat_mat, row_labels=strat_row_labels, col_labels=short_class,
        title="Marker Sensitivity by Value: Mean signed ΔP\n(+ = marker-positive patients; − = marker-negative patients)",
        out_path=out_dir / "marker_sensitivity_signed_stratified.png",
    )

    _plot_sensitivity_heatmap(
        mat=delta_acc_mat, row_labels=row_labels, col_labels=short_class,
        title="Marker Sensitivity: Mean ΔAccuracy by true class\n"
              "(positive = knowing marker improved correct prediction rate)",
        out_path=out_dir / "marker_sensitivity_delta_acc.png",
        cbar_label="ΔAccuracy",
    )

    bar_labels = [f"{_clean_marker_name(m)}\n(n={n_known_arr[i]})"
                  for i, m in enumerate(marker_names)]
    _plot_importance_bar(
        values=overall_imp, labels=bar_labels,
        title="Marker Overall Importance: Mean |ΔP| across classes",
        out_path=out_dir / "marker_sensitivity_abs.png",
    )
    print("  Replot complete.")


def _plot_importance_bar(values, labels, title, out_path):
    order    = np.argsort(values)[::-1]
    sv, sl   = values[order], [labels[i] for i in order]
    fig, ax  = plt.subplots(figsize=(max(6, len(values) * 0.95), 4))
    bars = ax.bar(range(len(sv)), sv, color="steelblue")
    ax.set_xticks(range(len(sl)));  ax.set_xticklabels(sl, fontsize=9)
    ax.set_ylabel("Mean |ΔP|")
    ax.set_title(title, fontsize=10)
    for bar, v in zip(bars, sv):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def _draw_stacked_bar(ax, overall_d, per_class_d, class_names, fs=12):
    """Draw branch-contribution stacked bar onto an existing axes."""
    present  = [cn for cn in class_names if cn in per_class_d]
    labels   = ["Overall"] + [_short_name(cn) for cn in present]
    img_vals = ([overall_d["image_frac"]]
                + [per_class_d[cn]["image_frac"] for cn in present])
    tab_vals = ([overall_d["tab_frac"]]
                + [per_class_d[cn]["tab_frac"]   for cn in present])
    n_vals   = ([overall_d["n_patients"]]
                + [per_class_d[cn]["n_patients"] for cn in present])
    x = np.arange(len(labels))
    ax.bar(x, img_vals, label="Imaging",
           color="#6B9DC4", edgecolor="#3A6B8A", linewidth=0.6, alpha=0.85)
    ax.bar(x, tab_vals, bottom=img_vals, label="IF markers",
           color="#F4A261", edgecolor="#3A6B8A", linewidth=0.6, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{l}\n(n={n})" for l, n in zip(labels, n_vals)],
        fontsize=fs, rotation=30, ha="right",
    )
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_ylabel("Branch contribution (%)", fontsize=fs)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1),
              borderaxespad=0, fontsize=fs, framealpha=0.9)
    for xi, (iv, tv) in enumerate(zip(img_vals, tab_vals)):
        if iv > 0.06:
            ax.text(xi, iv / 2,      f"{iv:.0%}", ha="center", va="center",
                    fontsize=fs - 2, color="black")
        if tv > 0.06:
            ax.text(xi, iv + tv / 2, f"{tv:.0%}", ha="center", va="center",
                    fontsize=fs - 2, color="black")
    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, color="#D0D0D0")
    ax.set_axisbelow(True)
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)


def _stacked_bar(overall_d, per_class_d, class_names, title, out_path):
    fs      = 12
    present = [cn for cn in class_names if cn in per_class_d]
    fig, ax = plt.subplots(figsize=(max(6, (1 + len(present)) * 1.3), 5))
    _draw_stacked_bar(ax, overall_d, per_class_d, class_names, fs=fs)
    ax.set_title(title, fontsize=fs + 2, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def replot_contribution_from_json(json_path: Path, dep_json_path: Path,
                                   out_dir: Path, class_names: list):
    """Regenerate branch-contribution bar plots from saved contribution_split.json."""
    with open(json_path) as f:
        primary = json.load(f)

    overall  = primary["overall"]
    per_gt   = primary["per_gt_class"]
    per_pred = primary["per_predicted_class"]

    _stacked_bar(
        overall, per_gt, class_names,
        "Relative Branch Contribution to Predicted-Class Logit\n"
        "Grouped by Ground-Truth Diagnosis",
        out_dir / "contribution_split_gt.png",
    )
    _stacked_bar(
        overall, per_pred, class_names,
        "Relative Branch Contribution to Predicted-Class Logit\n"
        "Grouped by Predicted Diagnosis",
        out_dir / "contribution_split_pred.png",
    )

    if dep_json_path.exists():
        with open(dep_json_path) as f:
            deprecated = json.load(f)
        _stacked_bar(
            deprecated["overall"], deprecated["per_gt_class"], class_names,
            "Branch Contribution — gradient × feature sum [DEPRECATED]\n"
            "Grouped by Ground-Truth Diagnosis",
            out_dir / "contribution_split_deprecated.png",
        )
    print("  Replot complete.")


def save_explain_combined_figure(contrib_json_path: Path, sens_json_path: Path,
                                  out_path: Path, class_names: list, dpi: int = 150):
    """Two-row combined figure: (a) branch contribution by GT class,
    (b) marker sensitivity ΔAccuracy heatmap."""
    with open(contrib_json_path) as f:
        contrib = json.load(f)
    overall = contrib["overall"]
    per_gt  = contrib["per_gt_class"]

    with open(sens_json_path) as f:
        sens_data = json.load(f)
    marker_names = list(sens_data.keys())
    n_markers    = len(marker_names)
    n_classes    = len(class_names)
    short_class  = [_short_name(c) for c in class_names]

    delta_acc_mat = np.full((n_markers, n_classes), np.nan)
    n_known_arr   = np.zeros(n_markers, dtype=int)
    for i, m in enumerate(marker_names):
        n_known_arr[i] = sens_data[m].get("n_known", 0)
        for c_idx, cn in enumerate(class_names):
            v = sens_data[m].get("delta_acc_by_true_class", {}).get(cn)
            if v is not None:
                delta_acc_mat[i, c_idx] = v
    row_labels = [f"{_clean_marker_name(m)}  (n={n_known_arr[i]})"
                  for i, m in enumerate(marker_names)]

    fs = 9   # match the base font size used in the publication CM+ROC figure

    # ── Inch-based layout ─────────────────────────────────────────────────────
    # At fs=9 the heatmap y-labels ("IgA  (n=204)") are ≈ 0.67 in wide, so
    # LEFT_IN=1.40 keeps a ~0.7 in gap between the letter and the tick labels.
    # axes_w = 7.09 - 1.40 - 0.95 = 4.74 in
    FIG_W    = 7.09
    LEFT_IN  = 1.40   # axes left edge in inches from figure left
    RIGHT_IN = 0.95   # axes right edge in inches from figure right
    AX_W_IN  = FIG_W - LEFT_IN - RIGHT_IN   # ≈ 4.74 in axes width

    # Bar chart: target ratio ≈ 2.6 including the right-margin legend space:
    # (4.74 + 0.95) / BAR_H_IN = 2.6 → BAR_H_IN ≈ 2.2 in
    BAR_H_IN  = 2.20
    # Heatmap: rows at ~0.42 in each (reduced from 0.55 to keep overall fig compact)
    HEAT_H_IN = max(3.0, n_markers * 0.42)

    # Vertical margins (inches) — enough for titles, rotated tick labels, gap
    TOP_M  = 0.80   # space above bar chart for its title
    GAP    = 1.80   # space between bar bottom and heatmap top
                    # (bar x-ticks ≈ 0.5 in + heat title ≈ 0.5 in + breathing room)
    BOT_M  = 0.80   # space below heatmap for its x-tick labels

    fig_h = TOP_M + BAR_H_IN + GAP + HEAT_H_IN + BOT_M
    fig   = plt.figure(figsize=(FIG_W, fig_h))

    # Convert to figure fractions
    def frac(inches): return inches / fig_h
    LEFT_F  = LEFT_IN  / FIG_W
    RIGHT_F = 1.0 - RIGHT_IN / FIG_W
    AX_W_F  = AX_W_IN  / FIG_W

    bar_bot  = 1.0 - frac(TOP_M + BAR_H_IN)
    bar_top  = 1.0 - frac(TOP_M)
    heat_bot = frac(BOT_M)
    heat_top = frac(BOT_M + HEAT_H_IN)

    ax_bar  = fig.add_axes([LEFT_F, bar_bot,  AX_W_F, frac(BAR_H_IN)])
    ax_heat = fig.add_axes([LEFT_F, heat_bot, AX_W_F, frac(HEAT_H_IN)])

    _draw_stacked_bar(ax_bar, overall, per_gt, class_names, fs=fs)
    ax_bar.set_title(
        "Relative Branch Contribution to Predicted-Class Logit\n"
        "Grouped by Ground-Truth Diagnosis",
        fontsize=fs + 2, fontweight="bold",
    )

    _draw_sensitivity_heatmap(ax_heat, fig, delta_acc_mat, row_labels, short_class,
                              cbar_label="ΔAccuracy", fs=fs)
    ax_heat.set_title(
        "Marker Sensitivity: Mean ΔAccuracy by true class\n"
        "(positive = knowing marker improved correct prediction rate)",
        fontsize=fs + 2, fontweight="bold",
    )

    # Letter labels — positions computed directly from the known inch layout.
    # LETTER_X is deliberately small (near the figure left edge); pad_inches on
    # save adds a margin to the left of the tight bounding box so the letters
    # don't feel cramped against the edge of the saved image.
    LETTER_X = 0.008   # figure fraction ≈ 0.057 in from figure left
    fig.text(LETTER_X, (bar_bot  + bar_top)  / 2, "a",
             ha="center", va="center", fontsize=fs + 3, fontweight="bold")
    fig.text(LETTER_X, (heat_bot + heat_top) / 2, "b",
             ha="center", va="center", fontsize=fs + 3, fontweight="bold")

    fig.patch.set_facecolor("white")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE CONTRIBUTION SPLIT ACROSS PATIENTS
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_contribution_split(patient_splits: list, class_names: list,
                                  out_dir: Path) -> dict:
    """
    Aggregate image vs IF/marker branch contributions across patients.

    patient_splits : list of (true_label, pred_class, split_dict or None)

    PRIMARY method — logit decomposition:
      Reports three aggregations:
        overall          : all patients regardless of correct/incorrect prediction
        per_gt_class     : grouped by ground truth label — "for patients with
                           disease X, how did the model reason (whatever it predicted)?"
        per_pred_class   : grouped by predicted label — "when the model predicts X,
                           what evidence drove that decision?"
      Both gt and pred groupings mix correct and incorrect predictions, but
      per_pred_class is internally consistent: all decompositions in a group
      are for the same predicted-class logit row of W.

    DEPRECATED method — gradient × feature sum:
      Overall + per_gt_class only (kept for comparison).

    Outputs:
      contribution_split.json            primary results (all three aggregations)
      contribution_split_deprecated.json deprecated results
      contribution_split_gt.png          primary, grouped by ground truth class
      contribution_split_pred.png        primary, grouped by predicted class
      contribution_split_deprecated.png  deprecated plot (gt grouping)
    """
    valid = [(tl, pc, s) for tl, pc, s in patient_splits if s is not None]
    if not valid:
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Helper: aggregate a list of split dicts ───────────────────────────────
    def _agg(items):
        if not items:
            return None
        logit_imgs = [s["logit_img"] for s in items if s.get("logit_img") is not None]
        logit_tabs = [s["logit_tab"] for s in items if s.get("logit_tab") is not None]
        return {
            "image_frac":     float(np.mean([s["image_frac"] for s in items
                                             if s.get("image_frac") is not None])),
            "tab_frac":       float(np.mean([s["tab_frac"]   for s in items
                                             if s.get("tab_frac") is not None])),
            "mean_logit_img": float(np.mean(logit_imgs)) if logit_imgs else None,
            "mean_logit_tab": float(np.mean(logit_tabs)) if logit_tabs else None,
            "n_patients":     len(items),
        }

    # ── Three primary aggregations ────────────────────────────────────────────
    all_splits_list = [s for _, _, s in valid]
    overall         = _agg(all_splits_list)

    by_gt   = defaultdict(list)
    by_pred = defaultdict(list)
    for tl, pc, s in valid:
        by_gt[tl].append(s)
        by_pred[pc].append(s)

    per_gt   = {class_names[c]: _agg(by_gt[c])   for c in range(len(class_names)) if by_gt[c]}
    per_pred = {class_names[c]: _agg(by_pred[c])  for c in range(len(class_names)) if by_pred[c]}

    # ── Deprecated: gradient × feature sum ────────────────────────────────────
    def _agg_dep(items):
        if not items:
            return None
        return {
            "image_frac": float(np.mean([s["dep_image_frac"] for s in items])),
            "tab_frac":   float(np.mean([s["dep_tab_frac"]   for s in items])),
            "n_patients": len(items),
        }

    dep_overall = _agg_dep(all_splits_list)
    dep_by_gt   = defaultdict(list)
    for tl, _, s in valid:
        dep_by_gt[tl].append(s)
    dep_per_gt  = {class_names[c]: _agg_dep(dep_by_gt[c])
                   for c in range(len(class_names)) if dep_by_gt[c]}

    # ── Save JSONs ────────────────────────────────────────────────────────────
    primary = {
        "method": "logit_decomposition",
        "note": (
            "image_frac = |W_c·p_img| / (|W_c·p_img| + |W_c·p_tab|). "
            "Signed logit values show whether each branch promoted (+) or suppressed (−) "
            "the predicted class. Bias term excluded (constant across patients). "
            "per_gt_class groups by true diagnosis; per_pred_class groups by prediction."
        ),
        "overall":          overall,
        "per_gt_class":     per_gt,
        "per_predicted_class": per_pred,
    }
    deprecated = {
        "method": "gradient_x_feature_sum_DEPRECATED",
        "note": (
            "Σ|∂ℓ/∂p[i] · p[i]| per branch. Has dimensionality bias (D_img >> D_tab). "
            "Replaced by logit_decomposition in contribution_split.json."
        ),
        "overall":      dep_overall,
        "per_gt_class": dep_per_gt,
    }

    p_json = out_dir / "contribution_split.json"
    with open(p_json, "w") as f:
        json.dump(primary, f, indent=2)
    print(f"  Saved → {p_json}")

    p_dep_json = out_dir / "contribution_split_deprecated.json"
    with open(p_dep_json, "w") as f:
        json.dump(deprecated, f, indent=2)
    print(f"  Saved → {p_dep_json}")

    _stacked_bar(
        overall, per_gt, class_names,
        "Relative Branch Contribution to Predicted-Class Logit\n"
        "Grouped by Ground-Truth Diagnosis",
        out_dir / "contribution_split_gt.png",
    )
    _stacked_bar(
        overall, per_pred, class_names,
        "Relative Branch Contribution to Predicted-Class Logit\n"
        "Grouped by Predicted Diagnosis",
        out_dir / "contribution_split_pred.png",
    )
    _stacked_bar(
        dep_overall, dep_per_gt, class_names,
        "Branch Contribution — gradient × feature sum [DEPRECATED]\n"
        "Grouped by Ground-Truth Diagnosis",
        out_dir / "contribution_split_deprecated.png",
    )

    return primary


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Explainability analysis for trained EM3 MIL models."
    )
    parser.add_argument("--run-dir",  required=True,
                        help="Run directory containing checkpoint + config .yml")
    parser.add_argument("--folds",    default=None,
                        help="Path to folds.json (default: cfg.FOLDS_PATH)")
    parser.add_argument("--split",    default="test", choices=["test", "val", "train"],
                        help="Data split to explain (default: test)")
    parser.add_argument("--fold",     type=int, default=None,
                        help="Fold index for val/train splits (default: cfg.FOLD)")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint filename inside --run-dir (default: auto-detect best)")
    parser.add_argument("--config",   default=None,
                        help="Config .yml override (default: auto-detect inside --run-dir)")
    parser.add_argument("--out",      default=None,
                        help="Output directory (default: <run-dir>/explain_<split>/)")
    parser.add_argument("--top-k",    type=int, default=5,
                        help="Top-K images per patient for attention maps (default: 5)")
    parser.add_argument("--levels",   nargs="+", default=["1", "2", "3a", "3b"],
                        choices=["1", "2", "3a", "3b"],
                        help="Analysis levels to run (default: all)")
    parser.add_argument("--n-patients", type=int, default=None,
                        help="Process only first N patients (debug mode)")
    parser.add_argument("--encode-batch-size", type=int, default=32,
                        help="Batch size for backbone image encoding (default: 32)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore existing feature cache and re-encode all images")
    parser.add_argument("--replot", action="store_true",
                        help="Regenerate sensitivity and contribution plots from existing "
                             "JSONs without re-running the model")
    args   = parser.parse_args()

    run_dir = Path(args.run_dir)
    fold    = args.fold if args.fold is not None else cfg.FOLD
    levels  = set(args.levels)
    device  = torch.device(cfg.DEVICE)
    out_dir = Path(args.out) if args.out else run_dir / f"explain_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.replot:
        # Derive class names from the JSON rather than cfg.NUM_CLASSES, which
        # may come from a YAML with a different class count than what was used
        # when the JSON was produced.
        sens_json    = out_dir / "marker_sensitivity.json"
        contrib_json = out_dir / "contribution_split.json"
        class_names  = None
        if contrib_json.exists():
            with open(contrib_json) as _f:
                _d = json.load(_f)
            # per_gt_class keys are in insertion order (Python 3.7+)
            class_names = list(_d.get("per_gt_class", {}).keys()) or list(_d.get("per_predicted_class", {}).keys())
        if not class_names and sens_json.exists():
            with open(sens_json) as _f:
                _d = json.load(_f)
            # use signed_delta_p keys from the first marker that has them
            for _m in _d.values():
                class_names = list(_m.get("signed_delta_p", {}).keys())
                if class_names:
                    break
        if not class_names:
            class_names = [IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))]
        print(f"  Class names ({len(class_names)}): {class_names}")
        if sens_json.exists():
            print(f"  Replotting sensitivity from {sens_json}")
            replot_sensitivity_from_json(sens_json, out_dir, class_names)
        else:
            print(f"  Skipping sensitivity replot (no {sens_json})")
        if contrib_json.exists():
            print(f"  Replotting contribution from {contrib_json}")
            replot_contribution_from_json(
                contrib_json,
                out_dir / "contribution_split_deprecated.json",
                out_dir, class_names,
            )
        else:
            print(f"  Skipping contribution replot (no {contrib_json})")
        if sens_json.exists() and contrib_json.exists():
            print(f"  Generating combined explain figure")
            save_explain_combined_figure(
                contrib_json, sens_json,
                out_dir / "explain_combined_figure.png",
                class_names,
            )
        return

    cache_dir = out_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  EM3 Explainability")
    print(f"  Run dir  : {run_dir}")
    print(f"  Split    : {args.split}  |  Fold: {fold}")
    print(f"  Levels   : {sorted(levels)}")
    print(f"  Out dir  : {out_dir}")
    print(f"  Device   : {device}")
    print(f"  Feature cache: {cache_dir}{'  (--no-cache: will re-encode)' if args.no_cache else ''}")
    print(f"{'='*65}\n")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    df = build_dataframe_v2(cfg.PROCESSED_JSON_PATH)

    folds_path = args.folds or getattr(cfg, "FOLDS_PATH", None)
    if folds_path is None:
        raise ValueError("--folds not given and cfg.FOLDS_PATH is not set.")
    with open(folds_path) as f:
        fold_data = json.load(f)

    split_df = build_split_df(df, fold_data, args.split, fold)

    # ── 2. Tabular features ───────────────────────────────────────────────────
    tab_features, marker_names, tab_dim = {}, [], 0
    has_tab = getattr(cfg, "TAB_FUSION", False)
    if has_tab:
        tab_features, marker_names, tab_dim = load_tabular_features(
            csv_path    = cfg.TAB_FEATURES_CSV,
            id_col      = cfg.TAB_ID_COL,
            marker_cols = cfg.TAB_MARKER_COLS,
        )
        print(f"  Tabular fusion: {tab_dim} markers")
    else:
        print("  Tabular fusion: disabled — levels 3a and 3b will be skipped")

    # ── 3. Model ──────────────────────────────────────────────────────────────
    # ── Resolve checkpoint path ───────────────────────────────────────────────
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        candidates = [p for p in sorted(run_dir.glob("*.pt")) if "best" in p.name.lower()]
        if len(candidates) == 1:
            ckpt_path = candidates[0]
        elif not candidates:
            raise FileNotFoundError(f"No 'best' .pt checkpoint found in {run_dir}.")
        else:
            raise ValueError(
                f"Multiple 'best' checkpoints found: {[p.name for p in candidates]}. "
                f"Specify with --checkpoint."
            )
    else:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_absolute():
            ckpt_path = run_dir / ckpt_path

    # ── Sync NUM_CLASSES from checkpoint (mirrors test.py) ───────────────────
    # The checkpoint stores cfg.__dict__ at save time — use it as ground truth
    # for the architecture, guarding against early-config not syncing NUM_CLASSES.
    _raw = torch.load(ckpt_path, map_location="cpu")
    _ckpt_nc = _raw.get("cfg", {}).get("NUM_CLASSES", None)
    if _ckpt_nc is not None and _ckpt_nc != cfg.NUM_CLASSES:
        print(f"  WARNING: checkpoint NUM_CLASSES={_ckpt_nc} != cfg.NUM_CLASSES={cfg.NUM_CLASSES}. "
              f"Using checkpoint value.")
        cfg.NUM_CLASSES = _ckpt_nc
    del _raw

    model = build_model(tab_dim=tab_dim).to(device)
    model = load_checkpoint(ckpt_path, model, device)
    model.eval()

    # ── Feature cache validation ──────────────────────────────────────────────
    # Cache is keyed by checkpoint path + mtime so stale entries are detected
    # automatically when the checkpoint changes between runs.
    _ckpt_mtime   = os.path.getmtime(ckpt_path)
    _cache_meta   = cache_dir / "cache_meta.json"
    _cache_valid  = False
    if not args.no_cache:
        if _cache_meta.exists():
            with open(_cache_meta) as _f:
                _meta = json.load(_f)
            if (_meta.get("checkpoint") == str(ckpt_path.resolve())
                    and abs(_meta.get("mtime", 0) - _ckpt_mtime) < 1.0):
                _cache_valid = True
            else:
                print("  Feature cache: checkpoint changed — cache invalidated, will re-encode")
        else:
            print("  Feature cache: no existing cache — will encode and save")
    # Write/overwrite meta so the cache reflects this run's checkpoint
    with open(_cache_meta, "w") as _f:
        json.dump({"checkpoint": str(ckpt_path.resolve()), "mtime": _ckpt_mtime}, _f, indent=2)

    # ── Attention availability diagnostic (Level 2) ───────────────────────────
    _bb = getattr(model, "backbone", None)
    if _bb is not None:
        _has_glsa   = hasattr(_bb, "get_last_self_attention")
        _has_blocks = hasattr(_bb, "blocks") and len(getattr(_bb, "blocks", [])) > 0
        _has_qkv    = (
            _has_blocks
            and hasattr(getattr(_bb.blocks[-1], "attn", None), "qkv")
        )
        print(f"  Level 2 attention methods available:")
        print(f"    get_last_self_attention : {'yes' if _has_glsa else 'no (will use hook fallback)'}")
        print(f"    blocks + qkv hook       : {'yes' if _has_qkv else 'no'}")
        if not _has_glsa and not _has_qkv:
            print("    WARNING: neither approach available — Level 2 will produce no maps")

    # ── 4. Transforms ─────────────────────────────────────────────────────────
    eval_transform    = get_transforms(train=False)   # full: resize/crop + ToTensor + Normalize
    display_transform = get_display_transform()        # spatial only: resize/crop, PIL output

    is_dinov2 = getattr(cfg, "BACKBONE", "resnet50") == "dinov2_vits14"
    if "2" in levels and not is_dinov2:
        print("  Level 2 skipped: backbone is not dinov2_vits14")
        levels.discard("2")

    # ── 5. Patient loop ───────────────────────────────────────────────────────
    patients    = split_df["patient_id"].unique().tolist()
    class_names = [IDX_TO_CLASS[i] for i in range(cfg.NUM_CLASSES)]
    max_imgs    = getattr(cfg, "MAX_IMGS_PER_PATIENT", None)

    if args.n_patients:
        patients = patients[:args.n_patients]
        print(f"  [debug] limited to {len(patients)} patients")

    all_importance   = []   # Level 1 accumulator → JSON
    all_splits       = []   # Level 3b accumulator → aggregation
    all_sensitivity  = []   # Level 3a accumulator → aggregation

    if "2" in levels:
        (out_dir / "attention_maps").mkdir(parents=True, exist_ok=True)

    for pat_idx, pid in enumerate(tqdm(patients, desc="Patients")):
        pid_str    = str(pid)
        pid_rows   = split_df[split_df["patient_id"] == pid].reset_index(drop=True)
        true_label = int(pid_rows["label"].iloc[0])
        img_paths  = pid_rows["full_path"].tolist()

        # Tabular features for this patient
        tab_feats = None
        if has_tab:
            if pid_str in tab_features:
                tab_feats = torch.from_numpy(tab_features[pid_str]).to(device)
            else:
                print(f"    WARNING: {pid_str} not in tab_features — using zeros")
                tab_feats = torch.zeros(tab_dim, device=device)

        # Deterministic image cap (epoch=0, same convention as val PatientMILDataset)
        if max_imgs and len(img_paths) > max_imgs:
            pseed     = cfg.SEED + pat_idx           # stable across explain.py runs
            img_paths = np.random.default_rng(pseed).choice(
                img_paths, max_imgs, replace=False
            ).tolist()

        # Encode all images (load from cache if available and valid)
        cache_file = cache_dir / f"{pid_str}.npz"
        H = valid_paths = None
        if _cache_valid and not args.no_cache and cache_file.exists():
            try:
                _cached      = np.load(cache_file, allow_pickle=True)
                H            = torch.from_numpy(_cached["H"])          # [N, D] float32
                valid_paths  = list(_cached["valid_paths"])
            except Exception as _e:
                print(f"    WARNING: cache load failed for {pid_str} ({_e}) — re-encoding")
                H = valid_paths = None

        if H is None:
            H, valid_paths = encode_patient_images(
                model, img_paths, eval_transform, device, args.encode_batch_size
            )
            if H is not None and H.shape[0] > 0:
                np.savez_compressed(
                    cache_file,
                    H           = H.cpu().numpy().astype(np.float32),
                    valid_paths = np.array(valid_paths),
                )

        if H is None or H.shape[0] == 0:
            print(f"    WARNING: no valid images for {pid_str} — skipping")
            continue

        H_dev = H.to(device)

        # Baseline prediction
        with torch.no_grad():
            logits, _  = model.aggregate(H_dev, tab_feats)
            probs      = F.softmax(logits, dim=-1)
            pred_class = int(probs.argmax().item())
            pred_prob  = float(probs[pred_class].item())

        # ── Level 1: image importance ─────────────────────────────────────────
        if "1" in levels:
            importance = compute_image_importance(H)    # [N], CPU
            sorted_idx = importance.argsort(descending=True)
            all_importance.append({
                "patient_id":  pid_str,
                "true_label":  true_label,
                "true_class":  class_names[true_label],
                "pred_class":  pred_class,
                "pred_name":   class_names[pred_class],
                "pred_prob":   pred_prob,
                "images": [
                    {
                        "rank":       int(r) + 1,
                        "path":       valid_paths[int(sorted_idx[r])],
                        "importance": float(importance[sorted_idx[r]]),
                    }
                    for r in range(len(sorted_idx))
                ],
            })
        else:
            # Still need importance for Level 2 top-K selection
            importance = compute_image_importance(H) if "2" in levels else None
            sorted_idx = importance.argsort(descending=True) if importance is not None else None

        # ── Level 2: attention maps on top-K images ───────────────────────────
        if "2" in levels and importance is not None:
            topk = min(args.top_k, len(valid_paths))
            for rank in range(topk):
                img_idx   = int(sorted_idx[rank])
                img_path  = valid_paths[img_idx]
                imp_score = float(importance[img_idx])

                attn_map = get_dinov2_attention(
                    model, img_path, eval_transform, device, patch_size=14
                )
                if attn_map is not None:
                    out_png = out_dir / "attention_maps" / f"patient_{pid_str}_top{rank+1}.png"
                    save_attention_overlay(img_path, attn_map, out_png,
                                           display_transform, imp_score,
                                           gt_class=class_names[true_label],
                                           alpha=0.5)

        # ── Level 3b: image vs tab contribution ──────────────────────────────
        if "3b" in levels and has_tab:
            split_info = compute_contribution_split(model, H_dev, tab_feats, pred_class, device)
            all_splits.append((true_label, pred_class, split_info))

        # ── Level 3a: marker sensitivity ─────────────────────────────────────
        if "3a" in levels and has_tab and marker_names:
            sensitivity = compute_marker_sensitivity(
                model, H_dev, tab_feats, marker_names, cfg.NUM_CLASSES, device
            )
            all_sensitivity.append((true_label, sensitivity))

    # ── 6. Save aggregated results ────────────────────────────────────────────
    print()

    if "1" in levels and all_importance:
        p = out_dir / "patient_importance.json"
        with open(p, "w") as f:
            json.dump(all_importance, f, indent=2)
        print(f"  Saved → {p}")

    if "3b" in levels and all_splits:
        result_3b = aggregate_contribution_split(all_splits, class_names, out_dir)
        ov = result_3b.get("overall", {})
        print(f"  Image vs IF/Marker (overall, logit decomp) : "
              f"image {ov.get('image_frac', 0):.1%}  "
              f"IF/marker {ov.get('tab_frac', 0):.1%}  "
              f"(n={ov.get('n_patients', 0)})")

    if "3a" in levels and all_sensitivity:
        aggregate_sensitivity(all_sensitivity, marker_names, class_names, out_dir)

    print(f"\n{'='*65}")
    print(f"  Done.  Results → {out_dir}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
