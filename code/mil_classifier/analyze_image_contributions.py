"""
analyze_image_contributions.py
─────────────────────────────────────────────────────────────────────────────
Loads a trained EM3 patient-level MIL model and runs two levels of inference:

  1. Patient-level: feed all images as a bag → one prediction per patient
  2. Image-level:   feed each image alone as a bag of size 1 → per-image vote

For each patient, computes:
  - Whether MIL prediction was correct
  - Per-image predictions and whether each image voted correctly
  - Fraction of images that voted for the correct class
  - Majority vote across images and whether it matches MIL
  - Mean / median pixel size of correctly-voting vs incorrectly-voting images
  - Attention weight per image (if MIL_MODE=attention)

Outputs:
  <out_dir>/patient_predictions.csv     — one row per patient
  <out_dir>/image_predictions.csv       — one row per image
  <out_dir>/patient_summary.csv         — aggregated stats per patient
  <out_dir>/plots/                      — per-patient visualisations

Usage:
    python analyze_image_contributions.py \\
        --checkpoint  checkpoints/dino_finetuned_patient_attention_0/patient_best_fold0.pt \\
        --json        /path/to/images_dict_final.json \\
        --out         analysis/dino_finetuned_fold0 \\
        [--split      val]          # val (default) | train | all
        [--device     cuda]
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Add EM3 directory to path so we can import config, model, dataset, etc.
# The script assumes it lives inside (or next to) the EM3 folder.
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from config  import cfg, _parse_config_arg
from dataset import build_dataframe_v2, patient_level_split, CLASS_TO_IDX, IDX_TO_CLASS, get_transforms
from model   import build_model
import torchvision.transforms as T


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────



def load_image(path: str, transform) -> torch.Tensor:
    """Load a single TIF as [1, 3, H, W]."""
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)  # [1, 3, H, W]


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint, restoring cfg overrides stored in it."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Restore config fields from checkpoint snapshot so model is built correctly
    stored_cfg = ckpt.get("cfg", {})
    for key in ["BACKBONE", "MIL_MODE", "NUM_CLASSES", "IMG_SIZE",
                "FREEZE_BACKBONE", "VIT_CHECKPOINT", "PRETRAINED"]:
        if key in stored_cfg:
            setattr(cfg, key, stored_cfg[key])

    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    fold       = stored_cfg.get("FOLD", cfg.FOLD)
    mil_mode   = stored_cfg.get("MIL_MODE", cfg.MIL_MODE)
    backbone   = stored_cfg.get("BACKBONE", cfg.BACKBONE)
    print(f"\n  Model loaded: {backbone}, MIL={mil_mode}, fold={fold}")
    print(f"  Classes: {cfg.NUM_CLASSES}")
    return model, fold


@torch.no_grad()
def predict_patient(model, img_tensors: list, device: torch.device):
    """
    Full patient-level MIL prediction.
    img_tensors: list of [1, 3, H, W] tensors (one per image)
    Returns: probs [C], attn [N] or None
    """
    imgs = torch.cat(img_tensors, dim=0).to(device)  # [N, 3, H, W]
    logits, attn = model.forward_patient(imgs)        # [C], [N] or None
    probs = F.softmax(logits.float(), dim=0).cpu().numpy()
    attn_np = attn.cpu().numpy() if attn is not None else None
    return probs, attn_np


@torch.no_grad()
def predict_image(model, img_tensor: torch.Tensor, device: torch.device):
    """
    Single-image prediction (bag of size 1).
    img_tensor: [1, 3, H, W]
    Returns: probs [C]
    """
    imgs = img_tensor.to(device)                    # [1, 3, H, W]
    logits, _ = model.forward_patient(imgs)         # treat single image as a bag
    probs = F.softmax(logits.float(), dim=0).cpu().numpy()
    return probs


# ─────────────────────────────────────────────────────────────────────────────
# METADATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_metadata(json_path: str) -> dict:
    """Returns {full_path: {pixelsize_nm, magnification, ...}}"""
    with open(json_path) as f:
        records = json.load(f)
    meta = {}
    for r in records:
        fp = r.get("fullpath", "")
        px = r.get("pixelsizex")
        meta[fp] = {
            "pixelsize_nm": float(px) * 1e9 if px is not None else None,
            "magnification": r.get("magnification"),
        }
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

COLORS = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB"]


def plot_patient(patient_id: str, true_class: str, true_label: int,
                 patient_probs: np.ndarray, patient_pred: int,
                 image_rows: list, class_names: list,
                 attn: np.ndarray, out_path: Path) -> None:
    """
    One figure per patient:
      Top:    patient-level probability bar chart
      Middle: per-image predicted class heatmap (images × classes)
      Bottom: attention weights (if available)
    """
    n_images = len(image_rows)
    n_classes = len(class_names)

    has_attn = attn is not None and len(attn) == n_images
    n_rows = 3 if has_attn else 2
    height = 3 + n_images * 0.25 + (1.5 if has_attn else 0)
    height = max(6, min(height, 20))

    fig = plt.figure(figsize=(14, height))
    gs  = gridspec.GridSpec(n_rows, 1, hspace=0.45,
                             height_ratios=[2, max(2, n_images * 0.25)] +
                                           ([1.5] if has_attn else []))

    short_names = [c[:20] for c in class_names]
    colors = [COLORS[i % len(COLORS)] for i in range(n_classes)]

    # ── Panel 1: patient-level probabilities ─────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    bars = ax0.bar(range(n_classes), patient_probs, color=colors, alpha=0.8,
                   edgecolor="white")
    ax0.bar(true_label, patient_probs[true_label], color=colors[true_label],
            edgecolor="black", linewidth=2, alpha=0.9)
    ax0.axvline(patient_pred - 0.4, color="red", linewidth=1.5,
                linestyle="--", label="Predicted")
    ax0.set_xticks(range(n_classes))
    ax0.set_xticklabels(short_names, rotation=20, ha="right", fontsize=8)
    ax0.set_ylabel("Probability")
    correct_str = "✓" if patient_pred == true_label else "✗"
    ax0.set_title(
        f"Patient {patient_id}  |  True: {true_class[:30]}  |  "
        f"Pred: {class_names[patient_pred][:30]}  {correct_str}",
        fontsize=9, pad=4
    )
    ax0.set_ylim(0, 1.05)
    ax0.spines[["top", "right"]].set_visible(False)

    # ── Panel 2: per-image prediction heatmap ────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    prob_matrix = np.array([r["image_probs"] for r in image_rows])  # [N, C]
    im = ax1.imshow(prob_matrix, aspect="auto", cmap="Blues",
                    vmin=0, vmax=1, interpolation="nearest")

    # Mark true class column with a box
    for spine in ["top", "bottom", "left", "right"]:
        ax1.spines[spine].set_visible(False)
    ax1.axvline(true_label - 0.5, color="green", linewidth=2)
    ax1.axvline(true_label + 0.5, color="green", linewidth=2)

    # Colour image labels by correctness
    ytick_labels = []
    ytick_colors = []
    for r in image_rows:
        name = Path(r["full_path"]).stem[:25]
        px   = r.get("pixelsize_nm")
        px_s = f" {px:.1f}nm" if px else ""
        ytick_labels.append(f"{name}{px_s}")
        ytick_colors.append("green" if r["image_correct"] else "red")

    ax1.set_yticks(range(n_images))
    ax1.set_yticklabels(ytick_labels, fontsize=6)
    for tick, col in zip(ax1.get_yticklabels(), ytick_colors):
        tick.set_color(col)

    ax1.set_xticks(range(n_classes))
    ax1.set_xticklabels(short_names, rotation=20, ha="right", fontsize=7)
    ax1.set_xlabel("Predicted class probability")
    ax1.set_title(f"Per-image predictions ({n_images} images)  "
                  f"[green=correct vote, red=wrong vote]", fontsize=8)
    plt.colorbar(im, ax=ax1, fraction=0.02, pad=0.01)

    # ── Panel 3: attention weights ────────────────────────────────────────────
    if has_attn:
        ax2 = fig.add_subplot(gs[2])
        ax2.barh(range(n_images), attn[::-1], color=[
            "green" if r["image_correct"] else "red"
            for r in image_rows[::-1]
        ], alpha=0.75, edgecolor="white")
        ax2.set_yticks(range(n_images))
        ax2.set_yticklabels(
            [Path(r["full_path"]).stem[:20] for r in image_rows[::-1]],
            fontsize=5)
        ax2.set_xlabel("Attention weight")
        ax2.set_title("Attention weights  [green=correct vote, red=wrong vote]",
                      fontsize=8)
        ax2.spines[["top", "right"]].set_visible(False)

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--json",       required=True,
                        help="Path to images_dict_final.json")
    parser.add_argument("--out",        default="analysis",
                        help="Output directory")
    parser.add_argument("--split",      default="val",
                        choices=["val", "train", "all"],
                        help="Which split to evaluate (default: val)")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--config",     default=None,
                        help="Optional YAML config override")
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    if args.config:
        import yaml
        with open(args.config) as f:
            overrides = yaml.safe_load(f)
        for k, v in overrides.items():
            setattr(cfg, k, v)

    out_dir = Path(args.out)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    transform = get_transforms(train=False)

    # ── Load model ────────────────────────────────────────────────────────────
    model, fold = load_model(args.checkpoint, device)
    class_names = cfg.CLASSES

    # ── Load data ─────────────────────────────────────────────────────────────
    # Resolve JSON path: cfg.PROCESSED_JSON_PATH takes priority over --json arg
    json_path = str(getattr(cfg, "PROCESSED_JSON_PATH", None) or args.json)
    print(f"\nLoading data from {json_path} ...")
    df = build_dataframe_v2(Path(json_path))

    # Add metadata (pixel size, magnification) from the same JSON
    print("Loading image metadata ...")
    meta = load_metadata(json_path)
    df["pixelsize_nm"]  = df["full_path"].map(lambda p: meta.get(p, {}).get("pixelsize_nm"))
    df["magnification"] = df["full_path"].map(lambda p: meta.get(p, {}).get("magnification"))

    # ── Split ─────────────────────────────────────────────────────────────────
    train_df, val_df = patient_level_split(
        df, n_folds=cfg.N_FOLDS, fold=fold, seed=cfg.SEED
    )

    if args.split == "val":
        eval_df = val_df.copy()
        eval_df["split"] = "val"
    elif args.split == "train":
        eval_df = train_df.copy()
        eval_df["split"] = "train"
    else:
        train_df["split"] = "train"
        val_df["split"]   = "val"
        eval_df = pd.concat([train_df, val_df], ignore_index=True)

    patients = eval_df["patient_id"].unique().tolist()
    print(f"\nEvaluating {len(patients)} patients ({args.split} split) ...")

    # ── Inference ─────────────────────────────────────────────────────────────
    patient_rows = []
    image_rows   = []

    for pid in tqdm(patients, desc="  Inference", unit="patient"):
        prows = eval_df[eval_df["patient_id"] == pid]
        true_label = int(prows["label"].iloc[0])
        true_class = IDX_TO_CLASS[true_label]
        split      = prows["split"].iloc[0]

        # Apply same cap as PatientMILDataset.__getitem__ during training validation
        # — reset index first, then sample with fixed seed, to get identical subset
        max_imgs = getattr(cfg, "MAX_IMGS_PER_PATIENT", None)
        prows = prows.reset_index(drop=True)
        if max_imgs and len(prows) > max_imgs:
            prows = prows.sample(max_imgs, random_state=cfg.SEED)

        # Load all images
        img_tensors = []
        valid_rows  = []
        for _, row in prows.iterrows():
            try:
                t = load_image(row["full_path"], transform)
                img_tensors.append(t)
                valid_rows.append(row)
            except Exception:
                pass

        if not img_tensors:
            continue

        # Patient-level prediction
        patient_probs, attn = predict_patient(model, img_tensors, device)
        patient_pred  = int(np.argmax(patient_probs))
        patient_correct = (patient_pred == true_label)

        # Per-image predictions
        img_pred_rows = []
        for i, (t, row) in enumerate(zip(img_tensors, valid_rows)):
            img_probs = predict_image(model, t, device)
            img_pred  = int(np.argmax(img_probs))
            img_correct = (img_pred == true_label)
            img_attn = float(attn[i]) if attn is not None else None

            img_pred_rows.append({
                "patient_id":    pid,
                "split":         split,
                "true_label":    true_label,
                "true_class":    true_class,
                "full_path":     row["full_path"],
                "pixelsize_nm":  row.get("pixelsize_nm"),
                "magnification": row.get("magnification"),
                "image_pred":    img_pred,
                "image_class":   IDX_TO_CLASS[img_pred],
                "image_correct": img_correct,
                "attention_weight": img_attn,
                "image_probs":   img_probs,  # kept for plotting, dropped before CSV save
                **{f"prob_{IDX_TO_CLASS[c][:15]}": float(img_probs[c])
                   for c in range(len(class_names))},
            })
            image_rows.append({k: v for k, v in img_pred_rows[-1].items()
                                if k != "image_probs"})

        # Majority vote
        img_preds_arr = np.array([r["image_pred"] for r in img_pred_rows])
        counts = np.bincount(img_preds_arr, minlength=len(class_names))
        majority_pred = int(np.argmax(counts))
        majority_correct = (majority_pred == true_label)

        # Fraction voting correctly
        n_correct_votes = int(np.sum(img_preds_arr == true_label))
        frac_correct    = n_correct_votes / len(img_preds_arr)

        # Pixel size stats for correct vs wrong voting images
        px_correct = [r["pixelsize_nm"] for r in img_pred_rows
                      if r["image_correct"] and r["pixelsize_nm"] is not None]
        px_wrong   = [r["pixelsize_nm"] for r in img_pred_rows
                      if not r["image_correct"] and r["pixelsize_nm"] is not None]

        patient_rows.append({
            "patient_id":           pid,
            "split":                split,
            "true_label":           true_label,
            "true_class":           true_class,
            "patient_pred":         patient_pred,
            "patient_class":        IDX_TO_CLASS[patient_pred],
            "patient_correct":      patient_correct,
            "majority_pred":        majority_pred,
            "majority_class":       IDX_TO_CLASS[majority_pred],
            "majority_correct":     majority_correct,
            "n_images":             len(img_pred_rows),
            "n_correct_votes":      n_correct_votes,
            "frac_correct_votes":   frac_correct,
            "mil_majority_agree":   (patient_pred == majority_pred),
            "px_correct_mean":      np.mean(px_correct)   if px_correct else None,
            "px_correct_median":    np.median(px_correct) if px_correct else None,
            "px_wrong_mean":        np.mean(px_wrong)     if px_wrong   else None,
            "px_wrong_median":      np.median(px_wrong)   if px_wrong   else None,
            **{f"prob_{IDX_TO_CLASS[c][:15]}": float(patient_probs[c])
               for c in range(len(class_names))},
        })

        # Per-patient plot
        plot_patient(
            patient_id    = pid,
            true_class    = true_class,
            true_label    = true_label,
            patient_probs = patient_probs,
            patient_pred  = patient_pred,
            image_rows    = img_pred_rows,
            class_names   = class_names,
            attn          = attn,
            out_path      = plot_dir / f"{pid.replace(' ', '_')}.png",
        )

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    patient_df = pd.DataFrame(patient_rows)
    image_df   = pd.DataFrame(image_rows)

    patient_df.to_csv(out_dir / "patient_predictions.csv",
                      index=False, float_format="%.4f")
    image_df.to_csv(out_dir / "image_predictions.csv",
                    index=False, float_format="%.4f")

    # ── Summary stats ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Results ({args.split} split, {len(patient_df)} patients)")
    print(f"{'='*65}")
    print(f"  Patient-level accuracy : {patient_df['patient_correct'].mean():.3f}")
    print(f"  Majority vote accuracy : {patient_df['majority_correct'].mean():.3f}")
    print(f"  MIL == Majority vote   : {patient_df['mil_majority_agree'].mean():.3f}")
    print(f"\n  Mean fraction of correct image votes per patient:")
    print(f"    Overall              : {patient_df['frac_correct_votes'].mean():.3f}")
    print(f"    Correct patients     : "
          f"{patient_df[patient_df['patient_correct']]['frac_correct_votes'].mean():.3f}")
    print(f"    Wrong patients       : "
          f"{patient_df[~patient_df['patient_correct']]['frac_correct_votes'].mean():.3f}")

    # ── Per-class AUC ─────────────────────────────────────────────────────────
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize

    y_true  = patient_df["true_label"].values
    prob_cols = [f"prob_{IDX_TO_CLASS[c][:15]}" for c in range(len(class_names))]
    y_score = patient_df[prob_cols].values  # [N, C]

    print(f"\n  Per-class summary (patient-level):")
    print(f"    {'Class':<35} {'Acc':>6}  {'AUC':>6}  {'MajVote':>8}  {'ImgVote%':>9}  {'N'}")
    print(f"    {'-'*35} {'-'*6}  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*4}")
    for c, cls in enumerate(class_names):
        sub_p  = patient_df[patient_df["true_class"] == cls]
        sub_i  = image_df[image_df["true_class"] == cls]
        if len(sub_p) == 0:
            continue
        acc        = sub_p["patient_correct"].mean()
        maj_acc    = sub_p["majority_correct"].mean()
        img_pct    = sub_i["image_correct"].mean() if len(sub_i) else float("nan")
        # one-vs-rest AUC for this class
        try:
            auc = roc_auc_score(
                (y_true == c).astype(int),
                y_score[:, c]
            )
        except Exception:
            auc = float("nan")
        print(f"    {cls[:35]:<35} {acc:>6.3f}  {auc:>6.3f}  {maj_acc:>8.3f}  {img_pct:>9.3f}  {len(sub_p)}")

    print(f"\n  Pixel size: correct vs wrong voting images")
    px_c = image_df[image_df["image_correct"]]["pixelsize_nm"].dropna()
    px_w = image_df[~image_df["image_correct"]]["pixelsize_nm"].dropna()
    print(f"    Correct images: mean={px_c.mean():.1f} nm/px  "
          f"median={px_c.median():.1f} nm/px  (n={len(px_c)})")
    print(f"    Wrong images:   mean={px_w.mean():.1f} nm/px  "
          f"median={px_w.median():.1f} nm/px  (n={len(px_w)})")

    print(f"\n  Outputs saved to: {out_dir}/")
    print(f"  Patient plots   : {plot_dir}/")


if __name__ == "__main__":
    main()