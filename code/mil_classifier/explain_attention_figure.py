"""
explain_attention_figure.py
─────────────────────────────────────────────────────────────────────────────
Publication-quality multi-panel attention figure for EM3 MIL models.

Layout
------
  • Column labels (Original / Attention / Overlay) appear ONCE at the top.
  • One row per image:
        [a]  Disease Name (italic)
             [original]  [attention]  [overlay]
  • Gradient colorbar on the right: blue = low attention, red = high.

Output: JPEG (default), TIFF, or EPS at ≥ 300 DPI.
Font:   Arial → Calibri → Helvetica → DejaVu Sans fallback chain.

Usage
-----
    python explain_attention_figure.py \\
        --run-dir  checkpoints/<run>/    \\
        --images   images.json           \\
        --out      figure_attention.jpg  \\
        --dpi      300                   \\
        --font-size 9

images.json format
------------------
    [
        {"path": "/abs/path/image1.tif", "disease": "IgA nephropathy"},
        {"path": "/abs/path/image2.tif", "disease": "Membranous nephropathy"}
    ]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))


# ── Early config apply — must run before any local import ────────────────────

def _apply_config_early():
    import yaml
    _p = argparse.ArgumentParser(add_help=False)
    _p.add_argument("--run-dir", default=None)
    _p.add_argument("--config",  default=None)
    _a, _ = _p.parse_known_args()
    cfg_path = _a.config
    if cfg_path is None and _a.run_dir:
        ymls = sorted(Path(_a.run_dir).glob("*.yml"))
        if ymls:
            cfg_path = str(ymls[0])
    if cfg_path:
        from config import cfg as _cfg
        with open(cfg_path) as _f:
            ov = yaml.safe_load(_f) or {}
        for k, v in ov.items():
            if hasattr(_cfg, k):
                setattr(_cfg, k, v)
        print(f"  Config: {cfg_path}")


_apply_config_early()

from config  import cfg
from dataset import get_transforms, load_tabular_features
from model   import build_model
from explain import load_checkpoint, get_dinov2_attention, get_display_transform

# ── Global style ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Calibri", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype":    42,   # embed fonts as TrueType in PDF/EPS
    "ps.fonttype":     42,
})

CMAP  = "jet"   # blue (low) → cyan → yellow → red (high)
ALPHA = 0.5     # overlay blend weight


# ── Per-image computation ─────────────────────────────────────────────────────

def compute_row(model, img_path: str, eval_transform, display_transform,
                device, alpha: float):
    """
    Returns (orig_arr, heatmap_arr, overlay_arr) — all [H,W,3] uint8 — or None.
    """
    attn_map = get_dinov2_attention(
        model, img_path, eval_transform, device, patch_size=14
    )
    if attn_map is None:
        return None

    try:
        orig = np.array(display_transform(Image.open(img_path).convert("RGB")))
    except Exception as e:
        print(f"  WARNING: cannot load {img_path}: {e}")
        return None

    cmap_fn = matplotlib.colormaps.get_cmap(CMAP) if hasattr(matplotlib, "colormaps") \
              else cm.get_cmap(CMAP)
    heatmap = (cmap_fn(attn_map)[:, :, :3] * 255).astype(np.uint8)
    overlay = (alpha * heatmap + (1 - alpha) * orig).astype(np.uint8)
    return orig, heatmap, overlay


# ── Figure builder ────────────────────────────────────────────────────────────

def make_figure(rows: list, out_path: Path, dpi: int = 300, font_size: int = 9):
    """
    rows: [(disease_name, orig_arr, heatmap_arr, overlay_arr), ...]

    Layout (GridSpec):
        row 0              — column headers (Original / Attention / Overlay)
        rows 1, 3, 5, ...  — title rows: bold letter left-aligned, disease name centred
        rows 2, 4, 6, ...  — image panel rows
    """
    n = len(rows)

    # ── Dimensions ─────────────────────────────────────────────────────────────
    # full-text width ≈ 18 cm = 7.09 in.
    FIG_W   = 7.09   # inches
    PANEL_H = 1.80   # inches per image-panel row
    TITLE_H = 0.28   # inches per title row
    HDR_H   = 0.50   # inches for the column-header row (label + colorbar below)
    MARGIN  = 0.01   # symmetric left/right margin
    WSPACE  = 0.02
    LEFT    = MARGIN
    RIGHT   = 1.0 - MARGIN
    TOP     = 0.99
    BOT     = 0.01

    # Match PANEL_H to image aspect ratio so cells fit images without whitespace
    h_px, w_px = rows[0][1].shape[:2]   # orig array shape (H, W, 3)
    col_w_in   = (RIGHT - LEFT) * FIG_W / 3
    PANEL_H    = col_w_in * h_px / w_px

    fig_h = HDR_H + n * (TITLE_H + PANEL_H) + 0.05

    fig = plt.figure(figsize=(FIG_W, fig_h))

    # ── GridSpec ───────────────────────────────────────────────────────────────
    height_ratios = [HDR_H] + [h for _ in range(n) for h in (TITLE_H, PANEL_H)]

    gs = fig.add_gridspec(
        1 + 2 * n, 3,
        height_ratios=height_ratios,
        left=LEFT, right=RIGHT,
        top=TOP, bottom=BOT,
        hspace=0.0, wspace=WSPACE,
    )


    # ── Column headers (row 0) ─────────────────────────────────────────────────
    for j, label in enumerate(["Original", "Attention", "Overlay"]):
        ax = fig.add_subplot(gs[0, j])
        ax.axis("off")
        if j == 1:
            # "Attention" at same height as Original / Overlay
            ax.text(0.5, 0.5, label,
                    ha="center", va="center",
                    fontsize=font_size, fontweight="bold",
                    transform=ax.transAxes)
            # Colorbar in the lower portion, "Low"/"High" flanking it
            cbar_ax = ax.inset_axes([0.18, -0.19, 0.64, 0.28])
            cb = fig.colorbar(
                plt.cm.ScalarMappable(cmap=CMAP, norm=Normalize(0, 1)),
                cax=cbar_ax,
                orientation="horizontal",
            )
            cb.set_ticks([])
            ax.text(0.15, -0.05, "Low",  ha="right", va="center",
                    fontsize=font_size - 2, transform=ax.transAxes)
            ax.text(0.85, -0.05, "High", ha="left",  va="center",
                    fontsize=font_size - 2, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, label,
                    ha="center", va="center",
                    fontsize=font_size, fontweight="bold",
                    transform=ax.transAxes)

    # ── Image rows ─────────────────────────────────────────────────────────────
    for i, (disease, orig, heatmap, overlay) in enumerate(rows):
        title_row = 1 + 2 * i
        img_row   = 2 + 2 * i
        letter    = chr(ord("a") + i)

        # Title row: bold letter left-aligned + disease name centred (not bold)
        ax_t = fig.add_subplot(gs[title_row, :])
        ax_t.axis("off")
        ax_t.text(0.0, 0.5, letter,
                  ha="left", va="center",
                  fontsize=font_size + 2, fontweight="bold",
                  transform=ax_t.transAxes)
        ax_t.text(0.04, 0.5, disease,
                  ha="left", va="center",
                  fontsize=font_size, fontstyle="italic",
                  transform=ax_t.transAxes)

        # Three image panels
        for j, arr in enumerate([orig, heatmap, overlay]):
            ax = fig.add_subplot(gs[img_row, j])
            ax.imshow(arr)
            ax.axis("off")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower().lstrip(".")
    kw  = dict(dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    if ext in ("jpg", "jpeg"):
        kw["format"] = "jpeg"
        kw["pil_kwargs"] = {"quality": 95}
    elif ext == "tiff":
        kw["format"] = "tiff"
    elif ext == "eps":
        kw["format"] = "eps"
    fig.savefig(out_path, **kw)
    plt.close(fig)
    print(f"  Saved → {out_path}  ({dpi} DPI, {ext.upper()})")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Publication attention figure for EM3 MIL models."
    )
    parser.add_argument("--run-dir",    required=True,
                        help="Run directory with checkpoint + config .yml")
    parser.add_argument("--images",     required=True,
                        help='JSON file: [{"path": ..., "disease": ...}, ...]')
    parser.add_argument("--out",        default="attention_figure.jpg",
                        help="Output file (.jpg / .tiff / .eps, default: .jpg)")
    parser.add_argument("--dpi",        type=int, default=300,
                        help="Output DPI (default: 300, minimum for JPEG halftone)")
    parser.add_argument("--font-size",  type=int, default=9,
                        help="Base font size in pt (default: 9)")
    parser.add_argument("--alpha",      type=float, default=0.5,
                        help="Attention overlay blend weight 0–1 (default: 0.5)")
    parser.add_argument("--config",     default=None,
                        help="Config .yml override (default: auto-detect in --run-dir)")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint filename in --run-dir (default: auto best)")
    args = parser.parse_args()

    import torch
    run_dir = Path(args.run_dir)
    device  = torch.device(cfg.DEVICE)

    with open(args.images) as f:
        entries = json.load(f)
    print(f"\n  {len(entries)} image(s) to process")

    # ── Resolve checkpoint ────────────────────────────────────────────────────
    if args.checkpoint is None:
        cands = [p for p in sorted(run_dir.glob("*.pt")) if "best" in p.name.lower()]
        if len(cands) != 1:
            raise FileNotFoundError(
                f"Expected exactly 1 'best' checkpoint in {run_dir}, "
                f"found: {[p.name for p in cands]}. Use --checkpoint."
            )
        ckpt = cands[0]
    else:
        ckpt = Path(args.checkpoint)
        if not ckpt.is_absolute():
            ckpt = run_dir / ckpt

    # ── Sync NUM_CLASSES from checkpoint (same as explain.py) ─────────────────
    _raw    = torch.load(ckpt, map_location="cpu")
    _ckpt_nc = _raw.get("cfg", {}).get("NUM_CLASSES", None)
    if _ckpt_nc is not None and _ckpt_nc != cfg.NUM_CLASSES:
        print(f"  WARNING: checkpoint NUM_CLASSES={_ckpt_nc} != cfg; using checkpoint value")
        cfg.NUM_CLASSES = _ckpt_nc
    del _raw

    # ── Build model ────────────────────────────────────────────────────────────
    # Determine tab_dim so the architecture matches the checkpoint exactly.
    has_tab = getattr(cfg, "TAB_FUSION", False)
    tab_dim = 0
    if has_tab:
        _, _, tab_dim = load_tabular_features(
            csv_path=cfg.TAB_FEATURES_CSV,
            id_col=cfg.TAB_ID_COL,
            marker_cols=cfg.TAB_MARKER_COLS,
        )

    model = build_model(tab_dim=tab_dim).to(device)
    model = load_checkpoint(ckpt, model, device)
    model.eval()

    eval_t    = get_transforms(train=False)
    display_t = get_display_transform()

    # ── Process each image ────────────────────────────────────────────────────
    rows = []
    for entry in entries:
        path    = entry["path"]
        disease = entry.get("disease", "")
        print(f"  {Path(path).name}  ({disease})")
        result = compute_row(model, path, eval_t, display_t, device, args.alpha)
        if result is None:
            print(f"    SKIPPED — attention map failed")
            continue
        orig, heatmap, overlay = result
        rows.append((disease, orig, heatmap, overlay))

    if not rows:
        print("ERROR: no valid rows, nothing saved.")
        return

    make_figure(rows, Path(args.out), dpi=args.dpi, font_size=args.font_size)
    print(f"\nDone. {len(rows)}/{len(entries)} rows → {args.out}")


if __name__ == "__main__":
    main()
