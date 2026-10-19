"""
ensemble_cv_evaluation.py
─────────────────────────────────────────────────────────────────────────────
Evaluates an ensemble of cross-validation models (one per fold) on both the
validation set and a held-out test set.

Expected folder structure for each fold model:
    <model_dir>/
        metrics_best.json           — validation metrics
        predictions_best.csv        — validation predictions
        <test_subdir>/
            metrics_test.json       — test metrics
            predictions_test.csv    — test predictions

Outputs (all saved to --out):
    agg_metrics_validation.json               — mean ± std across folds (val set)
    confusion_matrix_val_aggregated.png       — CM over all fold val predictions concatenated
    roc_val_oof.png                           — OOF ROC curves (concatenated val predictions)
    agg_metrics_test.json                     — mean ± std across folds (test set, per-model)
    confusion_matrix_test_fold_mean.png       — CM concatenated across fold models; counts are
                                                mean per fold (total / n_folds)
    roc_test_per_fold.png                     — per-fold test ROC: mean curve per class
    roc_test_per_fold_band.png                — per-fold test ROC: mean ± std band per class
    ensemble_test_metrics.json                — ensemble metrics on test set
    ensemble_test_predictions.csv             — per-patient ensemble predictions on test set
    confusion_matrix_ensemble_<strategy>.png  — one CM per ensemble strategy
    roc_test_ensemble.png                     — test ROC from mean_prob ensemble predictions

Usage:
    python ensemble_cv_evaluation.py \\
        --models  /path/fold_0 /path/fold_1 /path/fold_2 /path/fold_3 /path/fold_4 \\
        --out     /path/to/output_dir \\
        [--test-subdir   test_results_max50] \\
        [--labels        "IgA nephropathy" "Alport syndrome/thin membrane disease" ...] \\
        [--strategies    mean_prob weighted_mean majority_vote]
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import sys
import numpy as np
import pandas as pd
from pathlib import Path as _Path
from sklearn.metrics import confusion_matrix
sys.path.insert(0, str(_Path(__file__).parent))
from training import compute_metrics


# ── Short disease name map ────────────────────────────────────────────────────
_SHORT_NAME_MAP = {
    "IgA nephropathy":                                "IgAN",
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
    return _SHORT_NAME_MAP.get(name, name)


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_LABELS = [
    "IgA nephropathy",
    "Alport syndrome/thin membrane disease",
    "Membranous nephropathy",
    "Diabetic kidney disease",
    "Minimal change disease (MCD)",
    "Lupus nephritis",
]

PROB_COLS  = [f"prob_{c}" for c in range(6)]
STRATEGIES = ["mean_prob", "max_prob", "weighted_mean", "majority_vote"]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_predictions(path: Path, n_classes: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    prob_cols = [f"prob_{c}" for c in range(n_classes)]
    required  = {"patient_id", "true_label"} | set(prob_cols)
    missing   = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def load_metrics(path: Path) -> dict:
    with open(path) as f:
        raw = json.load(f)
    # Extract scalar metrics from nested or flat structure
    m = raw["metrics"] if ("metrics" in raw and isinstance(raw["metrics"], dict)) else raw
    # Normalise shortened key names produced by training.py compute_metrics()
    result = dict(m)
    # Also extract per-class f1/precision/recall from per_class section if present
    per_class = raw.get("per_class", {})
    per_class_f1  = {}
    per_class_prec = {}
    per_class_rec  = {}
    for cls, vals in per_class.items():
        if isinstance(vals, dict) and "f1-score" in vals:
            per_class_f1[cls]   = vals["f1-score"]
            per_class_prec[cls] = vals["precision"]
            per_class_rec[cls]  = vals["recall"]
    if per_class_f1:
        result["per_class_f1"]        = per_class_f1
        result["per_class_precision"] = per_class_prec
        result["per_class_recall"]    = per_class_rec
    return result


def load_classes_from_yml(model_dirs: list,
                          fallback: list) -> list:
    """
    Read CLASSES from the .yml config inside each model directory.
    Verifies all folds agree on the same class list.
    Falls back to --labels if no yml is found in any directory.
    """
    import yaml

    all_classes = {}  # model_name -> list[str]
    for model_dir in model_dirs:
        yml_files = sorted(model_dir.glob("*.yml"))
        if not yml_files:
            print(f"  WARNING: no .yml found in {model_dir.name} — "
                  f"will use --labels fallback if all dirs are missing yml.")
            continue
        with open(yml_files[0]) as f:
            cfg = yaml.safe_load(f) or {}
        classes = cfg.get("CLASSES")
        if classes is None:
            print(f"  WARNING: 'CLASSES' key not found in {yml_files[0].name} "
                  f"for {model_dir.name}.")
            continue
        all_classes[model_dir.name] = classes

    if not all_classes:
        print(f"  WARNING: No CLASSES found in any yml — "
              f"using --labels: {fallback}")
        return fallback

    # Check consistency across folds
    unique = list({tuple(v) for v in all_classes.values()})
    if len(unique) > 1:
        raise ValueError(
            "Models have inconsistent CLASSES across folds:\n" +
            "\n".join(f"  {k}: {v}" for k, v in all_classes.items())
        )

    classes = list(unique[0])
    print(f"  CLASSES loaded from yml ({len(classes)} classes): {classes}")
    return classes


def load_fold(model_dir: Path, test_subdir: str, n_classes: int, calibrated: bool = False):
    """
    Load one fold's validation and test predictions + metrics.
    Returns (val_df, val_metrics, test_df, test_metrics, fold_name).

    calibrated: if True, reads predictions_test_calibrated.csv / metrics_test_calibrated.json
    """
    fold_name = model_dir.name

    val_pred_path    = model_dir / "predictions_best.csv"
    test_dir         = model_dir / test_subdir
    suffix           = "_calibrated" if calibrated else ""
    test_pred_path   = test_dir  / f"predictions_test{suffix}.csv"
    test_metrics_path = test_dir / f"metrics_test{suffix}.json"

    # Prefer metrics_val.json (recomputed by test.py with full metrics) over
    # metrics_best.json (saved during training, may be missing newer keys).
    val_metrics_path = test_dir / "metrics_val.json"
    if not val_metrics_path.exists():
        val_metrics_path = model_dir / "metrics_best.json"
        print(f"  [{fold_name}] metrics_val.json not found, falling back to metrics_best.json")

    for p in (val_pred_path, val_metrics_path, test_pred_path, test_metrics_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    val_df      = load_predictions(val_pred_path,  n_classes)
    val_metrics = load_metrics(val_metrics_path)
    test_df     = load_predictions(test_pred_path, n_classes)
    test_metrics = load_metrics(test_metrics_path)

    val_auc  = val_metrics.get("auc", float("nan"))
    test_auc = test_metrics.get("auc", float("nan"))
    print(f"  {fold_name:<30}  val AUC={val_auc:.4f}  test AUC={test_auc:.4f}  "
          f"val_n={len(val_df)}  test_n={len(test_df)}")

    return val_df, val_metrics, test_df, test_metrics, fold_name


# ── Aggregation (mean ± std across folds) ─────────────────────────────────────

SCALAR_METRICS = ["acc", "top2_acc", "bacc", "auc", "f1_macro", "f1_weighted",
                  "youden_threshold", "youden_sensitivity", "youden_specificity",
                  "youden_acc", "youden_bacc", "youden_f1_macro"]


def aggregate_fold_metrics(fold_metrics_list: list, class_names: list) -> dict:
    """
    Given a list of per-fold metric dicts, compute mean and std
    for each scalar metric and each per-class metric.
    """
    result = {}

    # Scalar metrics — skip keys absent for this task (e.g. Youden for n_classes > 2)
    for key in SCALAR_METRICS:
        raw    = [m.get(key, float("nan")) for m in fold_metrics_list]
        values = [v for v in raw if not np.isnan(v)]
        if not values:
            continue
        result[key] = {
            "mean":     float(np.mean(values)),
            "std":      float(np.std(values)),
            "per_fold": raw,
        }

    # Per-class AUC
    result["per_class_auc"] = {}
    for cls in class_names:
        values = [
            m.get("per_class_auc", {}).get(cls, float("nan"))
            for m in fold_metrics_list
        ]
        values_clean = [v for v in values if not np.isnan(v)]
        result["per_class_auc"][cls] = {
            "mean": float(np.mean(values_clean)) if values_clean else float("nan"),
            "std":  float(np.std(values_clean))  if values_clean else float("nan"),
            "per_fold": values,
        }

    # Per-class accuracy
    result["per_class_acc"] = {}
    for cls in class_names:
        values = [
            m.get("per_class_acc", {}).get(cls, float("nan"))
            for m in fold_metrics_list
        ]
        values_clean = [v for v in values if not np.isnan(v)]
        result["per_class_acc"][cls] = {
            "mean": float(np.mean(values_clean)) if values_clean else float("nan"),
            "std":  float(np.std(values_clean))  if values_clean else float("nan"),
            "per_fold": values,
        }

    # Per-class F1 from saved json (if available)
    sample = fold_metrics_list[0]
    for metric_key in ("per_class_f1", "per_class_precision", "per_class_recall"):
        if metric_key not in sample:
            continue
        result[metric_key] = {}
        for cls in sample[metric_key]:
            values = [
                m.get(metric_key, {}).get(cls, float("nan"))
                for m in fold_metrics_list
            ]
            values_clean = [v for v in values if not np.isnan(v)]
            result[metric_key][cls] = {
                "mean":     float(np.mean(values_clean)) if values_clean else float("nan"),
                "std":      float(np.std(values_clean))  if values_clean else float("nan"),
                "per_fold": values,
            }

    result["n_folds"] = len(fold_metrics_list)
    return result


# ── Ensemble strategies ───────────────────────────────────────────────────────

def _mean_prob(prob_matrix):
    return prob_matrix.mean(axis=0)

def _max_prob(prob_matrix):
    return prob_matrix.max(axis=0)

def _weighted_mean(prob_matrix, weights):
    w = np.array(weights) / np.array(weights).sum()
    return (prob_matrix * w[:, None]).sum(axis=0)

def _majority_vote(prob_matrix):
    n_classes  = prob_matrix.shape[1]
    hard_preds = prob_matrix.argmax(axis=1)
    counts     = np.bincount(hard_preds, minlength=n_classes)
    max_count  = counts.max()
    tied       = np.where(counts == max_count)[0]
    if len(tied) == 1:
        winner = tied[0]
    else:
        mean_probs = prob_matrix.mean(axis=0)
        winner = tied[np.argmax(mean_probs[tied])]
    result = np.zeros(n_classes)
    result[winner] = 1.0
    return result


def run_ensemble(patient_data: dict, class_names: list, strategies: list,
                 youden_threshold: float = None):
    """
    Run ensemble strategies over patient_data.
    patient_data[pid] = {
        "true_label": int,
        "true_class": str,
        "model_probs": {fold_name: np.array},
        "model_auc":   {fold_name: float},
    }
    Returns (predictions_df, metrics_dict).
    """
    n_classes    = len(class_names)
    all_patients = sorted(patient_data.keys())
    strategy_results  = {}
    prediction_rows   = []

    for strategy in strategies:
        y_true, y_pred, y_score = [], [], []

        for pid in all_patients:
            entry  = patient_data[pid]
            avail  = list(entry["model_probs"].keys())
            if not avail:
                continue

            prob_matrix = np.stack(
                [entry["model_probs"][m] for m in avail], axis=0
            )  # [n_models, n_classes]
            auc_weights = np.array(
                [entry["model_auc"][m] for m in avail], dtype=float
            )

            if strategy == "mean_prob":
                agg = _mean_prob(prob_matrix)
            elif strategy == "max_prob":
                agg = _max_prob(prob_matrix)
            elif strategy == "weighted_mean":
                agg = _weighted_mean(prob_matrix, auc_weights)
            elif strategy == "majority_vote":
                agg = _majority_vote(prob_matrix)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            pred_label = int(np.argmax(agg))
            pred_class = class_names[pred_label] if pred_label < n_classes else str(pred_label)

            # score_vec is used for AUC (and saved to prob_* for bootstrap CI).
            # For majority_vote, agg is a one-hot hard vote; use mean_prob as
            # the score so bootstrap_ci.py computes AUC from the same values
            # as the point estimate here.  For all other strategies agg == score.
            score_vec = _mean_prob(prob_matrix) if strategy == "majority_vote" else agg

            y_true.append(entry["true_label"])
            y_pred.append(pred_label)
            y_score.append(score_vec)

            row = {
                "strategy":      strategy,
                "patient_id":    pid,
                "true_label":    entry["true_label"],
                "true_class":    entry["true_class"],
                "pred_label":    pred_label,
                "pred_class":    pred_class,
                "correct":       pred_label == entry["true_label"],
                "n_models_used": len(avail),
            }
            for c in range(n_classes):
                row[f"prob_{c}"] = float(score_vec[c])
            prediction_rows.append(row)

        y_score_mat = np.stack(y_score, axis=0)
        metrics = compute_metrics(y_score_mat.tolist(), y_true, n_classes,
                                  class_names=class_names,
                                  youden_threshold=youden_threshold)
        strategy_results[strategy] = metrics
        print(f"    {strategy:<20}  N={metrics['n_patients']}  "
              f"Acc={metrics['acc']:.3f}  "
              f"AUC={metrics['auc']:.3f}  "
              f"F1={metrics['f1_macro']:.3f}")

    return pd.DataFrame(prediction_rows), strategy_results


# ── Confusion matrix ──────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, class_names, title, path, count_divisor=1):
    """
    Save a row-normalised confusion matrix as a PNG.

    count_divisor: divide raw cell counts by this before displaying.
        1         → show integer counts  e.g. "(12)"
        n_folds   → show mean-per-fold counts  e.g. "(3.2)"
    Colours are always derived from the row-normalised fractions.
    """
    import matplotlib.pyplot as plt

    n      = len(class_names)
    short  = [_short_name(name) for name in class_names]

    cm_raw  = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True).clip(min=1)

    _, ax = plt.subplots(figsize=(max(6, n), max(5, n)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=9)

    for i in range(n):
        for j in range(n):
            colour = "white" if cm_norm[i, j] > 0.5 else "black"
            mean_count = cm_raw[i, j] / count_divisor
            count_str  = f"{mean_count:.1f}" if count_divisor > 1 else str(cm_raw[i, j])
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({count_str})",
                    ha="center", va="center", fontsize=7, color=colour)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix → {path}")


# ── ROC curves ────────────────────────────────────────────────────────────────

def save_roc_curves(y_true, y_prob, class_names, title, path):
    """
    Single-dataset OvR ROC: one curve per class + macro average.

    y_true : array-like [N] int
    y_prob : array-like [N, C] float — predicted probabilities
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    C      = len(class_names)
    short  = [_short_name(n) for n in class_names]
    colors = cm.tab10(np.linspace(0, 0.9, C))
    grid   = np.linspace(0, 1, 300)

    y_bin  = label_binarize(y_true, classes=list(range(C)))

    fig, ax = plt.subplots(figsize=(7, 6))
    interp_tprs = []

    for c in range(C):
        if y_bin[:, c].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
        roc_auc     = auc(fpr, tpr)
        itpr        = np.interp(grid, fpr, tpr);  itpr[0] = 0.0
        interp_tprs.append(itpr)
        ax.plot(fpr, tpr, color=colors[c], lw=1.8,
                label=f"{short[c]}  (AUC={roc_auc:.3f})")

    if interp_tprs:
        mean_tpr      = np.mean(interp_tprs, axis=0);  mean_tpr[-1] = 1.0
        macro_auc     = auc(grid, mean_tpr)
        ax.plot(grid, mean_tpr, color="black", lw=2.2, linestyle="--",
                label=f"Macro avg  (AUC={macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, linestyle=":", label="Random")
    ax.set_xlim([0, 1]);  ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate");  ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC curves → {path}")


def save_roc_curves_multifold(y_true, fold_probs, class_names, title, path,
                              with_band=True):
    """
    Multi-fold OvR ROC: compute one curve per fold per class, then plot the
    mean (± std band if with_band=True) across folds.

    y_true     : array [N] int — same test labels for every fold
    fold_probs : list of K arrays each [N, C] — one per fold model
    with_band  : if True, shade ±1 std around the mean curve
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    y_true = np.asarray(y_true)
    C      = len(class_names)
    K      = len(fold_probs)
    short  = [_short_name(n) for n in class_names]
    colors = cm.tab10(np.linspace(0, 0.9, C))
    grid   = np.linspace(0, 1, 300)

    y_bin  = label_binarize(y_true, classes=list(range(C)))

    fig, ax = plt.subplots(figsize=(7, 6))
    macro_tprs = []

    for c in range(C):
        if y_bin[:, c].sum() == 0:
            continue
        tprs, aucs = [], []
        for yp in fold_probs:
            yp = np.asarray(yp)
            fpr, tpr, _ = roc_curve(y_bin[:, c], yp[:, c])
            aucs.append(auc(fpr, tpr))
            itpr = np.interp(grid, fpr, tpr);  itpr[0] = 0.0
            tprs.append(itpr)

        mean_tpr = np.mean(tprs, axis=0);  mean_tpr[-1] = 1.0
        std_tpr  = np.std(tprs, axis=0)
        mean_auc = float(np.mean(aucs))
        std_auc  = float(np.std(aucs))
        macro_tprs.append(mean_tpr)

        ax.plot(grid, mean_tpr, color=colors[c], lw=1.8,
                label=f"{short[c]}  (AUC={mean_auc:.3f}±{std_auc:.3f})")
        if with_band:
            ax.fill_between(grid,
                            np.clip(mean_tpr - std_tpr, 0, 1),
                            np.clip(mean_tpr + std_tpr, 0, 1),
                            color=colors[c], alpha=0.12)

    if macro_tprs:
        grand_mean     = np.mean(macro_tprs, axis=0);  grand_mean[-1] = 1.0
        grand_std      = np.std(macro_tprs, axis=0)
        macro_auc      = auc(grid, grand_mean)
        ax.plot(grid, grand_mean, color="black", lw=2.2, linestyle="--",
                label=f"Macro avg  (AUC={macro_auc:.3f})")
        if with_band:
            ax.fill_between(grid,
                            np.clip(grand_mean - grand_std, 0, 1),
                            np.clip(grand_mean + grand_std, 0, 1),
                            color="black", alpha=0.08)

    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, linestyle=":", label="Random")
    ax.set_xlim([0, 1]);  ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate");  ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC curves → {path}")


# ── Combined 3×2 figure (CM + ROC, three splits) ─────────────────────────────

def save_combined_figure(
    # Row a — Validation OOF
    val_cm_true, val_cm_pred, val_roc_true, val_roc_prob,
    # Row b — Test per-fold  (CM uses concatenated N×K rows; ROC uses fold list)
    test_cm_true, test_cm_pred, test_cm_count_div,
    test_roc_true, test_fold_probs,
    # Row c — Ensemble
    ens_cm_true, ens_cm_pred, ens_roc_true, ens_roc_prob,
    class_names, path,
    font_size=9, dpi=300,
):
    """
    3 rows × 2 cols publication figure:
      a  Validation (OOF)   │  CM  │  ROC
      b  Test (per-fold)    │  CM  │  ROC  (mean of fold curves, no band)
      c  Ensemble           │  CM  │  ROC

    Styled to match explain_attention_figure.py:
    Arial font, square ROC axes, a/b/c labels in left margin.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as mpl_cm
    from sklearn.metrics import confusion_matrix as sk_cm, roc_curve, auc, roc_auc_score
    from sklearn.preprocessing import label_binarize

    plt.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Arial", "Calibri", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype":  42,
    })

    C      = len(class_names)
    short  = [_short_name(c) for c in class_names]
    colors = mpl_cm.tab10(np.linspace(0, 0.9, C))
    grid   = np.linspace(0, 1, 300)

    val_n   = len(val_cm_true)
    test_n  = len(ens_cm_true)
    n_folds = test_cm_count_div
    ROW_LABELS = [
        f"Validation OOF ({val_n} patients, {n_folds} folds concatenated)",
        f"Test ({test_n} patients, averaged across {n_folds} CV models)",
        f"Test Ensemble ({test_n} patients, mean probability across {n_folds} CV models)",
    ]

    # ── Layout ────────────────────────────────────────────────────────────────
    FIRST_TITLE_H = 0.35  # row a only — no panel above, no need for spill space
    TITLE_H = 1.00   # rows b/c — accommodates x-tick spill from panel above
    FIG_W   = 7.09   # full text width ≈ 18 cm
    LEFT    = 0.15   # True label + y-tick labels (letter now in title row)
    RIGHT   = 0.97
    WSPACE  = 0.55
    # col_w = (RIGHT-LEFT)*FIG_W / (2+WSPACE); PANEL_H matches for square CM cells
    PANEL_H = round((RIGHT - LEFT) * FIG_W / (2 + WSPACE), 2)
    TOP     = 0.98
    BOT     = 0.01
    N_ROWS  = 3

    fig_h = FIRST_TITLE_H + (N_ROWS - 1) * TITLE_H + N_ROWS * PANEL_H + 0.05
    fig   = plt.figure(figsize=(FIG_W, fig_h))

    hr = [FIRST_TITLE_H, PANEL_H] + [h for _ in range(N_ROWS - 1) for h in (TITLE_H, PANEL_H)]
    gs = fig.add_gridspec(
        2 * N_ROWS, 2,
        height_ratios=hr,
        left=LEFT, right=RIGHT,
        top=TOP, bottom=BOT,
        hspace=0.06, wspace=WSPACE,
    )

    # ── Helper: draw one CM onto ax ───────────────────────────────────────────
    def _draw_cm(ax, y_true, y_pred, count_div=1):
        cm_raw  = sk_cm(y_true, y_pred, labels=list(range(C)))
        cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.03)
        cbar.ax.tick_params(labelsize=font_size - 3)
        ax.set_xticks(range(C))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=font_size - 2)
        ax.set_yticks(range(C))
        ax.set_yticklabels(short, fontsize=font_size - 2)
        ax.set_xlabel("Predicted", fontsize=font_size, fontstyle="italic", labelpad=0)
        ax.xaxis.set_label_coords(0.5, -0.20)
        ax.tick_params(axis="both", labelsize=font_size - 2)
        for ri in range(C):
            for ci in range(C):
                col = "white" if cm_norm[ri, ci] > 0.5 else "black"
                cnt = cm_raw[ri, ci] / count_div
                cnt_str = f"{cnt:.1f}" if count_div > 1 else str(cm_raw[ri, ci])
                ax.text(ci, ri, f"{cm_norm[ri, ci]:.2f}\n({cnt_str})",
                        ha="center", va="center",
                        fontsize=font_size - 3, color=col)

    # ── Helper: draw single-set ROC onto ax ──────────────────────────────────
    def _draw_roc_single(ax, y_true, y_prob):
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        y_bin  = label_binarize(y_true, classes=list(range(C)))
        interp = []
        for c in range(C):
            if y_bin[:, c].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
            roc_auc = auc(fpr, tpr)
            itpr    = np.interp(grid, fpr, tpr);  itpr[0] = 0.0
            interp.append(itpr)
            ax.plot(fpr, tpr, color=colors[c], lw=1.2,
                    label=f"{short[c]}  ({roc_auc:.3f})")
        if interp:
            mean_tpr  = np.mean(interp, axis=0);  mean_tpr[-1] = 1.0
            row_sums  = y_prob.sum(axis=1, keepdims=True)
            y_prob_n  = y_prob / np.where(row_sums > 0, row_sums, 1.0)
            macro_auc = roc_auc_score(y_true, y_prob_n, multi_class="ovr", average="macro")
            ax.plot(grid, mean_tpr, color="black", lw=1.6, linestyle="--",
                    label=f"Macro  ({macro_auc:.3f})")

    # ── Helper: draw multi-fold ROC (mean, no band) onto ax ──────────────────
    def _draw_roc_multifold(ax, y_true, fold_probs):
        y_true = np.asarray(y_true)
        y_bin  = label_binarize(y_true, classes=list(range(C)))
        macro_tprs = []
        class_mean_aucs = []
        for c in range(C):
            if y_bin[:, c].sum() == 0:
                continue
            tprs, aucs = [], []
            for yp in fold_probs:
                fpr, tpr, _ = roc_curve(y_bin[:, c], np.asarray(yp)[:, c])
                aucs.append(auc(fpr, tpr))
                itpr = np.interp(grid, fpr, tpr);  itpr[0] = 0.0
                tprs.append(itpr)
            mean_tpr = np.mean(tprs, axis=0);  mean_tpr[-1] = 1.0
            macro_tprs.append(mean_tpr)
            class_mean_aucs.append(np.mean(aucs))
            ax.plot(grid, mean_tpr, color=colors[c], lw=1.2,
                    label=f"{short[c]}  ({np.mean(aucs):.3f}±{np.std(aucs):.3f})")
        if macro_tprs:
            grand = np.mean(macro_tprs, axis=0);  grand[-1] = 1.0
            ax.plot(grid, grand, color="black", lw=1.6, linestyle="--",
                    label=f"Macro  ({np.mean(class_mean_aucs):.3f})")

    def _style_roc(ax):
        ax.plot([0, 1], [0, 1], color="grey", lw=0.8, linestyle=":")
        ax.set_xlim([0, 1]);  ax.set_ylim([0, 1.02])
        # no set_aspect — fills row height, aligns with CM
        ax.set_xlabel("False Positive Rate", fontsize=font_size, fontstyle="italic")
        ax.set_ylabel("True Positive Rate", fontsize=font_size, fontstyle="italic")
        ax.tick_params(axis="both", labelsize=font_size - 1)
        ax.legend(loc="lower right", fontsize=font_size - 2,
                  framealpha=0.9, handlelength=1.5, borderpad=0.5)

    # ── Draw rows ─────────────────────────────────────────────────────────────
    rows = [
        # (row_label, cm_fn_args, roc_fn, roc_args)
        (ROW_LABELS[0],
         (val_cm_true, val_cm_pred, 1),
         _draw_roc_single, (val_roc_true, val_roc_prob)),
        (ROW_LABELS[1],
         (test_cm_true, test_cm_pred, test_cm_count_div),
         _draw_roc_multifold, (test_roc_true, test_fold_probs)),
        (ROW_LABELS[2],
         (ens_cm_true, ens_cm_pred, 1),
         _draw_roc_single, (ens_roc_true, ens_roc_prob)),
    ]

    for i, (row_label, cm_args, roc_fn, roc_args) in enumerate(rows):
        title_row = 2 * i
        panel_row = 2 * i + 1

        ax_t = fig.add_subplot(gs[title_row, :])
        ax_t.axis("off")
        y_txt  = 0.50 if i == 0 else 0.10
        va_txt = "center" if i == 0 else "bottom"
        letter = chr(ord("a") + i)
        ax_t.text(0.0, y_txt, letter, ha="left", va=va_txt,
                  fontsize=font_size + 3, fontweight="bold",
                  transform=ax_t.transAxes)
        ax_t.text(0.04, y_txt, row_label, ha="left", va=va_txt,
                  fontsize=font_size + 1,
                  transform=ax_t.transAxes)

        ax_cm = fig.add_subplot(gs[panel_row, 0])
        _draw_cm(ax_cm, *cm_args)

        ax_roc = fig.add_subplot(gs[panel_row, 1])
        roc_fn(ax_roc, *roc_args)
        _style_roc(ax_roc)

    # ── "True" y-label in left margin (one per panel row) ────────────────────
    total_h = sum(hr)
    usable  = TOP - BOT
    for i in range(N_ROWS):
        above_pnl = sum(hr[:2 * i + 1]) / total_h
        below_pnl = sum(hr[:2 * i + 2]) / total_h
        y_pnl = TOP - (above_pnl + below_pnl) / 2 * usable
        fig.text(0.07, y_pnl, "True", rotation=90,
                 ha="center", va="center",
                 fontsize=font_size, fontstyle="italic")

    # ── Save ──────────────────────────────────────────────────────────────────
    from pathlib import Path as _Path
    path = _Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower().lstrip(".")
    kw  = dict(dpi=dpi, bbox_inches="tight")
    if ext in ("jpg", "jpeg"):
        kw["format"] = "jpeg"
        kw["pil_kwargs"] = {"quality": 95}
    elif ext == "tiff":
        kw["format"] = "tiff"
    elif ext == "eps":
        kw["format"] = "eps"
    fig.savefig(path, **kw)
    plt.close(fig)
    print(f"  Combined figure → {path}  ({dpi} DPI)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-validation ensemble evaluation for EM3 MIL models."
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Paths to per-fold model directories"
    )
    parser.add_argument(
        "--out", required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--test-subdir", default="test_results",
        help="Subfolder name inside each model dir containing test results "
             "(default: test_results)"
    )
    parser.add_argument(
        "--labels", nargs="+", default=DEFAULT_LABELS,
        help="Class names in label index order"
    )
    parser.add_argument(
        "--strategies", nargs="+", default=STRATEGIES, choices=STRATEGIES,
        help="Ensemble strategies to run (default: all)"
    )
    parser.add_argument(
        "--calibrated", action="store_true",
        help="Use calibrated test predictions (predictions_test_calibrated.csv / "
             "metrics_test_calibrated.json) instead of the default files"
    )
    args = parser.parse_args()

    out_dir     = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CLASSES from yml files, falling back to --labels if not found
    print(f"\nChecking class definitions across {len(args.models)} model(s) ...")
    class_names = load_classes_from_yml(
        [Path(m) for m in args.models], fallback=args.labels
    )
    n_classes   = len(class_names)

    # ── Load all folds ────────────────────────────────────────────────────────
    print(f"\nLoading {len(args.models)} fold(s) ...")
    folds = []
    for m in args.models:
        fold = load_fold(Path(m), args.test_subdir, n_classes, calibrated=args.calibrated)
        folds.append(fold)
    # folds[i] = (val_df, val_metrics, test_df, test_metrics, fold_name)

    # ── 1. Aggregate validation metrics (mean ± std across folds) ────────────
    # Recompute from predictions so per_class_auc/acc are always populated,
    # regardless of whether metrics_val.json exists for each fold.
    print(f"\nAggregating validation metrics ...")
    val_metrics_list = []
    for val_df, _, _2, _3, fold_name in folds:
        prob_cols  = [f"prob_{c}" for c in range(n_classes)]
        val_probs  = val_df[prob_cols].values.tolist()
        val_labels = val_df["true_label"].tolist()
        val_metrics_list.append(
            compute_metrics(val_probs, val_labels, n_classes, class_names=class_names)
        )
    agg_val = aggregate_fold_metrics(val_metrics_list, class_names)

    val_out = out_dir / "agg_metrics_validation.json"
    with open(val_out, "w") as f:
        json.dump(agg_val, f, indent=2)
    print(f"  Saved → {val_out}")

    all_val_df = pd.concat([f[0] for f in folds], ignore_index=True)
    _prob_cols = [f"prob_{c}" for c in range(n_classes)]
    save_confusion_matrix(
        all_val_df["true_label"].tolist(),
        all_val_df[_prob_cols].values.argmax(1).tolist(),
        class_names,
        f"Validation — all folds concatenated ({len(all_val_df)} patients)",
        out_dir / "confusion_matrix_val_aggregated.png",
    )
    save_roc_curves(
        all_val_df["true_label"].values,
        all_val_df[_prob_cols].values,
        class_names,
        f"ROC — Validation OOF ({len(all_val_df)} patients, {len(folds)} folds concatenated)",
        out_dir / "roc_val_oof.png",
    )

    # OOF Youden threshold — fit on concatenated val predictions (one per patient)
    # Used as the ensemble test threshold to avoid deriving it from test data.
    oof_youden_threshold = None
    if n_classes == 2:
        oof_val_m = compute_metrics(
            all_val_df[[f"prob_{c}" for c in range(n_classes)]].values.tolist(),
            all_val_df["true_label"].tolist(),
            n_classes, class_names=class_names,
        )
        oof_youden_threshold = oof_val_m.get("youden_threshold")
        if oof_youden_threshold is not None:
            print(f"  OOF Youden threshold (ensemble): {oof_youden_threshold:.4f}")

    print(f"\n  Validation across {agg_val['n_folds']} folds:")
    for key in SCALAR_METRICS:
        if key not in agg_val:
            continue
        m = agg_val[key]
        print(f"    {key:<15}  {m['mean']:.4f} ± {m['std']:.4f}  "
              f"per_fold: {[f'{v:.4f}' for v in m['per_fold']]}")

    # ── 2. Aggregate test metrics (mean ± std across folds, per-model) ────────
    print(f"\nAggregating per-fold test metrics ...")

    # Recompute test metrics from predictions to ensure consistency
    # (in case metrics_test.json was saved with different class definitions)
    fold_test_metrics = []
    for _, val_m, test_df, test_m, fold_name in folds:
        y_true  = test_df["true_label"].tolist()
        y_score = test_df[[f"prob_{c}" for c in range(n_classes)]].values
        val_thresh = test_m.get("youden_threshold")
        m = compute_metrics(y_score.tolist(), y_true, n_classes,
                            class_names=class_names,
                            youden_threshold=val_thresh)
        fold_test_metrics.append(m)

    agg_test = aggregate_fold_metrics(fold_test_metrics, class_names)

    test_agg_out = out_dir / "agg_metrics_test.json"
    with open(test_agg_out, "w") as f:
        json.dump(agg_test, f, indent=2)
    print(f"  Saved → {test_agg_out}")

    all_test_df = pd.concat([f[2] for f in folds], ignore_index=True)
    save_confusion_matrix(
        all_test_df["true_label"].tolist(),
        all_test_df[_prob_cols].values.argmax(1).tolist(),
        class_names,
        f"Test — {len(folds)} fold models on same test set (counts = mean per fold)",
        out_dir / "confusion_matrix_test_fold_mean.png",
        count_divisor=len(folds),
    )

    # Per-fold test ROC: align all fold DataFrames by patient_id so curves are comparable
    _test_dfs_sorted = [f[2].sort_values("patient_id").reset_index(drop=True)
                        for f in folds]
    _y_true_test  = _test_dfs_sorted[0]["true_label"].values
    _fold_probs   = [df[_prob_cols].values for df in _test_dfs_sorted]
    save_roc_curves_multifold(
        _y_true_test, _fold_probs, class_names,
        f"ROC — Test, per-fold models (mean of {len(folds)} curves)",
        out_dir / "roc_test_per_fold.png",
        with_band=False,
    )
    save_roc_curves_multifold(
        _y_true_test, _fold_probs, class_names,
        f"ROC — Test, per-fold models (mean ± std across {len(folds)} folds)",
        out_dir / "roc_test_per_fold_band.png",
        with_band=True,
    )

    print(f"\n  Per-fold test metrics (individual models, same test set):")
    for key in SCALAR_METRICS:
        if key not in agg_test:
            continue
        m = agg_test[key]
        print(f"    {key:<15}  {m['mean']:.4f} ± {m['std']:.4f}  "
              f"per_fold: {[f'{v:.4f}' for v in m['per_fold']]}")

    # ── 3. Build test patient data for ensemble ───────────────────────────────
    print(f"\nBuilding ensemble on test set ...")

    patient_data = defaultdict(lambda: {
        "true_label":  None,
        "true_class":  None,
        "model_probs": {},
        "model_auc":   {},
    })

    for _, val_m, test_df, _, fold_name in folds:
        val_auc = val_m.get("auc", float("nan"))
        for _, row in test_df.iterrows():
            pid = row["patient_id"]
            if patient_data[pid]["true_label"] is None:
                patient_data[pid]["true_label"] = int(row["true_label"])
                patient_data[pid]["true_class"] = str(row.get("true_class", ""))
            patient_data[pid]["model_probs"][fold_name] = np.array(
                [row[f"prob_{c}"] for c in range(n_classes)], dtype=float
            )
            # Weight by validation AUC (not test AUC — avoid leakage)
            patient_data[pid]["model_auc"][fold_name] = val_auc

    print(f"  Unique test patients: {len(patient_data)}")

    # ── 4. Run ensemble strategies ────────────────────────────────────────────
    pred_df, ensemble_metrics = run_ensemble(
        patient_data, class_names, args.strategies,
        youden_threshold=oof_youden_threshold,
    )

    # Confusion matrices — one per ensemble strategy
    for strategy in args.strategies:
        s_df = pred_df[pred_df["strategy"] == strategy]
        save_confusion_matrix(
            s_df["true_label"].tolist(),
            s_df["pred_label"].tolist(),
            class_names,
            f"Ensemble: {strategy}",
            out_dir / f"confusion_matrix_ensemble_{strategy}.png",
        )

    # ROC curve for mean_prob ensemble (one clean curve per class)
    if "mean_prob" in args.strategies:
        _mp_df = pred_df[pred_df["strategy"] == "mean_prob"]
        save_roc_curves(
            _mp_df["true_label"].values,
            _mp_df[_prob_cols].values,
            class_names,
            f"ROC — Test ensemble (mean_prob, {len(_mp_df)} patients)",
            out_dir / "roc_test_ensemble.png",
        )
        save_combined_figure(
            # Row a — Validation OOF
            all_val_df["true_label"].tolist(),
            all_val_df[_prob_cols].values.argmax(1).tolist(),
            all_val_df["true_label"].values,
            all_val_df[_prob_cols].values,
            # Row b — Test per-fold
            all_test_df["true_label"].tolist(),
            all_test_df[_prob_cols].values.argmax(1).tolist(),
            len(folds),
            _y_true_test,
            _fold_probs,
            # Row c — Ensemble (mean_prob)
            _mp_df["true_label"].tolist(),
            _mp_df["pred_label"].tolist(),
            _mp_df["true_label"].values,
            _mp_df[_prob_cols].values,
            class_names,
            out_dir / "combined_figure.jpg",
            font_size=9,
            dpi=300,
        )

    # Save ensemble predictions
    ens_pred_out = out_dir / "ensemble_test_predictions.csv"
    pred_df.to_csv(ens_pred_out, index=False)
    print(f"\n  Saved → {ens_pred_out}")

    # Save ensemble metrics
    ens_metrics_out = out_dir / "ensemble_test_metrics.json"
    with open(ens_metrics_out, "w") as f:
        json.dump(ensemble_metrics, f, indent=2)
    print(f"  Saved → {ens_metrics_out}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Metric':<15}  {'Val mean±std':<22}  {'Test mean±std (per-fold)':<22}  {'Ensemble (mean_prob)'}")
    print(f"  {'-'*15}  {'-'*22}  {'-'*22}  {'-'*20}")
    for key in SCALAR_METRICS:
        if key not in agg_val and key not in agg_test:
            continue
        v      = agg_val.get(key,  {})
        t      = agg_test.get(key, {})
        ens_val = ensemble_metrics.get("mean_prob", {}).get(key, float("nan"))
        v_str = f"{v['mean']:.4f} ± {v['std']:.4f}" if v else "—"
        t_str = f"{t['mean']:.4f} ± {t['std']:.4f}" if t else "—"
        print(f"  {key:<15}  {v_str:<22}  {t_str:<22}  {ens_val:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()