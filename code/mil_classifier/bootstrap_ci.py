"""
bootstrap_ci.py
───────────────────────────────────────────────────────────────────────────────
Compute bootstrap 95% confidence intervals for all scalar metrics.

Takes the same arguments as ensemble_cv_evaluation.py and reads the same
directory structure, plus the ensemble_test_predictions.csv produced by that
script.

Outputs (written to --out):
  ci_val_oof.json        — CI on OOF validation (each patient appears once)
  ci_test_agg.json       — averaged CI bounds across per-fold test CIs (approximation)
  ci_test_per_fold.json  — per-fold test CIs (for reference)
  ci_ensemble.json       — CI per ensemble strategy

Usage:
    python bootstrap_ci.py \\
        --models /results/fold0 /results/fold1 /results/fold2 /results/fold3 /results/fold4 \\
        --out    /results/ensemble \\
        --n-bootstrap 1000
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from training import compute_metrics
from ensemble_cv_evaluation import (
    load_fold, load_classes_from_yml,
    STRATEGIES, DEFAULT_LABELS,
)


# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_metrics(probs: np.ndarray, labels: np.ndarray, n_classes: int,
                      n_bootstrap: int = 1000, ci: float = 0.95,
                      class_names: list = None,
                      youden_threshold: float = None) -> dict:
    """
    Bootstrap CI for all scalar metrics returned by compute_metrics.
    Each resample is passed directly to compute_metrics; per-metric NaN
    values (e.g. AUC when a class is absent from the resample) are silently
    dropped so CIs are computed from only the valid samples.

    Returns {metric: {point, ci_low, ci_high, n_valid}} where n_valid is
    the number of bootstrap samples that produced a finite value for that
    metric (AUC requires all classes to be present, so its n_valid may be
    lower than n_bootstrap on small test sets with many classes).
    """
    alpha = (1 - ci) / 2
    n = len(labels)
    rng = np.random.default_rng(42)
    bucket = defaultdict(list)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        b_probs, b_labels = probs[idx], labels[idx]
        m = compute_metrics(b_probs.tolist(), b_labels.tolist(), n_classes,
                            class_names=class_names,
                            youden_threshold=youden_threshold)
        for k, v in m.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                fv = float(v)
                if not np.isnan(fv):
                    bucket[k].append(fv)

    point_m = compute_metrics(probs.tolist(), labels.tolist(), n_classes,
                              class_names=class_names,
                              youden_threshold=youden_threshold)

    out = {}
    for k, vals in bucket.items():
        arr = np.array(vals)
        out[k] = {
            "point":   float(point_m.get(k, float("nan"))),
            "ci_low":  float(np.percentile(arr, 100 * alpha)),
            "ci_high": float(np.percentile(arr, 100 * (1 - alpha))),
            "n_valid": len(vals),
        }
    return out


def probs_labels_from_df(df: pd.DataFrame, n_classes: int):
    prob_cols = [f"prob_{c}" for c in range(n_classes)]
    probs  = df[prob_cols].values.astype(float)
    labels = df["true_label"].values.astype(int)
    return probs, labels


def average_fold_cis(fold_cis: list) -> dict:
    """Average ci_low and ci_high bounds across folds (rough approximation)."""
    all_keys = set(k for fc in fold_cis for k in fc)
    out = {}
    for k in all_keys:
        entries = [fc[k] for fc in fold_cis if k in fc]
        out[k] = {
            "mean_point": float(np.mean([e["point"]   for e in entries])),
            "ci_low":     float(np.mean([e["ci_low"]  for e in entries])),
            "ci_high":    float(np.mean([e["ci_high"] for e in entries])),
            "n_folds":    len(entries),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap confidence intervals for EM3 MIL ensemble evaluation."
    )
    parser.add_argument("--models", nargs="+", required=True,
                        help="Paths to per-fold model directories (same as ensemble script)")
    parser.add_argument("--out", required=True,
                        help="Output directory (same as ensemble script --out)")
    parser.add_argument("--test-subdir", default="test_results",
                        help="Subfolder inside each model dir with test results "
                             "(default: test_results)")
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS,
                        help="Class names in label index order")
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES, choices=STRATEGIES,
                        help="Ensemble strategies to compute CI for (default: all)")
    parser.add_argument("--calibrated", action="store_true",
                        help="Use calibrated test predictions")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Number of bootstrap resamples (default: 1000)")
    parser.add_argument("--ci", type=float, default=0.95,
                        help="Confidence level (default: 0.95)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_classes_from_yml(
        [Path(m) for m in args.models], fallback=args.labels
    )
    n_classes = len(class_names)
    B, CI = args.n_bootstrap, args.ci
    print(f"\nBootstrap CI  n={B}  level={CI:.0%}  classes={class_names}")

    # ── Load folds ─────────────────────────────────────────────────────────────
    print(f"\nLoading {len(args.models)} fold(s) ...")
    folds = []
    for m in args.models:
        fold = load_fold(Path(m), args.test_subdir, n_classes, calibrated=args.calibrated)
        folds.append(fold)
    # folds[i] = (val_df, val_metrics, test_df, test_metrics, fold_name)

    # ── 1. OOF VALIDATION CI ──────────────────────────────────────────────────
    print("\n[1/3] OOF validation CI ...")
    all_val_df = pd.concat([f[0] for f in folds], ignore_index=True)
    val_probs, val_labels = probs_labels_from_df(all_val_df, n_classes)
    ci_val = bootstrap_metrics(val_probs, val_labels, n_classes,
                               n_bootstrap=B, ci=CI, class_names=class_names)
    val_out = out_dir / "ci_val_oof.json"
    with open(val_out, "w") as f:
        json.dump(ci_val, f, indent=2)
    print(f"  Saved → {val_out}")
    _print_ci(ci_val, ["acc", "top2_acc", "bacc", "auc", "f1_macro"])

    # ── 2. PER-FOLD TEST CI → AVERAGED ────────────────────────────────────────
    print("\n[2/3] Per-fold test CI (then averaged) ...")
    fold_cis = []
    for val_df, val_m, test_df, test_m, fold_name in folds:
        print(f"  {fold_name} ...", end=" ", flush=True)
        test_probs, test_labels = probs_labels_from_df(test_df, n_classes)
        fold_ci = bootstrap_metrics(test_probs, test_labels, n_classes,
                                    n_bootstrap=B, ci=CI, class_names=class_names)
        fold_cis.append(fold_ci)
        print(f"acc {fold_ci['acc']['point']:.3f} "
              f"[{fold_ci['acc']['ci_low']:.3f}–{fold_ci['acc']['ci_high']:.3f}]")

    per_fold_out = out_dir / "ci_test_per_fold.json"
    with open(per_fold_out, "w") as f:
        json.dump({folds[i][4]: fc for i, fc in enumerate(fold_cis)}, f, indent=2)
    print(f"  Saved → {per_fold_out}")

    agg_ci = average_fold_cis(fold_cis)
    agg_out = out_dir / "ci_test_agg.json"
    with open(agg_out, "w") as f:
        json.dump(agg_ci, f, indent=2)
    print(f"  Saved → {agg_out}")
    _print_ci(agg_ci, ["acc", "top2_acc", "bacc", "auc", "f1_macro"],
              key_point="mean_point")

    # ── 3. ENSEMBLE CI ─────────────────────────────────────────────────────────
    ens_pred_path = out_dir / "ensemble_test_predictions.csv"
    if not ens_pred_path.exists():
        print(f"\n[3/3] Skipping ensemble CI — {ens_pred_path.name} not found. "
              f"Run ensemble_cv_evaluation.py first.")
        return

    print(f"\n[3/3] Ensemble strategy CI ...")
    ens_df = pd.read_csv(ens_pred_path)
    ens_ci = {}
    for strategy in args.strategies:
        s_df = ens_df[ens_df["strategy"] == strategy]
        if s_df.empty:
            print(f"  {strategy}: not found in predictions CSV, skipping")
            continue
        s_probs, s_labels = probs_labels_from_df(s_df, n_classes)

        # ── Diagnostic: sanity-check the full data before bootstrapping ──────
        unique_classes = np.unique(s_labels).tolist()
        print(f"  {strategy}  N={len(s_labels)}  probs={s_probs.shape}"
              f"  n_classes_cfg={n_classes}  classes_in_labels={unique_classes}")
        if len(unique_classes) != n_classes:
            print(f"    WARNING: {n_classes - len(unique_classes)} class(es) absent "
                  f"from this split — AUC will be computed over present classes only")
        # ─────────────────────────────────────────────────────────────────────

        print(f"  {strategy} ...", end=" ", flush=True)
        s_ci = bootstrap_metrics(s_probs, s_labels, n_classes,
                                 n_bootstrap=B, ci=CI, class_names=class_names)
        ens_ci[strategy] = s_ci
        auc_str = (f"  auc {s_ci['auc']['point']:.3f} "
                   f"[{s_ci['auc']['ci_low']:.3f}–{s_ci['auc']['ci_high']:.3f}]"
                   f" (n_valid={s_ci['auc']['n_valid']}/{B})"
                   if "auc" in s_ci else "  auc n/a (0 valid bootstrap samples)")
        print(f"acc {s_ci['acc']['point']:.3f} "
              f"[{s_ci['acc']['ci_low']:.3f}–{s_ci['acc']['ci_high']:.3f}]"
              f"{auc_str}")

    ens_out = out_dir / "ci_ensemble.json"
    with open(ens_out, "w") as f:
        json.dump(ens_ci, f, indent=2)
    print(f"  Saved → {ens_out}")

    print("\nDone.")


def _print_ci(ci: dict, keys: list, key_point: str = "point") -> None:
    for k in keys:
        if k in ci:
            e = ci[k]
            print(f"    {k:<16}  {e[key_point]:.3f}  "
                  f"[{e['ci_low']:.3f} – {e['ci_high']:.3f}]")


if __name__ == "__main__":
    main()
