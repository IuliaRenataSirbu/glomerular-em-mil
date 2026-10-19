"""
test.py
─────────────────────────────────────────────────────────────────────────────
Evaluates a trained EM3 model on the held-out test set defined in folds.json
(the "test" key, populated by create_stratified_data_folds.py with the
camera-based holdout patients).

Loads the validation functions directly from training.py so any changes there
are automatically reflected here.

Results are saved to:
    <run_dir>/test_results/
        predictions_test.csv
        confusion_matrix_test.png
        metrics_test.json

    With --calibrate:
        calibration_weights.json          — fitted weights + val metrics before/after
        predictions_test_calibrated.csv
        confusion_matrix_test_calibrated.png
        metrics_test_calibrated.json

Usage:
    python test.py \\
        --run-dir   /path/to/run_dir          \\  # folder containing checkpoint
        --folds     /path/to/folds.json       \\
        [--checkpoint  model_best.pt]         \\  # filename inside run_dir (default: model_best.pt)
        [--config      /path/to/config.yml]   \\  # override config (default: uses run_dir/config.yml)
        [--out-dir     test_results]          \\  # subfolder name inside run_dir (default: test_results)

Example:
    python test.py \\
        --run-dir  /export/runs/dino_finetuned_fold0 \\
        --folds    /export/data/folds.json
"""

import argparse
import json
import sys
import yaml
from pathlib import Path

import numpy as np
import torch

# ── Make sure local modules are importable ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Parse config early so cfg.CLASSES is correct before dataset.py is imported.
# dataset.py builds CLASS_TO_IDX at import time from cfg.CLASSES, so the yml
# must be applied to cfg BEFORE the import happens. ───────────────────────────
def _apply_config_early():
    """Find and apply the yml config before any other local imports."""
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
            overrides = yaml.safe_load(f)
        for k, v in (overrides or {}).items():
            if hasattr(_cfg, k):
                setattr(_cfg, k, v)
        print(f"  Config applied early from: {config_path}")

_apply_config_early()

from config   import cfg
from dataset  import (build_dataframe_v2,
                      PatientMILDataset, get_transforms,
                      IDX_TO_CLASS)
from model    import build_model
from training import (validate_patient_level,
                      save_validation_results, compute_metrics)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(path: Path, model, device):
    """Loads model weights from a checkpoint file."""
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state", ckpt)   # handle both wrapped and raw state dicts
    model.load_state_dict(state)
    epoch = ckpt.get("epoch", -1)
    val_metrics = ckpt.get("val_metrics", {})
    print(f"  Loaded checkpoint: {path}")
    print(f"  Saved at epoch: {epoch}")
    if val_metrics:
        print(f"  Val metrics at save time:")
        for k, v in val_metrics.items():
            if k != "loss" and not isinstance(v, dict):
                print(f"    {k:<12}: {v:.4f}")
    return model, epoch


def build_test_df(full_df, test_patients: list):
    """Filter full_df to only the test set patients."""
    test_set   = set(test_patients)
    test_df    = full_df[full_df["patient_id"].isin(test_set)].reset_index(drop=True)

    found      = test_df["patient_id"].nunique()
    not_found  = test_set - set(test_df["patient_id"].unique())

    print(f"\n  Test patients in folds.json : {len(test_patients)}")
    print(f"  Test patients with data     : {found}")
    if not_found:
        print(f"  WARNING: {len(not_found)} test patients have no images in the dataframe:")
        for pid in sorted(not_found):
            print(f"    {pid}")

    # Per-class breakdown
    print(f"\n  Test set class distribution:")
    for idx, cls in IDX_TO_CLASS.items():
        n = test_df[test_df["label"] == idx]["patient_id"].nunique()
        if n > 0:
            print(f"    [{idx}] {cls}: {n} patients")

    return test_df


def calibrate_on_val(val_csv: Path, metric: str, num_classes: int, max_weight: float = 10.0):
    """
    Read predictions_best.csv (saved val predictions from the best epoch),
    fit per-class probability scaling weights that optimise `metric`, and
    return those weights together with val metrics before and after calibration.

    max_weight > 0 : weights are clipped to [1/max_weight, max_weight] during
                     optimisation to prevent degenerate solutions (e.g. always
                     predicting the majority class). Default: 10.0.
    max_weight == 0: no bounds — weights are unconstrained (use with caution on
                     small val sets).

    Returns: (weights [num_classes], metrics_before dict, metrics_after dict)
    """
    try:
        from scipy.optimize import differential_evolution
    except ImportError:
        raise ImportError("scipy is required for --calibrate: pip install scipy")

    import pandas as pd

    if not val_csv.exists():
        raise FileNotFoundError(
            f"Val predictions not found: {val_csv}\n"
            f"Ensure the run completed at least one epoch and saved predictions_best.csv."
        )

    df        = pd.read_csv(val_csv)
    prob_cols = [f"prob_{c}" for c in range(num_classes)]
    missing   = [c for c in prob_cols if c not in df.columns]
    if missing:
        raise ValueError(f"predictions_best.csv is missing expected columns: {missing}")

    val_probs  = df[prob_cols].values                    # [N, C]
    val_labels = df["true_label"].values.astype(int)     # [N]

    bounded   = max_weight > 0
    log_bound = np.log(max_weight) if bounded else None
    print(f"  Val predictions : {val_csv}")
    print(f"  Val patients    : {len(df)}")
    if bounded:
        print(f"  Weight bounds   : [1/{max_weight:.4g}, {max_weight:.4g}]")
    else:
        print(f"  Weight bounds   : none (unbounded)")

    metrics_before = compute_metrics(val_probs, val_labels, num_classes)

    def neg_metric(log_w):
        # Clip in log-space to enforce weight bounds; no-op when unbounded.
        effective = np.clip(log_w, -log_bound, log_bound) if bounded else log_w
        scaled = val_probs * np.exp(effective)
        preds  = scaled.argmax(1)
        if metric == "accuracy":
            return -(preds == val_labels).mean()
        elif metric == "bacc":
            from sklearn.metrics import balanced_accuracy_score
            return -balanced_accuracy_score(val_labels, preds)
        else:  # f1_macro
            from sklearn.metrics import f1_score
            return -f1_score(val_labels, preds, average="macro", zero_division=0)

    # differential_evolution maintains a population spread across the full search
    # space, so it explores multiple plateaus simultaneously rather than growing
    # from a single point. This avoids Nelder-Mead's failure mode where the
    # entire initial simplex sits on the flat [1,1,...,1] plateau and the
    # optimizer stalls without ever flipping a prediction.
    # polish=False: L-BFGS-B polishing is useless on a step-function objective.
    bounds      = [(-log_bound, log_bound)] * num_classes if bounded else [(-5.0, 5.0)] * num_classes
    result      = differential_evolution(
        neg_metric,
        bounds=bounds,
        maxiter=2_000,
        tol=1e-7,
        seed=42,
        polish=False,
    )
    log_w_final = np.clip(result.x, -log_bound, log_bound) if bounded else result.x
    weights     = np.exp(log_w_final)

    cal_probs     = val_probs * weights
    cal_probs     = cal_probs / cal_probs.sum(1, keepdims=True)
    metrics_after = compute_metrics(cal_probs, val_labels, num_classes)

    return weights, metrics_before, metrics_after


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def resolve_checkpoint(run_dir: Path, checkpoint_arg) -> Path:
    """
    Resolve the checkpoint path.
    - If --checkpoint is given explicitly, use it directly.
    - Otherwise, search run_dir for a single .pt file whose name contains 'best'.
      Raises an error if zero or multiple such files are found.
    """
    if checkpoint_arg is not None:
        path = Path(checkpoint_arg)
        if not path.is_absolute():
            path = run_dir / checkpoint_arg
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    # Auto-detect: find .pt files containing "best" in the name
    candidates = sorted(run_dir.glob("*.pt"))
    best_candidates = [p for p in candidates if "best" in p.name.lower()]

    if len(best_candidates) == 1:
        print(f"  Auto-detected checkpoint: {best_candidates[0].name}")
        return best_candidates[0]
    elif len(best_candidates) == 0:
        raise FileNotFoundError(
            f"No .pt file containing 'best' found in {run_dir}. "
            f"Available .pt files: {[p.name for p in candidates]}. "
            f"Specify one explicitly with --checkpoint."
        )
    else:
        raise ValueError(
            f"Multiple 'best' checkpoints found in {run_dir}: "
            f"{[p.name for p in best_candidates]}. "
            f"Specify one explicitly with --checkpoint."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained EM3 model on the held-out test set."
    )
    parser.add_argument(
        "--run-dir", required=True,
        help="Path to the run directory containing the checkpoint"
    )
    parser.add_argument(
        "--folds", default=None,
        help="Path to folds.json (must contain a 'test' key). "
             "If not given, read FOLDS_PATH from the run's config .yml."
    )
    parser.add_argument(
        "--split", default="test", choices=["test", "val"],
        help="Which split to evaluate: 'test' (default) or 'val' (for sanity-checking "
             "that scores match training). For 'val', --fold selects which fold to use."
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Fold index to use when --split=val. If not given, reads FOLD from config."
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Checkpoint filename or path. If not given, auto-detects the "
             "single .pt file whose name contains 'best' in --run-dir."
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to config .yml to override defaults. "
             "If not given, looks for a .yml file inside --run-dir."
    )
    parser.add_argument(
        "--out-dir", default="test_results",
        help="Output subfolder name inside --run-dir (default: test_results)"
    )
    parser.add_argument(
        "--calibrate", default=None,
        choices=["accuracy", "bacc", "f1_macro"],
        help="Fit per-class probability weights on predictions_best.csv (saved val "
             "predictions) and apply them to the test set before reporting metrics. "
             "Choose the metric to optimise: accuracy | bacc | f1_macro. "
             "Saves calibration_weights.json and calibrated predictions alongside "
             "the uncalibrated results."
    )
    parser.add_argument(
        "--calibrate-max-weight", type=float, default=10.0,
        dest="calibrate_max_weight",
        help="Upper bound for each per-class weight; lower bound is 1/max. "
             "Prevents degenerate solutions on small val sets. "
             "Set to 0 to disable bounds entirely (unbounded Nelder-Mead). "
             "Default: 10.0  →  weights in [0.1, 10]."
    )
    args = parser.parse_args()
    split   = args.split

    run_dir = Path(args.run_dir)
    out_dir = run_dir / args.out_dir  # split name appended below once split is known
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config ───────────────────────────────────────────────────────────
    # Priority: --config arg > any .yml inside run_dir > defaults in config.py
    config_path = args.config
    if config_path is None:
        yml_files = sorted(run_dir.glob("*.yml"))
        if yml_files:
            config_path = str(yml_files[0])
            print(f"  Auto-detected config: {config_path}")

    if config_path:
        from config import cfg as _cfg
        import yaml
        with open(config_path) as f:
            overrides = yaml.safe_load(f)
        for k, v in (overrides or {}).items():
            if hasattr(_cfg, k):
                setattr(_cfg, k, v)
        print(f"  Config loaded from: {config_path}")

    # ── Resolve checkpoint ────────────────────────────────────────────────────
    ckpt_path = resolve_checkpoint(run_dir, args.checkpoint)

    # ── Resolve folds path ────────────────────────────────────────────────────
    # Priority: --folds arg > FOLDS_PATH in config
    folds_path = args.folds
    if folds_path is None:
        folds_path = getattr(cfg, "FOLDS_PATH", None)
        if folds_path:
            print(f"  Folds path from config: {folds_path}")
        else:
            raise ValueError(
                "--folds not specified and FOLDS_PATH not set in the config. "
                "Pass --folds /path/to/folds.json explicitly."
            )
    folds_path = Path(folds_path)
    if not folds_path.exists():
        raise FileNotFoundError(f"folds.json not found: {folds_path}")

    device = torch.device(cfg.DEVICE)

    print(f"\n{'='*65}")
    print(f"  EM3 — {split.capitalize()} Set Evaluation")
    print(f"  Run dir    : {run_dir}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Folds file : {folds_path}")
    print(f"  Output dir : {out_dir}")
    print(f"  Device     : {device}")
    print(f"  MIL mode   : {cfg.MIL_MODE}")
    print(f"  Classes    : {cfg.NUM_CLASSES}")
    print(f"{'='*65}\n")

    # ── Load folds.json — get patients for the requested split ──────────────
    with open(folds_path) as f:
        fold_data = json.load(f)

    if split == "test":
        if "test" not in fold_data:
            raise ValueError(
                "'test' key not found in folds.json. "
                "Re-run create_stratified_data_folds.py with --test-cameras to generate it."
            )
        eval_patients = fold_data["test"]
        if not eval_patients:
            raise ValueError(
                "folds.json['test'] is empty — no test patients defined. "
                "Pass --test-cameras to create_stratified_data_folds.py."
            )
        out_dir = run_dir / args.out_dir  # keep default subfolder name for test
    else:
        # val split — use the fold saved in the checkpoint or overridden via --fold
        fold_idx = args.fold
        if fold_idx is None:
            # try checkpoint cfg first, then current cfg
            ckpt_fold = (raw_ckpt.get("cfg", {}).get("FOLD")
                         if "raw_ckpt" in dir() else None)
            fold_idx = ckpt_fold if ckpt_fold is not None else getattr(cfg, "FOLD", 0)
        fold_key = str(fold_idx)
        if fold_key not in fold_data.get("folds", {}):
            raise ValueError(f"Fold '{fold_key}' not found in folds.json.")
        eval_patients = fold_data["folds"][fold_key]["val"]
        out_dir = run_dir / f"{args.out_dir}_val_fold{fold_idx}"
        print(f"  Sanity-check mode: evaluating val split, fold {fold_idx}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  {split.capitalize()} patients: {len(eval_patients)}")

    # ── Build dataframe ───────────────────────────────────────────────────────
    print(f"\nUsing pre-built JSON index: {cfg.PROCESSED_JSON_PATH}")
    full_df = build_dataframe_v2(cfg.PROCESSED_JSON_PATH)

    test_df = build_test_df(full_df, eval_patients)

    if len(test_df) == 0:
        raise ValueError("No test images found — check that patient IDs in folds.json "
                         "match the 'patient_id' column in the dataframe.")

    # ── Build test dataset ────────────────────────────────────────────────────
    transform  = get_transforms(train=False)
    test_ds    = PatientMILDataset(
        test_df,
        transform=transform,
        max_imgs=None,                        # use ALL images at test time
    )
    print(f"\n  Test dataset: {len(test_ds)} patients  |  {len(test_df)} images")

    # ── Build model and load checkpoint ───────────────────────────────────────
    # Read num_classes and tab_dim from the checkpoint's state dict to avoid
    # architecture mismatches when configs differ from the saved model.
    raw_ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_cfg  = raw_ckpt.get("cfg", {})
    ckpt_num_classes = ckpt_cfg.get("NUM_CLASSES", None)
    if ckpt_num_classes is not None and ckpt_num_classes != cfg.NUM_CLASSES:
        print(f"\n  WARNING: checkpoint has NUM_CLASSES={ckpt_num_classes}, "
              f"config has NUM_CLASSES={cfg.NUM_CLASSES}. "
              f"Building model with checkpoint's NUM_CLASSES={ckpt_num_classes}.")
        cfg.NUM_CLASSES = ckpt_num_classes
    del raw_ckpt  # free memory before building model

    # Load tabular features if training used mid-level fusion
    tab_features = None
    tab_dim      = 0
    if getattr(cfg, "TAB_FUSION", False):
        from dataset import load_tabular_features
        tab_features, _, tab_dim = load_tabular_features(
            csv_path    = cfg.TAB_FEATURES_CSV,
            id_col      = cfg.TAB_ID_COL,
            marker_cols = cfg.TAB_MARKER_COLS,
        )
        missing = [str(pid) for pid in test_df["patient_id"].unique()
                   if str(pid) not in tab_features]
        if missing:
            raise ValueError(
                f"TAB: {len(missing)} test patients not found in TAB_FEATURES_CSV: {missing[:5]}"
            )
        print(f"  TAB test: all {test_df['patient_id'].nunique()} patients found in tab_features.")
        test_ds.tab_features = tab_features  # inject into already-built dataset

    model = build_model(tab_dim=tab_dim).to(device)
    model, ckpt_epoch = load_checkpoint(ckpt_path, model, device)
    model.eval()

    # ── Recompute val metrics with current code; extract Youden threshold ─────
    import pandas as pd
    val_youden_threshold = None
    val_pred_csv = run_dir / "predictions_best.csv"
    if val_pred_csv.exists():
        val_df     = pd.read_csv(val_pred_csv)
        prob_cols  = sorted([c for c in val_df.columns if c.startswith("prob_")])
        val_probs  = val_df[prob_cols].values
        val_labels = val_df["true_label"].values
        val_m      = compute_metrics(val_probs.tolist(), val_labels.tolist(), cfg.NUM_CLASSES)
        val_youden_threshold = val_m.get("youden_threshold")
        val_metrics_out = out_dir / "metrics_val.json"
        with open(val_metrics_out, "w") as f:
            json.dump(val_m, f, indent=2)
        print(f"  Val metrics (recomputed) → {val_metrics_out}")
        if val_youden_threshold is not None:
            print(f"  Val Youden threshold     : {val_youden_threshold:.4f}")

    # ── Run inference ─────────────────────────────────────────────────────────
    print(f"\nRunning test set inference ...")
    metrics, (preds, labels, pids) = validate_patient_level(
        model, test_ds, device,
        epoch=ckpt_epoch,
        total_epochs=ckpt_epoch,
        return_preds=True,
        youden_threshold=val_youden_threshold,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  TEST SET RESULTS")
    print(f"{'='*65}")
    print(f"  Patients evaluated : {len(pids)}")
    for k, v in metrics.items():
        if k != "loss" and not isinstance(v, dict):
            print(f"  {k:<12} : {v:.4f}")
    print(f"{'='*65}\n")

    # ── Save predictions, confusion matrix, metrics JSON ─────────────────────
    save_validation_results(
        preds=preds,
        labels=labels,
        pids=pids,
        save_dir=out_dir,
        epoch=ckpt_epoch,
        tag=split,
        youden_threshold=val_youden_threshold,
    )

    # ── Calibration ───────────────────────────────────────────────────────────
    if args.calibrate:
        val_csv = run_dir / "predictions_best.csv"

        print(f"\n{'='*65}")
        print(f"  CALIBRATION  (optimising: {args.calibrate})")
        print(f"{'='*65}")

        weights, val_m_before, val_m_after = calibrate_on_val(
            val_csv, args.calibrate, cfg.NUM_CLASSES,
            max_weight=args.calibrate_max_weight,
        )

        w_str = "  ".join(f"{w:.3f}" for w in weights)
        print(f"  Weights : [{w_str}]")

        metric_keys = [k for k, v in val_m_before.items()
                       if k != "loss" and not isinstance(v, dict)]
        print(f"\n  {'metric':<14}  {'val before':>10}  {'val after':>9}")
        print(f"  {'-'*38}")
        for k in metric_keys:
            print(f"  {k:<14}  {val_m_before[k]:>10.4f}  {val_m_after[k]:>9.4f}")

        # Apply weights to test predictions
        test_probs  = np.array(preds)                          # [N, C]
        cal_probs   = test_probs * weights
        cal_probs   = cal_probs / cal_probs.sum(1, keepdims=True)
        cal_metrics = compute_metrics(cal_probs, np.array(labels), cfg.NUM_CLASSES)

        print(f"\n  {'metric':<14}  {'test uncal':>10}  {'test cal':>8}")
        print(f"  {'-'*36}")
        for k in metric_keys:
            delta = cal_metrics[k] - metrics[k]
            sign  = "+" if delta >= 0 else ""
            print(f"  {k:<14}  {metrics[k]:>10.4f}  {cal_metrics[k]:>8.4f}  ({sign}{delta:.4f})")
        print(f"{'='*65}\n")

        # Save calibration weights + val metrics
        cal_info = {
            "metric":             args.calibrate,
            "weights":            weights.tolist(),
            "val_metrics_before": val_m_before,
            "val_metrics_after":  val_m_after,
        }
        cal_path = out_dir / "calibration_weights.json"
        with open(cal_path, "w") as f:
            json.dump(cal_info, f, indent=2)
        print(f"  Calibration weights → {cal_path}")

        for tag, m in [("before", val_m_before), ("after", val_m_after)]:
            p = out_dir / f"metrics_val_{tag}.json"
            with open(p, "w") as f:
                json.dump(m, f, indent=2)
            print(f"  Val metrics ({tag})  → {p}")

        # Save calibrated predictions, confusion matrix, metrics JSON
        save_validation_results(
            preds=cal_probs,
            labels=labels,
            pids=pids,
            save_dir=out_dir,
            epoch=ckpt_epoch,
            tag=f"{split}_calibrated",
        )

    print(f"\nAll test results saved to: {out_dir}/")


if __name__ == "__main__":
    main()