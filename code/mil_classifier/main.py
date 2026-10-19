import os
from pathlib import Path
# main.py
# ─────────────────────────────────────────────────────────────────────────────
# Entry point. Run with:  python main.py
#
# Reads cfg.TRAINING_MODE from config.py and dispatches to one of two
# training strategies:
#   "image"   → run_image_level()   (image-level classification)
#   "patient" → run_patient_level() (patient-level MIL)
#
# All hyperparameters and paths are controlled from config.py only.
# ─────────────────────────────────────────────────────────────────────────────

import contextlib
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as _DDP
from tqdm import tqdm                                  # used for epoch-level progress bar

from config   import cfg, _parse_config_arg
from dataset  import build_dataframe_v2, patient_level_split, build_dataloaders, CLASS_TO_IDX
from model    import build_model                          # factory: returns ImageClassifier or MILClassifier based on cfg.TRAINING_MODE
from training import (train_image_level, train_patient_level,
                      validate_image_level, validate_patient_level,
                      compute_metrics, save_validation_results,
                      build_feature_cache)


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_seeds(seed: int):
    """Fix all random seeds so every run with the same config is identical."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)                                       # safe even without GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True                              # deterministic cuDNN ops
    torch.backends.cudnn.benchmark     = False                             # no auto-tuning (adds randomness)


# ─────────────────────────────────────────────────────────────────────────────
# DISTRIBUTED TRAINING HELPERS
# Single-GPU: python main.py --config ...
# Multi-GPU:  torchrun --nproc_per_node=<N> main.py --config ...
# ─────────────────────────────────────────────────────────────────────────────

def _init_distributed():
    """
    Auto-detects torchrun via the LOCAL_RANK env var and initialises NCCL.
    Returns (is_ddp, rank, world_size). Single-GPU runs return (False, 0, 1).
    """
    rank = int(os.environ.get("LOCAL_RANK", -1))
    if rank == -1:
        return False, 0, 1
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
    return True, rank, dist.get_world_size()


def _rank_indices(n: int, rank: int, world_size: int) -> list:
    """Interleaved split: rank r gets indices [r, r+world_size, r+2*world_size, ...]."""
    return list(range(rank, n, world_size))


def _gather_val_preds(preds, labels, pids, world_size):
    """All-gather validation predictions from every rank. Returns merged lists."""
    gathered = [None] * world_size
    dist.all_gather_object(gathered, (preds, labels, pids))
    all_preds  = [p for r_preds,  _,        _      in gathered for p in r_preds]
    all_labels = [l for _,        r_labels,  _      in gathered for l in r_labels]
    all_pids   = [p for _,        _,        r_pids  in gathered for p in r_pids]
    return all_preds, all_labels, all_pids


# ─────────────────────────────────────────────────────────────────────────────
# LOGGER CLASSES
# Switch by setting cfg.LOGGER in config.py — no other file changes needed.
# Each logger is only imported when actually used, so unused packages are
# never required to be installed.
# ─────────────────────────────────────────────────────────────────────────────

class WandbLogger:
    def __init__(self, cfg, optimizer, run_name: str):
        import wandb                                       # only imported when cfg.LOGGER = "wandb"
        wandb.init(
            project = cfg.PROJECT_NAME,                    # set in config.py under LOGGER section
            name    = run_name,
            config  = {
                "training_mode": cfg.TRAINING_MODE,        # log the approach being used
                "fold": cfg.FOLD, "n_folds": cfg.N_FOLDS,
                "epochs": cfg.EPOCHS, "batch_size": cfg.BATCH_SIZE,
                "lr": cfg.LR, "backbone_lr": getattr(cfg, "BACKBONE_LR", None),
                "weight_decay": cfg.WEIGHT_DECAY,
                "mil_mode": cfg.MIL_MODE, "img_size": cfg.IMG_SIZE,
                "pretrained": cfg.PRETRAINED, "amp": cfg.AMP,

                "classes": cfg.CLASSES,
            },
        )
        self._wandb = wandb
        print(f"  [Logger] wandb - view at: {wandb.run.url}\n")

    def log(self, train_metrics, val_metrics, optimizer, epoch, prefix=""):
        d = {
            f"{prefix}train/loss":  train_metrics.get("loss", 0.0),       # loss (0 if not applicable)
            f"{prefix}train/auc":   train_metrics["auc"],
            f"{prefix}train/bacc":  train_metrics["bacc"],
            f"{prefix}val/auc":     val_metrics["auc"],
            f"{prefix}val/bacc":    val_metrics["bacc"],
            f"{prefix}lr/head":     optimizer.param_groups[0]["lr"],
            **({f"{prefix}lr/backbone": optimizer.param_groups[-1]["lr"]}
               if len(optimizer.param_groups) > 1 else {}),
        }
        self._wandb.log(d)

    def finish(self):
        self._wandb.finish()                               # mark run complete and sync to dashboard


class TensorBoardLogger:
    def __init__(self, cfg, optimizer, run_name: str):
        from torch.utils.tensorboard import SummaryWriter  # only imported when cfg.LOGGER = "tensorboard"
        log_dir = cfg.SAVE_DIR / "tensorboard" / cfg.PROJECT_NAME / run_name  # project name groups runs together
        self._writer = SummaryWriter(log_dir=str(log_dir))
        print(f"  [Logger] TensorBoard - run: tensorboard --logdir {cfg.SAVE_DIR / 'tensorboard' / cfg.PROJECT_NAME}\n")

    def log(self, train_metrics, val_metrics, optimizer, epoch, prefix=""):
        self._writer.add_scalar(f"{prefix}Loss/train",  train_metrics.get("loss", 0.0), epoch)
        self._writer.add_scalar(f"{prefix}AUC/train",   train_metrics["auc"],            epoch)
        self._writer.add_scalar(f"{prefix}bAcc/train",  train_metrics["bacc"],           epoch)
        self._writer.add_scalar(f"{prefix}AUC/val",     val_metrics["auc"],              epoch)
        self._writer.add_scalar(f"{prefix}bAcc/val",    val_metrics["bacc"],             epoch)
        self._writer.add_scalar(f"{prefix}LR/head",     optimizer.param_groups[0]["lr"], epoch)
        if len(optimizer.param_groups) > 1:
            self._writer.add_scalar(f"{prefix}LR/backbone", optimizer.param_groups[-1]["lr"], epoch)

    def finish(self):
        self._writer.close()                               # flush and close log file


class NoOpLogger:
    def __init__(self, cfg, optimizer, run_name: str):
        print("  [Logger] none - logging disabled\n")

    def log(self, *args, **kwargs):
        pass                                               # silently discard all log calls

    def finish(self):
        pass                                               # nothing to close


def build_logger(cfg, optimizer, run_name: str):
    """
    Returns the logger instance matching cfg.LOGGER.
    run_name is used to label the run in wandb / TensorBoard subfolder.
    """
    if cfg.LOGGER == "wandb":
        return WandbLogger(cfg, optimizer, run_name)
    elif cfg.LOGGER == "tensorboard":
        return TensorBoardLogger(cfg, optimizer, run_name)
    elif cfg.LOGGER == "none":
        return NoOpLogger(cfg, optimizer, run_name)
    else:
        raise ValueError(f"Unknown LOGGER '{cfg.LOGGER}'. Choose: 'wandb', 'tensorboard', 'none'.")


# ─────────────────────────────────────────────────────────────────────────────
# SHARED SETUP
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification (Lin et al. 2017).
    Reduces the relative loss for well-classified examples, focusing
    training on hard misclassified examples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma  : focusing parameter (0 = standard CE, 2 = typical focal)
        weight : per-class weights tensor (same as CrossEntropyLoss weight)
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard CE loss per sample
        ce_loss = nn.functional.cross_entropy(
            logits, targets, weight=self.weight, reduction="none"
        )
        # p_t = probability of the true class
        pt = torch.exp(-ce_loss)
        # Apply focal weighting
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def build_optimizer(model):
    """
    AdamW with differential learning rates.

    When BACKBONE_LR is not set in config (None), uses the same scaling as before
    BACKBONE_LR was introduced — this preserves backward compatibility with all
    existing configs that worked well before:
        ViT/DINOv2 backbone: LR * 0.01  (pretrained on EM data — fine-tune very gently)
        ResNet50 backbone:   LR * 0.1   (pretrained on ImageNet — fine-tune gently)

    Set BACKBONE_LR explicitly in a config to override (e.g. BACKBONE_LR: 3e-4 for
    uniform LRs, or BACKBONE_LR: 3e-6 to match a specific multiplier exactly).

    For frozen ViT/DINOv2 (FREEZE_BACKBONE=True):
        backbone : excluded  — no gradients flow through frozen backbone
        head     : LR
        attention: LR
    """
    backbone_frozen = (getattr(cfg, "FREEZE_BACKBONE", False) and
                       getattr(cfg, "BACKBONE", "resnet50") == "dinov2_vits14")

    has_attention    = hasattr(model, "attention") and cfg.MIL_MODE == "attention"
    attention_params = list(model.attention.parameters()) if has_attention else []
    has_tab_proj     = hasattr(model, "tab_proj")
    tab_proj_params  = list(model.tab_proj.parameters()) if has_tab_proj else []

    backbone_lr = getattr(cfg, "BACKBONE_LR", None)
    if backbone_lr is None:
        # Restore the same auto-scaling that was hardcoded before BACKBONE_LR was added.
        backbone_name = getattr(cfg, "BACKBONE", "resnet50")
        if backbone_name == "dinov2_vits14":
            backbone_lr = cfg.LR * 0.01   # ViT/DINOv2: 100× smaller to protect EM pretrained features
        else:
            backbone_lr = cfg.LR * 0.1    # ResNet50: 10× smaller (ImageNet pretrained)

    if backbone_frozen:
        param_groups = [
            {"params": model.head.parameters(), "lr": cfg.LR},
        ]
        if attention_params:
            param_groups.append({"params": attention_params, "lr": cfg.LR})
        if tab_proj_params:
            param_groups.append({"params": tab_proj_params, "lr": cfg.LR})
        print(f"  Optimizer: AdamW — backbone FROZEN, head LR={cfg.LR:.0e}")
    else:
        # head first so param_groups[0] aligns with the logger's lr/head key
        param_groups = [
            {"params": model.head.parameters(),     "lr": cfg.LR},
            {"params": model.backbone.parameters(), "lr": backbone_lr},
        ]
        if attention_params:
            param_groups.append({"params": attention_params, "lr": cfg.LR})
        if tab_proj_params:
            param_groups.append({"params": tab_proj_params, "lr": cfg.LR})
        print(f"  Optimizer: AdamW — backbone LR={backbone_lr:.0e}, head LR={cfg.LR:.0e}")
    if tab_proj_params:
        print(f"  Optimizer: tab_proj included at LR={cfg.LR:.0e}")

    optimizer = optim.AdamW(param_groups, weight_decay=cfg.WEIGHT_DECAY)
    return optimizer


def print_corrupted_summary(train_ds, val_ds):
    """Prints distinct corrupted files found across train and val datasets."""
    n_tr = len(train_ds.corrupted_files)                                   # distinct corrupted files in training
    n_va = len(val_ds.corrupted_files)                                     # distinct corrupted files in validation
    if n_tr + n_va > 0:
        print(f"  WARNING: Distinct corrupted files - train: {n_tr}, val: {n_va}, total: {n_tr + n_va}")
        for p in sorted(train_ds.corrupted_files):
            print(f"    [train] {p}")
        for p in sorted(val_ds.corrupted_files):
            print(f"    [val]   {p}")


def save_checkpoint(model, optimizer, epoch, val_metrics, path):
    """Saves a complete checkpoint with model weights, optimizer state, and metadata."""
    raw_model = model.module if hasattr(model, 'module') else model        # unwrap DDP before saving
    torch.save(
        {
            "epoch":           epoch,
            "model_state":     raw_model.state_dict(), # always saved without DDP wrapper prefix
            "optimizer_state": optimizer.state_dict(), # optimizer momentum / state (for resuming training)
            "val_metrics":     val_metrics,            # validation metrics at this checkpoint
            "cfg":             cfg.__dict__,           # full config snapshot for reproducibility
            "class_to_idx":    CLASS_TO_IDX,           # label mapping saved alongside weights
        },
        path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING SUMMARY HELPER
# ─────────────────────────────────────────────────────────────────────────────

def print_best_summary(best_path, best_epoch, total_epochs, best_val_m, selection_metric="auc"):
    """
    Prints a clearly formatted summary box at the end of training showing
    the best checkpoint path, the metric used to select it, and all val metrics.
    """
    w = 65
    print(f"\n{'='*w}")
    print(f"  BEST CHECKPOINT SUMMARY")
    print(f"{'='*w}")
    print(f"  Path      : {best_path}")
    print(f"  Epoch     : {best_epoch} / {total_epochs}")
    print(f"  Selected by: val/{selection_metric} (higher = better)")
    print(f"{'-'*w}")
    print(f"  Validation metrics at best epoch:")
    for k, v in best_val_m.items():
        if k == "loss": continue
        if isinstance(v, (dict, list)): continue   # skip per_class_auc / per_class_acc
        print(f"    {k:<12} : {v:.4f}")
    print(f"{'='*w}\n")


# ─────────────────────────────────────────────────────────────────────────────
# APPROACH 1 — IMAGE LEVEL
# Training: image-by-image  |  Validation: image-by-image
# ─────────────────────────────────────────────────────────────────────────────

def run_image_level(model, train_loader, val_loader, train_ds, val_img_ds,
                    optimizer, criterion, scaler, scheduler, device,
                    run_name: str = "run"):  # Approach 1: image-level train + val
    """
    Approach 1: full pipeline at image level.
    Both training and validation treat each image as an independent sample.
    The MIL aggregation module is never used.
    Best checkpoint is selected based on cfg.CHECKPOINT_METRIC (default: auc).
    """
    ckpt_metric = getattr(cfg, "CHECKPOINT_METRIC", "auc")
    _valid      = {"auc", "acc", "bacc", "f1_macro"}
    if ckpt_metric not in _valid:
        raise ValueError(f"CHECKPOINT_METRIC='{ckpt_metric}' invalid. Choose: {sorted(_valid)}")

    best_score = 0.0
    best_epoch = 0
    best_val_m = {}
    best_path  = cfg.SAVE_DIR / f"image_best_fold{cfg.FOLD}.pt"
    logger     = build_logger(cfg, optimizer, run_name=f"{run_name}_fold{cfg.FOLD}")

    print(f"\n  [Approach 1 - Image Level]  {cfg.EPOCHS} epochs")

    for epoch in range(cfg.EPOCHS):

        train_m = train_image_level(
            model, train_loader, optimizer, criterion, scaler, device,
            epoch=epoch, total_epochs=cfg.EPOCHS,
        )
        val_m, val_preds_data = validate_image_level(
            model, val_loader, device,
            epoch=epoch, total_epochs=cfg.EPOCHS,
            return_preds=True,
        )
        scheduler.step()

        print(
            f"  Epoch {epoch+1:03d}/{cfg.EPOCHS} | "
            f"Train  loss: {train_m['loss']:.4f}  auc: {train_m['auc']:.4f}  bacc: {train_m['bacc']:.4f}  "
            f"acc: {train_m['acc']:.4f}  f1m: {train_m['f1_macro']:.4f}  ||  "
            f"Val    auc: {val_m['auc']:.4f}  bacc: {val_m['bacc']:.4f}  "
            f"acc: {val_m['acc']:.4f}  f1m: {val_m['f1_macro']:.4f}"
        )

        logger.log(train_m, val_m, optimizer, epoch)                      # log to wandb / TensorBoard
        print_corrupted_summary(train_ds, val_img_ds)                     # report any corrupted files

        if val_m[ckpt_metric] > best_score:                               # save checkpoint if validation improved
            best_score = val_m[ckpt_metric]
            best_epoch = epoch + 1
            best_val_m = val_m
            save_checkpoint(model, optimizer, epoch, val_m, best_path)
            save_validation_results(*val_preds_data, save_dir=cfg.SAVE_DIR,
                                    epoch=best_epoch, tag="best")
            print(f"  New best - Epoch {best_epoch:03d}/{cfg.EPOCHS}  Val {ckpt_metric} = {best_score:.4f}  -> {best_path}")

    last_path = cfg.SAVE_DIR / f"image_last_fold{cfg.FOLD}.pt"
    save_checkpoint(model, optimizer, cfg.EPOCHS - 1, val_m, last_path)
    print(f"  Last checkpoint saved → {last_path}")
    logger.finish()
    print_best_summary(best_path, best_epoch, cfg.EPOCHS, best_val_m, selection_metric=ckpt_metric)
    return best_path


# ─────────────────────────────────────────────────────────────────────────────
# APPROACH 2 — PATIENT LEVEL (FULL MIL)
# Training: patient-by-patient (full bag)  |  Validation: patient-by-patient
# ─────────────────────────────────────────────────────────────────────────────

def run_patient_level(model, train_patient_ds, val_patient_ds,
                      optimizer, criterion, scaler, scheduler, device,
                      run_name: str = "run",
                      rank: int = 0, world_size: int = 1, is_main: bool = True):
    """
    Approach 2: full pipeline at patient level.
    Training feeds full image bags through the model; gradients flow through
    backbone + attention + head. Validation is also at patient level.
    Best checkpoint is selected based on cfg.CHECKPOINT_METRIC (default: auc).

    With DDP (world_size > 1): patients are split across ranks for both
    training and validation; val predictions are gathered before metric
    computation; logging and checkpointing happen on rank 0 only.
    Train metrics are per-rank (approximate); val metrics are exact (gathered).
    """
    is_ddp = world_size > 1

    ckpt_metric = getattr(cfg, "CHECKPOINT_METRIC", "auc")
    _valid      = {"auc", "acc", "bacc", "f1_macro"}
    if ckpt_metric not in _valid:
        raise ValueError(f"CHECKPOINT_METRIC='{ckpt_metric}' invalid. Choose: {sorted(_valid)}")

    best_score = 0.0
    best_epoch = 0
    best_val_m = {}
    best_path  = cfg.SAVE_DIR / f"patient_best_fold{cfg.FOLD}.pt"
    logger     = (build_logger(cfg, optimizer, run_name=f"{run_name}_fold{cfg.FOLD}")
                  if is_main else NoOpLogger(cfg, optimizer, ""))

    if is_main:
        suffix = f"  [{world_size} GPUs, DDP]" if is_ddp else ""
        print(f"\n  [Approach 2 - Patient Level]  {cfg.EPOCHS} epochs{suffix}")

    train_indices = _rank_indices(len(train_patient_ds), rank, world_size)
    val_indices   = _rank_indices(len(val_patient_ds),   rank, world_size)

    for epoch in range(cfg.EPOCHS):

        train_m = train_patient_level(
            model, train_patient_ds, optimizer, criterion, scaler, device,
            epoch=epoch, total_epochs=cfg.EPOCHS, indices=train_indices,
        )
        val_m, val_preds_data = validate_patient_level(
            model, val_patient_ds, device,
            epoch=epoch, total_epochs=cfg.EPOCHS,
            return_preds=True, indices=val_indices,
        )
        scheduler.step()

        # Gather val predictions from all ranks, recompute metrics on the full set
        if is_ddp:
            val_preds_data = _gather_val_preds(*val_preds_data, world_size)
            val_m = compute_metrics(val_preds_data[0], val_preds_data[1], cfg.NUM_CLASSES)

        if is_main:
            print(
                f"  Epoch {epoch+1:03d}/{cfg.EPOCHS} | "
                f"Train  loss: {train_m['loss']:.4f}  auc: {train_m['auc']:.4f}  bacc: {train_m['bacc']:.4f}  "
                f"acc: {train_m['acc']:.4f}  f1m: {train_m['f1_macro']:.4f}  ||  "
                f"Val    auc: {val_m['auc']:.4f}  bacc: {val_m['bacc']:.4f}  "
                f"acc: {val_m['acc']:.4f}  f1m: {val_m['f1_macro']:.4f}"
            )

        logger.log(train_m, val_m, optimizer, epoch)
        if is_main:
            print_corrupted_summary(train_patient_ds, val_patient_ds)

        if is_main and val_m[ckpt_metric] > best_score:
            best_score = val_m[ckpt_metric]
            best_epoch = epoch + 1
            best_val_m = val_m
            save_checkpoint(model, optimizer, epoch, val_m, best_path)
            save_validation_results(*val_preds_data, save_dir=cfg.SAVE_DIR,
                                    epoch=best_epoch, tag="best")
            print(f"  New best - Epoch {best_epoch:03d}/{cfg.EPOCHS}  Val {ckpt_metric} = {best_score:.4f}  -> {best_path}")

    if is_main:
        last_path = cfg.SAVE_DIR / f"patient_last_fold{cfg.FOLD}.pt"
        save_checkpoint(model, optimizer, cfg.EPOCHS - 1, val_m, last_path)
        print(f"  Last checkpoint saved → {last_path}")
    logger.finish()
    if is_main:
        print_best_summary(best_path, best_epoch, cfg.EPOCHS, best_val_m, selection_metric=ckpt_metric)
    return best_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():

    # ── 0. Setup ──────────────────────────────────────────────────────────────
    is_ddp, rank, world_size = _init_distributed()
    is_main = (rank == 0)

    set_seeds(cfg.SEED)

    run_name     = getattr(cfg, "RUN_NAME", "default_run")
    cfg.SAVE_DIR = cfg.SAVE_DIR / run_name
    if is_main:
        cfg.SAVE_DIR.mkdir(parents=True, exist_ok=True)
    if is_ddp:
        dist.barrier()                                                      # wait for rank 0 to create directory

    # ── Copy config + log file into the run folder (rank 0 only) ─────────────
    if is_main:
        import shutil
        config_path = _parse_config_arg()
        if config_path:
            shutil.copy2(config_path, cfg.SAVE_DIR / Path(config_path).name)
            print(f"  Config copied  → {cfg.SAVE_DIR / Path(config_path).name}")
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id:
            import glob
            matches = glob.glob(f"logs*/*{slurm_job_id}*.log") + \
                      glob.glob(f"logs*/*{slurm_job_id}*.out")
            if matches:
                log_src = matches[0]
                def _copy_log_at_exit(src=log_src, dst_dir=cfg.SAVE_DIR):
                    try:
                        shutil.copy2(src, dst_dir / Path(src).name)
                    except Exception:
                        pass
                import atexit
                atexit.register(_copy_log_at_exit)
                print(f"  Log registered → {cfg.SAVE_DIR / Path(log_src).name}  (copied at exit)")
            else:
                print(f"  SLURM job {slurm_job_id} detected but log file not found in logs*/")

    device = torch.device(f"cuda:{rank}") if is_ddp else torch.device(cfg.DEVICE)

    if is_main:
        print(f"\n{'='*65}")
        print(f"  Kidney EM — {cfg.BACKBONE} + MIL — {cfg.NUM_CLASSES}-class classification")
        print(f"  Mode        : {cfg.TRAINING_MODE}")
        print(f"  Device      : {device}" + (f"  ({world_size} GPUs, DDP)" if is_ddp else ""))
        print(f"  Fold        : {cfg.FOLD} / {cfg.N_FOLDS - 1}")
        print(f"  MIL mode    : {cfg.MIL_MODE}")
        print(f"  Classes     : {cfg.NUM_CLASSES}")
        print(f"  Epochs      : {cfg.EPOCHS}  |  Batch size: {cfg.BATCH_SIZE}")
        print(f"{'='*65}\n")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    if not getattr(cfg, 'FOLDS_PATH', None):
        raise ValueError(
            "FOLDS_PATH is not set in the config. "
            "All configs must specify FOLDS_PATH pointing to folds_total_fixed.json. "
            "The legacy on-the-fly split (patient_level_split_legacy) has been removed."
        )

    if is_main:
        print(f"Using pre-built JSON index: {cfg.PROCESSED_JSON_PATH}")
    df = build_dataframe_v2(cfg.PROCESSED_JSON_PATH)

    if is_main:
        print('Loading the pre-existing data folds.')
    import json
    with open(cfg.FOLDS_PATH) as f:
        fold_data = json.load(f)

    train_df, val_df = patient_level_split(df, fold_data, fold=cfg.FOLD)
    if is_main:
        train_df.to_csv(Path(cfg.SAVE_DIR) / 'df_train.csv', index=False)
        val_df.to_csv(Path(cfg.SAVE_DIR) / 'df_val.csv', index=False)

    # ── 1b. Tabular features (mid-level fusion) ───────────────────────────────
    tab_features = None
    tab_dim      = 0
    if getattr(cfg, "TAB_FUSION", False):
        from dataset import load_tabular_features
        if cfg.TAB_FEATURES_CSV is None:
            raise ValueError("TAB_FUSION=True but TAB_FEATURES_CSV is not set.")
        tab_features, _marker_cols, tab_dim = load_tabular_features(
            csv_path    = cfg.TAB_FEATURES_CSV,
            id_col      = cfg.TAB_ID_COL,
            marker_cols = cfg.TAB_MARKER_COLS,
        )
        if is_main:
            print(f"  Tabular fusion enabled: {tab_dim} markers from {cfg.TAB_FEATURES_CSV}")

    data = build_dataloaders(train_df, val_df, device, tab_features=tab_features)
    class_weights = data["class_weights"]

    # ── 2. Model ──────────────────────────────────────────────────────────────
    model     = build_model(tab_dim=tab_dim).to(device)
    raw_model = model                                                       # keep ref before DDP wrap

    # ── 2b. Feature cache (frozen ViT only) ───────────────────────────────────
    backbone_frozen = (getattr(cfg, 'BACKBONE', 'resnet50') == 'dinov2_vits14' and
                       getattr(cfg, 'FREEZE_BACKBONE', False))
    if backbone_frozen and cfg.TRAINING_MODE == 'patient':
        from dataset import PatientMILDataset, get_transforms
        cache_dir = cfg.SAVE_DIR / 'feature_cache'
        feat_path = cache_dir / 'features.pt'
        all_patient_ds = PatientMILDataset(
            df        = df,
            transform = get_transforms(train=False),
            max_imgs  = None,
        )
        if is_main:
            full_cache = build_feature_cache(
                model           = model,
                patient_dataset = all_patient_ds,
                device          = device,
                cache_dir       = cache_dir,
                checkpoint_path = str(getattr(cfg, 'VIT_CHECKPOINT', '')),
            )
        if is_ddp:
            dist.barrier()                                                  # wait for rank 0 to save the cache
        if not is_main:
            full_cache = torch.load(feat_path, map_location="cpu")
            print(f"  [Rank {rank}] Feature cache loaded from {feat_path}")
        if 'train_patient_ds' in data:
            data['train_patient_ds'].feature_cache = full_cache
        if 'val_patient_ds' in data:
            data['val_patient_ds'].feature_cache = full_cache
        if is_main:
            print(f'  Feature cache active — backbone forward pass skipped during training\n')

    # ── 2c. DDP wrap (after cache extraction, before optimizer) ──────────────
    if is_ddp:
        model = _DDP(model, device_ids=[rank], find_unused_parameters=False)

    # ── 3. Optimizer, scheduler, loss, scaler ─────────────────────────────────
    optimizer = build_optimizer(raw_model)                                  # raw_model exposes .backbone/.head/.attention
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=1e-6
    )
    loss_type = getattr(cfg, "LOSS", "ce")
    if loss_type == "focal":
        focal_gamma = getattr(cfg, "FOCAL_GAMMA", 2.0)
        criterion   = FocalLoss(gamma=focal_gamma, weight=class_weights)
        if is_main: print(f"  Loss: FocalLoss (gamma={focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        if is_main: print(f"  Loss: CrossEntropyLoss")
    scaler = GradScaler(enabled=cfg.AMP)

    # ── 4. Dispatch to the selected approach ──────────────────────────────────
    if cfg.TRAINING_MODE == "image":
        best_path = run_image_level(
            model,
            train_loader   = data["train_loader"],
            val_loader     = data["val_img_loader"],
            train_ds       = data["train_loader"].dataset,
            val_img_ds     = data["val_img_loader"].dataset,
            optimizer      = optimizer,
            criterion      = criterion,
            scaler         = scaler,
            scheduler      = scheduler,
            device         = device,
            run_name       = run_name,
        )

    elif cfg.TRAINING_MODE == "patient":
        best_path = run_patient_level(
            model,
            train_patient_ds = data["train_patient_ds"],
            val_patient_ds   = data["val_patient_ds"],
            optimizer        = optimizer,
            criterion        = criterion,
            scaler           = scaler,
            scheduler        = scheduler,
            device           = device,
            run_name         = run_name,
            rank             = rank,
            world_size       = world_size,
            is_main          = is_main,
        )

    else:
        raise ValueError(
            f"Unknown TRAINING_MODE '{cfg.TRAINING_MODE}'. Choose: 'image', 'patient'."
        )

    if is_main:
        print(f"\n  Final checkpoint: {best_path}\n")
    if is_ddp:
        dist.destroy_process_group()
    return model, best_path


if __name__ == "__main__":
    model, ckpt_path = main()