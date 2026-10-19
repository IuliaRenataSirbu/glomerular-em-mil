from pathlib import Path
# training.py
# ─────────────────────────────────────────────────────────────────────────────
# All training and evaluation functions:
#
#   train_image_level()     – one epoch of image-level training (TRAINING_MODE="image")
#   train_patient_level()   – one epoch of patient-level MIL training (TRAINING_MODE="patient")
#   validate_image_level()  – image-level validation
#   validate_patient_level()– patient-level validation
#   infer_patient()         – single-patient inference with attention visualisation
#
# Which functions are called is controlled entirely by main.py based on cfg.TRAINING_MODE.
# ─────────────────────────────────────────────────────────────────────────────

import contextlib
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm                                  # progress bars for all training and validation loops

from config  import cfg
from dataset import IDX_TO_CLASS                       # int → diagnosis string, used for logging


# ─────────────────────────────────────────────────────────────────────────────
# AVERAGE METER
# ─────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    """Tracks a running weighted average — used to accumulate loss across batches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0   # reset all accumulators

    def update(self, val: float, n: int = 1):
        self.val    = val                                  # most recent value
        self.sum   += val * n                              # cumulative weighted sum
        self.count += n                                    # total sample count
        self.avg    = self.sum / self.count                # running weighted average


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(preds: list, labels: list, num_classes: int,
                    youden_threshold: float = None,
                    class_names: list = None) -> dict:
    """
    Computes classification metrics.
    Works for both image-level, patient-level, and ensemble evaluation.

    Args:
        preds            : list/array of [num_classes] probability arrays, one per sample
        labels           : list/array of integer true labels, one per sample
        num_classes      : total number of classes
        youden_threshold : if given, apply this threshold for binary Youden metrics
                           instead of deriving it from the data (use val threshold at test time)
        class_names      : list of class name strings; if None, derived from IDX_TO_CLASS

    Returns:
        dict with keys: acc, bacc, auc, f1_macro, f1_weighted,
                        per_class_auc, per_class_acc, n_patients,
                        + Youden keys for binary tasks
    """
    from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                                  accuracy_score, f1_score, roc_curve,
                                  confusion_matrix)
    from dataset import IDX_TO_CLASS

    preds  = np.array(preds)                                               # [N, C]
    labels = np.array(labels)                                              # [N]

    if class_names is None:
        class_names = [IDX_TO_CLASS.get(i, str(i)) for i in range(num_classes)]

    pred_cls    = preds.argmax(1)
    acc         = float(accuracy_score(labels, pred_cls))
    bacc        = float(balanced_accuracy_score(labels, pred_cls))
    f1_macro    = float(f1_score(labels, pred_cls, average="macro",     zero_division=0))
    f1_weighted = float(f1_score(labels, pred_cls, average="weighted",  zero_division=0))

    try:
        if num_classes == 2:
            auc = float(roc_auc_score(labels, preds[:, 1]))
        else:
            # sklearn ≥ 1.3 requires rows to sum to exactly 1.0 for multiclass
            # OvR AUC.  Averaged / max-pooled ensemble predictions accumulate
            # floating-point drift after the CSV round-trip, so we normalise
            # here.  AUC is rank-based: dividing each row by its sum does not
            # change the metric value, only satisfies the strict sum-to-1 check.
            row_sums = preds.sum(axis=1, keepdims=True)
            preds_prob = preds / np.where(row_sums > 0, row_sums, 1.0)
            auc = float(roc_auc_score(labels, preds_prob,
                                      multi_class="ovr", average="macro"))
    except Exception as _auc_exc:
        # Broad catch so unexpected exception types don't silently suppress AUC.
        # First failure is warned so the caller can diagnose the root cause.
        if not getattr(compute_metrics, "_auc_warned", False):
            import warnings
            warnings.warn(
                f"AUC computation failed ({type(_auc_exc).__name__}: {_auc_exc}); "
                f"preds.shape={preds.shape}  unique_labels={np.unique(labels).tolist()}  "
                f"num_classes={num_classes}",
                stacklevel=2,
            )
            compute_metrics._auc_warned = True
        auc = float("nan")

    # Per-class AUC (one-vs-rest)
    per_class_auc = {}
    for c, name in enumerate(class_names):
        try:
            per_class_auc[name] = float(roc_auc_score(
                (labels == c).astype(int), preds[:, c]
            ))
        except ValueError:
            per_class_auc[name] = float("nan")

    # Per-class accuracy from confusion matrix diagonal
    cm = confusion_matrix(labels, pred_cls, labels=list(range(num_classes)))
    per_class_acc = {}
    for c, name in enumerate(class_names):
        total = cm[c].sum()
        per_class_acc[name] = float(cm[c, c] / total) if total > 0 else float("nan")

    metrics = {
        "acc":           acc,
        "bacc":          bacc,
        "auc":           auc,
        "f1_macro":      f1_macro,
        "f1_weighted":   f1_weighted,
        "per_class_auc": per_class_auc,
        "per_class_acc": per_class_acc,
        "n_patients":    int(len(labels)),
    }

    if num_classes >= 3:
        top2_idx = np.argsort(preds, axis=1)[:, -2:]
        metrics["top2_acc"] = float(np.mean(
            (top2_idx == labels[:, None]).any(axis=1)
        ))

    # Binary-only: Youden-optimal threshold metrics
    if num_classes == 2 and not np.isnan(auc):
        if youden_threshold is None:
            fpr, tpr, thresholds = roc_curve(labels, preds[:, 1])
            j_idx  = np.argmax(tpr - fpr)
            thresh = float(thresholds[j_idx])
            sens   = float(tpr[j_idx])
            spec   = float(1.0 - fpr[j_idx])
        else:
            thresh = float(youden_threshold)
            fpr, tpr, thresholds = roc_curve(labels, preds[:, 1])
            idx    = np.argmin(np.abs(thresholds - thresh))
            sens   = float(tpr[idx])
            spec   = float(1.0 - fpr[idx])
        youden_cls = (preds[:, 1] >= thresh).astype(int)
        metrics["youden_threshold"]   = thresh
        metrics["youden_sensitivity"] = sens
        metrics["youden_specificity"] = spec
        metrics["youden_acc"]         = float(accuracy_score(labels, youden_cls))
        metrics["youden_bacc"]        = float(balanced_accuracy_score(labels, youden_cls))
        metrics["youden_f1_macro"]    = float(f1_score(labels, youden_cls, average="macro", zero_division=0))

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING — IMAGE LEVEL
# Used by: TRAINING_MODE="image"
# ─────────────────────────────────────────────────────────────────────────────


def save_validation_results(preds: list, labels: list, pids: list,
                            save_dir, epoch: int, tag: str = "best",
                            youden_threshold: float = None) -> None:
    """
    Saves full validation results when a new best checkpoint is found.

    Writes to save_dir/:
        predictions_{tag}.csv      — patient_id, true_label, pred_label, prob_0..prob_C
        confusion_matrix_{tag}.png — normalised confusion matrix heatmap
        metrics_{tag}.json         — all scalar metrics at this checkpoint
    """
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report
    from config import cfg
    from dataset import IDX_TO_CLASS

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    preds_arr  = np.array(preds)                              # [N, C]
    labels_arr = np.array(labels)                             # [N]
    pred_cls   = preds_arr.argmax(1)                          # [N]

    # ── CSV: one row per sample ───────────────────────────────────────────────
    rows = []
    for pid, true_l, pred_l, prob in zip(pids, labels_arr, pred_cls, preds_arr):
        row = {
            "patient_id":   pid,
            "true_label":   int(true_l),
            "true_class":   IDX_TO_CLASS.get(int(true_l), str(true_l)),
            "pred_label":   int(pred_l),
            "pred_class":   IDX_TO_CLASS.get(int(pred_l), str(pred_l)),
            "correct":      int(true_l) == int(pred_l),
        }
        for c in range(len(prob)):
            row[f"prob_{c}"] = float(prob[c])
        rows.append(row)

    csv_path = save_dir / f"predictions_{tag}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Predictions saved → {csv_path}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    class_names = [IDX_TO_CLASS.get(i, str(i)) for i in range(preds_arr.shape[1])]
    # Shorten long names for readability
    short_names = [n.split("(")[0].strip()[:18] for n in class_names]

    cm      = confusion_matrix(labels_arr, pred_cls, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names))))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {tag}  (epoch {epoch})")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                    ha="center", va="center", fontsize=7, color=color)

    plt.tight_layout()
    cm_path = save_dir / f"confusion_matrix_{tag}.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved → {cm_path}")

    # ── Metrics JSON ─────────────────────────────────────────────────────────
    metrics = compute_metrics(preds, labels, len(class_names),
                              youden_threshold=youden_threshold)
    report  = classification_report(
        labels_arr, pred_cls,
        target_names=class_names, zero_division=0, output_dict=True
    )
    out = {
        "epoch":        epoch,
        "tag":          tag,
        "metrics":      metrics,
        "per_class":    report,
    }
    json_path = save_dir / f"metrics_{tag}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Metrics JSON saved  → {json_path}")




def build_feature_cache(model, patient_dataset, device, cache_dir,
                        checkpoint_path) -> dict:
    """
    Pre-extracts frozen backbone features for ALL images of every patient and
    saves them to cache_dir/features.pt. Sampling (MAX_IMGS_PER_PATIENT, bag
    dropout) is applied later in PatientMILDataset.__getitem__ on the cached
    feature tensors, so epoch-varying diversity is preserved.

    On subsequent calls the cache is reused if the checkpoint mtime is unchanged.

    Returns:
        dict {patient_id: tensor [N_all_imgs, embed_dim]}  (tensors on CPU)
    """
    import json
    import os

    cache_dir  = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    feat_path  = cache_dir / "features.pt"
    manif_path = cache_dir / "cache_manifest.json"

    _vit_keywords = {"imagenet", "random"}
    if checkpoint_path and str(checkpoint_path) not in _vit_keywords:
        ckpt_mtime = os.path.getmtime(checkpoint_path)
    else:
        ckpt_mtime = 0  # keyword init (imagenet/random) has no file — treat as stable

    # ── Check if valid cache already exists ──────────────────────────────────
    if feat_path.exists() and manif_path.exists():
        with open(manif_path) as f:
            manifest = json.load(f)
        ckpt_ok    = abs(manifest.get("ckpt_mtime", 0) - ckpt_mtime) < 1.0
        all_imgs_ok = manifest.get("all_images", False)
        if ckpt_ok and all_imgs_ok:
            print(f"  Feature cache found — loading from {feat_path}")
            cache = torch.load(feat_path, map_location="cpu")
            print(f"  Loaded features for {len(cache)} patients")
            return cache
        elif not all_imgs_ok:
            print(f"  Feature cache stale (built with max_imgs cap, not all images) — rebuilding...")
        else:
            print(f"  Feature cache stale (checkpoint changed) — rebuilding...")

    # ── Extract features ──────────────────────────────────────────────────────
    print(f"\n  Building feature cache for {len(patient_dataset)} patients...")
    print(f"  Cache will be saved to: {feat_path}")
    model.eval()
    SUB_BATCH = 16
    cache = {}

    progress = tqdm(range(len(patient_dataset)),
                    desc="  Extracting features", unit="patient")

    with torch.no_grad():
        for i in progress:
            batch = patient_dataset[i]
            imgs  = batch["images"].to(device)
            label = batch["label"]
            pid   = batch["patient_id"]

            feats = []
            for start in range(0, len(imgs), SUB_BATCH):
                sub = imgs[start : start + SUB_BATCH]
                with autocast(enabled=cfg.AMP):
                    f = model.encode(sub)                  # [≤SUB_BATCH, D]
                feats.append(f.cpu())
            cache[pid] = torch.cat(feats, dim=0)           # [N_all_imgs, D] on CPU

            progress.set_postfix(patient=pid)

    # ── Save cache + manifest ─────────────────────────────────────────────────
    torch.save(cache, feat_path)
    with open(manif_path, "w") as f:
        json.dump({"ckpt_mtime": ckpt_mtime, "all_images": True,
                   "n_patients": len(cache), "embed_dim": next(iter(cache.values())).shape[-1]}, f, indent=2)

    print(f"  Feature cache saved ({len(cache)} patients) → {feat_path}\n")
    return cache



def train_image_level(model, loader, optimizer, criterion, scaler, device,
                      epoch: int = 0, total_epochs: int = 0) -> dict:
    """
    One epoch of image-level training. Each image is classified independently
    using its patient's label as supervision. The MIL aggregation module is
    NOT called here — gradients only flow through backbone + head.

    Used by TRAINING_MODE="image".

    Args:
        model         : KidneyResNet instance
        loader        : DataLoader over EMImageDataset (image-level batches)
        optimizer     : AdamW optimizer
        criterion     : weighted CrossEntropyLoss
        scaler        : GradScaler for AMP
        device        : torch.device
        epoch         : current epoch index (0-based, used for tqdm display)
        total_epochs  : total number of epochs in this stage (used for tqdm display)

    Returns:
        dict: {"loss", "auc", "bacc"}
    """
    model.train()                                                          # enable dropout + BatchNorm training mode

    loss_meter            = AverageMeter()
    all_preds, all_labels = [], []

    progress = tqdm(                                                       # live progress bar
        loader,
        desc=f"  Train[img]  Epoch {epoch+1:03d}/{total_epochs}",         # shows current epoch in the bar label
        unit="batch",
        leave=False,                                                       # bar disappears when epoch ends (cleaner output)
    )

    for imgs, labels, _ in progress:                                       # _ = patient_ids (not needed for image training)
        imgs   = imgs.to(device)                                           # [B, 3, 224, 224] → GPU
        labels = labels.to(device)                                         # [B] integer class labels → GPU

        optimizer.zero_grad()                                              # clear gradients from previous batch

        with autocast(enabled=cfg.AMP):                                    # float16 forward pass (faster, less VRAM)
            # ImageClassifier uses forward(); MILClassifier uses forward_image_batch()
            # Both produce [B, num_classes] logits — no MIL aggregation in either case
            if hasattr(model, "forward_image_batch"):
                logits = model.forward_image_batch(imgs)                  # MILClassifier image-level path (Hybrid Stage 1)
            else:
                logits = model(imgs)                                      # ImageClassifier standard forward (Approach 1)
            loss = criterion(logits, labels)                              # weighted cross-entropy scalar

        scaler.scale(loss).backward()                                      # backprop with AMP loss scaling
        scaler.unscale_(optimizer)                                         # undo scaling before gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)       # clip gradient norm
        scaler.step(optimizer)                                             # apply weight update
        scaler.update()                                                    # update AMP scale factor

        loss_meter.update(loss.item(), imgs.size(0))                      # accumulate weighted average loss

        probs = torch.softmax(logits.detach().float(), dim=1).cpu().numpy()  # [B, 7] class probabilities
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

        progress.set_postfix(loss=f"{loss_meter.avg:.4f}")                # show running loss on the bar

    metrics         = compute_metrics(all_preds, all_labels, cfg.NUM_CLASSES)
    metrics["loss"] = loss_meter.avg
    return metrics                                                         # {"loss", "auc", "bacc"}


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING — PATIENT LEVEL (FULL MIL)
# Used by: TRAINING_MODE="patient"
# ─────────────────────────────────────────────────────────────────────────────

def train_patient_level(model, patient_dataset, optimizer, criterion, scaler, device,
                        epoch: int = 0, total_epochs: int = 0, indices=None) -> dict:
    """
    One epoch of patient-level MIL training. For each patient, all their images
    are encoded, then aggregated via MIL into a single patient-level prediction.
    Gradients flow through the ENTIRE model: backbone + attention + head.

    Used by TRAINING_MODE="patient".

    Args:
        model          : KidneyResNet or DDP-wrapped KidneyResNet
        patient_dataset: PatientMILDataset (one item = full bag of images per patient)
        optimizer      : AdamW optimizer
        criterion      : weighted CrossEntropyLoss
        scaler         : GradScaler for AMP
        device         : torch.device
        epoch          : current epoch index (0-based)
        total_epochs   : total epochs in this stage
        indices        : subset of patient indices for this rank (None = all patients)

    Returns:
        dict: {"loss", "auc", "bacc"}
    """
    patient_dataset.epoch = epoch                                          # update sampling seed: cfg.SEED + epoch
    model.train()

    # Unwrap DDP to access custom methods (encode, aggregate).
    # Gradient sync still happens via DDP's parameter hooks during backward.
    _model   = model.module if hasattr(model, 'module') else model
    _no_sync = model.no_sync if hasattr(model, 'no_sync') else contextlib.nullcontext

    loss_meter            = AverageMeter()
    all_preds, all_labels = [], []
    SUB_BATCH             = 16

    indices     = list(indices) if indices is not None else list(range(len(patient_dataset)))
    n_patients  = len(indices)
    accum_steps = max(1, getattr(cfg, 'GRAD_ACCUM_STEPS', 1))

    progress = tqdm(
        range(n_patients),
        desc=f"  Train[pat]  Epoch {epoch+1:03d}/{total_epochs}",
        unit="patient",
        leave=False,
    )

    optimizer.zero_grad()

    for step in progress:
        i = indices[step]
        batch     = patient_dataset[i]
        imgs      = batch["images"].to(device)
        label     = batch["label"]
        pid       = batch["patient_id"]
        tab_feats = batch.get("tab_feats", None)
        label = torch.tensor([label], dtype=torch.long).to(device)

        if imgs.dim() == 2:
            feats = imgs.float()
        else:
            feats = []
            for start in range(0, len(imgs), SUB_BATCH):
                sub = imgs[start : start + SUB_BATCH]
                with autocast(enabled=cfg.AMP):
                    f = _model.encode(sub)
                feats.append(f)
            feats = torch.cat(feats, dim=0).float()

        tab = tab_feats.to(device).float() if tab_feats is not None else None
        with autocast(enabled=cfg.AMP):
            logits, _ = _model.aggregate(feats, tab_feats=tab)
            logits    = logits.unsqueeze(0)
            loss      = criterion(logits, label)

        is_last_accum = (step + 1) % accum_steps == 0 or (step + 1) == n_patients
        # Skip DDP gradient sync on intermediate accumulation steps
        with contextlib.nullcontext() if is_last_accum else _no_sync():
            scaler.scale(loss / accum_steps).backward()

        if is_last_accum:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(_model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_meter.update(loss.item(), 1)

        prob = torch.softmax(logits.detach().float(), dim=1).cpu().numpy()
        all_preds.append(prob[0])
        all_labels.append(label.item())

        progress.set_postfix(loss=f"{loss_meter.avg:.4f}", patient=pid)

    metrics         = compute_metrics(all_preds, all_labels, cfg.NUM_CLASSES)
    metrics["loss"] = loss_meter.avg
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION — IMAGE LEVEL
# Used by: TRAINING_MODE="image"
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()                                                           # no gradient tracking needed for validation
def validate_image_level(model, loader, device,
                         epoch: int = 0, total_epochs: int = 0,
                         return_preds: bool = False):
    """
    Image-level validation. Each image is classified independently.

    Args:
        return_preds : if True, also return (all_preds, all_labels, all_pids)
                       for saving predictions at the best checkpoint.

    Returns:
        metrics dict — always returned
        (preds, labels, pids) — only when return_preds=True
    """
    model.eval()

    all_preds, all_labels, all_pids = [], [], []

    progress = tqdm(
        loader,
        desc=f"  Val  [img]  Epoch {epoch+1:03d}/{total_epochs}",
        unit="batch",
        leave=False,
    )

    for imgs, labels, pids in progress:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        with autocast(enabled=cfg.AMP):
            if hasattr(model, "forward_image_batch"):
                logits = model.forward_image_batch(imgs)
            else:
                logits = model(imgs)

        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()        # [B, C]
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        all_pids.extend(list(pids))

    metrics         = compute_metrics(all_preds, all_labels, cfg.NUM_CLASSES)
    metrics["loss"] = 0.0

    if return_preds:
        return metrics, (all_preds, all_labels, all_pids)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION — PATIENT LEVEL
# Used by: TRAINING_MODE="patient"
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate_patient_level(model, patient_dataset, device,
                           epoch: int = 0, total_epochs: int = 0,
                           return_preds: bool = False, indices=None,
                           youden_threshold: float = None):
    """
    Patient-level validation. All images for each patient are encoded and
    aggregated via MIL into one prediction per patient.

    Used by TRAINING_MODE="patient".

    Args:
        indices : subset of patient indices for this rank (None = all patients).
                  When using DDP, each rank validates its subset; the caller
                  gathers predictions across ranks before computing final metrics.

    Returns:
        dict: {"auc", "bacc"}
    """
    model.eval()

    _model = model.module if hasattr(model, 'module') else model

    all_preds, all_labels = [], []
    SUB_BATCH = 16

    indices = list(indices) if indices is not None else list(range(len(patient_dataset)))

    progress = tqdm(
        range(len(indices)),
        desc=f"  Val  [pat]  Epoch {epoch+1:03d}/{total_epochs}",
        unit="patient",
        leave=False,
    )

    all_pids = []

    for step in progress:
        i = indices[step]
        batch     = patient_dataset[i]
        imgs      = batch["images"].to(device)
        label     = batch["label"]
        pid       = batch["patient_id"]
        tab_feats = batch.get("tab_feats", None)

        if imgs.dim() == 2:
            feats = imgs.float()
        else:
            feats = []
            for start in range(0, len(imgs), SUB_BATCH):
                sub = imgs[start : start + SUB_BATCH]
                with autocast(enabled=cfg.AMP):
                    f = _model.encode(sub)
                feats.append(f)
            feats = torch.cat(feats, dim=0).float()

        tab = tab_feats.to(device).float() if tab_feats is not None else None
        with autocast(enabled=cfg.AMP):
            logits, _ = _model.aggregate(feats, tab_feats=tab)
        prob = torch.softmax(logits.float(), dim=0).cpu().numpy()

        all_preds.append(prob)
        all_labels.append(label)
        all_pids.append(pid)

        progress.set_postfix(patient=pid)

    metrics = compute_metrics(all_preds, all_labels, cfg.NUM_CLASSES,
                              youden_threshold=youden_threshold)
    if return_preds:
        return metrics, (all_preds, all_labels, all_pids)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE — SINGLE PATIENT
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_patient(model, image_paths: list, device, transform=None):
    """
    Runs inference on a single patient's image bag and prints interpretability info.
    Uses the patient-level forward pass regardless of TRAINING_MODE.

    Args:
        model        : trained KidneyResNet
        image_paths  : list of .tif paths for this patient
        device       : torch.device
        transform    : preprocessing pipeline; defaults to val transforms if None

    Returns:
        probs : [num_classes] numpy array of predicted probabilities
        attn  : [N_imgs] numpy array of per-image attention weights, or None
    """
    from PIL import Image
    from dataset import get_transforms

    if transform is None:
        transform = get_transforms(train=False)                            # deterministic val preprocessing

    model.eval()

    imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")                                # open .tif → 3-channel RGB
        imgs.append(transform(img))
    imgs = torch.stack(imgs).to(device)                                    # [N_imgs, 3, 224, 224]

    if not hasattr(model, "forward_patient"):                             # guard: ImageClassifier has no forward_patient
        raise RuntimeError(
            "infer_patient() requires a MILClassifier (TRAINING_MODE='patient'). "
            "ImageClassifier (TRAINING_MODE='image') does not support patient-level inference."
        )
    logits, attn = model.forward_patient(imgs)                            # patient-level MIL prediction
    probs = torch.softmax(logits.float(), dim=0).cpu().numpy()            # [num_classes] probabilities

    top3 = probs.argsort()[::-1][:3]                                      # indices of top-3 predictions
    print("Predicted probabilities (top 3):")
    for idx in top3:
        print(f"  {probs[idx]:.4f}  [{idx}] {IDX_TO_CLASS[idx]}")

    if attn is not None:                                                   # attention MIL only
        attn   = attn.cpu().numpy()
        ranked = sorted(zip(image_paths, attn), key=lambda x: -x[1])     # sort by descending attention weight
        print("\nTop 5 most influential images (by attention weight):")
        for path, weight in ranked[:5]:
            print(f"  {weight:.4f}  {path}")

    return probs, attn