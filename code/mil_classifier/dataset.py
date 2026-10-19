# dataset.py
# ─────────────────────────────────────────────────────────────────────────────
# Handles all data loading and preparation:
#   1. build_dataframe_v2()  – loads a pre-built JSON image index, filters to
#                              target classes, encodes labels → integers.
#   2. patient_level_split() – loads pre-computed folds and returns (train_df, val_df).
#   3. get_transforms()      – image augmentation (train) and preprocessing (val).
#   4. EMImageDataset        – image-level Dataset used during training.
#   5. PatientMILDataset     – patient-level Dataset (bag of images) used
#                              during validation and inference.
#   6. build_dataloaders()   – mode-aware builder: reads cfg.TRAINING_MODE and
#                              returns exactly the datasets/loaders each approach needs.
#                              Returns a dict — keys depend on the active mode.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from config import cfg                                 # central config object


# ─────────────────────────────────────────────────────────────────────────────
# CLASS → INTEGER LABEL MAPPING
# Built from cfg.CLASSES so the mapping is always consistent with the config.
# ─────────────────────────────────────────────────────────────────────────────

CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(cfg.CLASSES)}

IDX_TO_CLASS = {idx: cls_name for cls_name, idx in CLASS_TO_IDX.items()}
# Reverse mapping used when converting predicted integer back to a diagnosis name.


# ─────────────────────────────────────────────────────────────────────────────
# 1. BUILD DATAFRAME FROM JSON INDEX
# ─────────────────────────────────────────────────────────────────────────────

def build_dataframe_v2(json_path: Path) -> pd.DataFrame:
    """
    Reads a pre-built JSON index file (see data/images_index_example.json for format).
    The JSON must be a list of dicts; relevant keys per record:

        "folder_above"  – patient folder name (= patient ID)
        "fullpath"      – absolute path to the .tif image file
        "diagnosis"     – string diagnosis label

    Returns a DataFrame with columns:
        patient_id  – folder name
        full_path   – absolute path to the .tif image file
        label       – integer class index
        diagnosis   – original string diagnosis
    """
    import json

    with open(json_path, "r") as f:
        records = json.load(f)

    df = pd.DataFrame(records)

    # Rename JSON keys to the canonical column names used everywhere else
    df = df.rename(columns={
        "folder_above": "patient_id",
        "fullpath":     "full_path",
        "patient_id":   "patient_id_short",       # deprecated key — keep but rename
    })

    # Filter to target classes; everything else is discarded
    mask = df["diagnosis"].isin(cfg.CLASSES)
    df   = df[mask].copy().reset_index(drop=True)

    print(f"JSON loaded: {len(records)} total records → {len(df)} kept after "
          f"filtering to {cfg.NUM_CLASSES} classes")
    for cls in cfg.CLASSES:
        n = (df["diagnosis"] == cls).sum()
        print(f"  [{CLASS_TO_IDX[cls]}] {cls}: {n} images")

    # Convert string diagnosis to integer label
    df["label"] = df["diagnosis"].map(CLASS_TO_IDX)

    # Keep only the four canonical columns (drop any extra JSON keys)
    df = df[["patient_id", "full_path", "diagnosis", "label"]].copy()

    print(f"\nFinal dataset: {df['patient_id'].nunique()} patients, {len(df)} .tif images")
    return df



# ─────────────────────────────────────────────────────────────────────────────
# 2. PATIENT-LEVEL TRAIN / VAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def patient_level_split(df: pd.DataFrame, fold_data: dict, fold: int):
    """Load pre-computed stratified split. Returns (train_df, val_df)."""
    split = fold_data["folds"][str(fold)]
    train_patients = set(split["train"])
    val_patients   = set(split["val"])

    train_df = df[df["patient_id"].isin(train_patients)].reset_index(drop=True)
    val_df   = df[df["patient_id"].isin(val_patients)].reset_index(drop=True)

    print(f"\nFold {fold}:")
    print(f"  Train: {train_df['patient_id'].nunique()} patients | "
          f"{len(train_df)} images")
    print(f"  Val  : {val_df['patient_id'].nunique()} patients | "
          f"{len(val_df)} images")
    for cls in cfg.CLASSES:
        idx = CLASS_TO_IDX[cls]
        n = val_df[val_df["label"] == idx]["patient_id"].nunique()
        print(f"    Val [{idx}] {cls}: {n} patients")

    return train_df, val_df

# ─────────────────────────────────────────────────────────────────────────────
# 3. IMAGE TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(train: bool) -> T.Compose:
    """
    Returns the image preprocessing pipeline for the active cfg.AUG_MODE.

    AUG_MODE = "v1"  — original: squish to 256×256, RandomCrop(224), HFlip, VFlip,
                       ±15° rotation, ColorJitter(0.2)
    AUG_MODE = "v1a" — v1 with aspect-preserving resize (shorter edge → 256) instead
                       of squish; everything else identical to v1
    AUG_MODE = "v2"  — improved: aspect-preserving resize+crop, VFlip, discrete 90°
                       rotation, ±15° jitter, GaussianBlur, stronger ColorJitter(0.4)
    AUG_MODE = "v2s" — v2 with squish resize instead of aspect-preserving resize;
                       everything else identical to v2 (discrete 90° rotation,
                       ±15° jitter, GaussianBlur, stronger ColorJitter(0.4))
    """
    _norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    aug_mode = getattr(cfg, "AUG_MODE", "v2")

    if aug_mode == "v1":
        if train:
            return T.Compose([
                T.Resize((cfg.IMG_SIZE + 32, cfg.IMG_SIZE + 32)),  # squish to 256×256 (aspect ratio NOT preserved)
                T.RandomCrop(cfg.IMG_SIZE),                        # random 224×224 crop
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                _norm,
            ])
        else:
            return T.Compose([
                T.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),            # squish to 224×224
                T.ToTensor(),
                _norm,
            ])

    elif aug_mode == "v1a":
        if train:
            return T.Compose([
                T.Resize(cfg.IMG_SIZE + 32),                       # shorter edge → 256, aspect ratio preserved
                T.RandomCrop(cfg.IMG_SIZE),                        # random 224×224 crop
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                _norm,
            ])
        else:
            return T.Compose([
                T.Resize(cfg.IMG_SIZE + 32),                       # shorter edge → 256
                T.CenterCrop(cfg.IMG_SIZE),                        # deterministic 224×224 center crop
                T.ToTensor(),
                _norm,
            ])

    elif aug_mode == "v2":
        if train:
            return T.Compose([
                T.Resize(cfg.IMG_SIZE + 32),                   # shorter edge → 256, aspect ratio preserved
                T.RandomCrop(cfg.IMG_SIZE),                    # random 224×224 crop
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomChoice([                               # discrete 90° step — tissue has no canonical orientation
                    T.RandomRotation((0, 0)),
                    T.RandomRotation((90, 90)),
                    T.RandomRotation((180, 180)),
                    T.RandomRotation((270, 270)),
                ]),
                T.RandomRotation(15),                          # fine ±15° jitter on top of the discrete step
                T.RandomApply([T.GaussianBlur(kernel_size=23,
                                              sigma=(0.1, 2.0))], p=0.5),
                T.ColorJitter(brightness=0.4, contrast=0.4),
                T.ToTensor(),
                _norm,
            ])
        else:
            return T.Compose([
                T.Resize(cfg.IMG_SIZE + 32),                   # shorter edge → 256 — same scale as training
                T.CenterCrop(cfg.IMG_SIZE),                    # deterministic 224×224 center crop
                T.ToTensor(),
                _norm,
            ])

    elif aug_mode == "v2s":
        if train:
            return T.Compose([
                T.Resize((cfg.IMG_SIZE + 32, cfg.IMG_SIZE + 32)),  # squish to 256×256 (aspect ratio NOT preserved)
                T.RandomCrop(cfg.IMG_SIZE),                        # random 224×224 crop
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomChoice([                                   # discrete 90° step
                    T.RandomRotation((0, 0)),
                    T.RandomRotation((90, 90)),
                    T.RandomRotation((180, 180)),
                    T.RandomRotation((270, 270)),
                ]),
                T.RandomRotation(15),                              # fine ±15° jitter on top
                T.RandomApply([T.GaussianBlur(kernel_size=23,
                                              sigma=(0.1, 2.0))], p=0.5),
                T.ColorJitter(brightness=0.4, contrast=0.4),
                T.ToTensor(),
                _norm,
            ])
        else:
            return T.Compose([
                T.Resize((cfg.IMG_SIZE + 32, cfg.IMG_SIZE + 32)),  # squish to 256×256 — same scale as training
                T.CenterCrop(cfg.IMG_SIZE),                        # deterministic 224×224 center crop
                T.ToTensor(),
                _norm,
            ])

    else:
        raise ValueError(
            f"Unknown AUG_MODE '{aug_mode}'. Choose: 'v1', 'v1a', 'v2', 'v2s'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3b. TABULAR FEATURE LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_tabular_features(csv_path, id_col: str, marker_cols=None):
    """
    Load and ternary-encode marker features from a CSV.

    Encoding: 1.0 → +1 (positive), 0.0 → -1 (negative), NaN → 0 (not measured).

    Returns:
        tab_features  : {patient_id_str: np.ndarray float32 [n_markers]}
        marker_cols   : list of marker column names used
        tab_dim       : len(marker_cols)
    """
    df = pd.read_csv(csv_path)
    if id_col not in df.columns:
        raise ValueError(f"TAB_ID_COL '{id_col}' not found in {csv_path}. "
                         f"Available columns: {list(df.columns)}")

    if marker_cols is None:
        marker_cols = [c for c in df.columns if c.endswith("bin_corrected")]
        if not marker_cols:
            raise ValueError(
                f"No columns ending in 'bin_corrected' found in {csv_path}. "
                f"Set TAB_MARKER_COLS explicitly."
            )
        print(f"  TAB auto-detected {len(marker_cols)} marker columns: {marker_cols}")
    else:
        missing = [c for c in marker_cols if c not in df.columns]
        if missing:
            raise ValueError(f"TAB_MARKER_COLS not found in CSV: {missing}")

    tab_features = {}
    for _, row in df.iterrows():
        pid   = str(row[id_col])
        feats = np.array(
            [1.0 if row[c] == 1.0 else (-1.0 if row[c] == 0.0 else 0.0)
             for c in marker_cols],
            dtype=np.float32,
        )
        tab_features[pid] = feats

    print(f"  TAB features loaded: {len(tab_features)} patients, {len(marker_cols)} markers")
    return tab_features, marker_cols, len(marker_cols)


# ─────────────────────────────────────────────────────────────────────────────
# 4. IMAGE-LEVEL DATASET  (training)
# ─────────────────────────────────────────────────────────────────────────────

class EMImageDataset(Dataset):
    """
    Each __getitem__ returns ONE .tif image with its patient's integer label.
    Used during training where images are processed independently.
    The patient's class label is propagated to every image it contains.
    """

    def __init__(self, df: pd.DataFrame, transform: T.Compose = None):
        self.df             = df.reset_index(drop=True)        # image-level DataFrame; reset so iloc[i] == row i
        self.transform      = transform                        # augmentation / preprocessing pipeline
        self.corrupted_files = set()                           # ← CHANGED: stores DISTINCT corrupted file paths (set ignores duplicates)

    def __len__(self) -> int:
        return len(self.df)                                    # total number of images in this split

    def __getitem__(self, idx: int):
        for offset in range(len(self.df)):
            row = self.df.iloc[(idx + offset) % len(self.df)]
            try:
                img = Image.open(row["full_path"]).convert("RGB")
            except Exception:
                self.corrupted_files.add(row["full_path"])
                print(f"  WARNING: skipping corrupted file: {row['full_path']}")
                continue
            if self.transform:
                img = self.transform(img)
            return img, int(row["label"]), row["patient_id"]
        raise RuntimeError(f"All {len(self.df)} files in EMImageDataset are corrupted.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. PATIENT-LEVEL DATASET  (validation / inference)
# ─────────────────────────────────────────────────────────────────────────────

class PatientMILDataset(Dataset):
    """
    Each __getitem__ returns ALL images for ONE patient stacked into a tensor.
    This "bag" is fed to the MIL aggregation module to produce one prediction
    per patient — the clinically meaningful unit of evaluation.
    """

    def __init__(self, df: pd.DataFrame, transform: T.Compose = None, max_imgs: int = None,
                 feature_cache: dict = None, bag_dropout: float = 0.0,
                 tab_features: dict = None,
                 tab_dropout: float = 0.0,
                 tab_full_dropout_prob: float = 0.0):
        self.transform       = transform
        self.max_imgs        = max_imgs
        self.corrupted_files = set()
        # feature_cache: {patient_id: tensor [N_imgs, embed_dim]} — if provided,
        # __getitem__ returns pre-extracted features instead of loading images.
        # Only valid when backbone is frozen (features never change).
        self.feature_cache        = feature_cache or {}
        self.bag_dropout          = bag_dropout
        # tab_features: {patient_id: np.ndarray [n_markers]} — ternary-encoded markers.
        # When provided, __getitem__ includes "tab_feats" in the returned dict.
        # tab_dropout and tab_full_dropout_prob are applied only during training
        # (caller passes 0.0 for val/test datasets).
        self.tab_features         = tab_features or {}
        self.tab_dropout          = tab_dropout
        self.tab_full_dropout_prob = tab_full_dropout_prob
        # Epoch counter — updated each epoch by train_patient_level so that
        # image sampling uses a different (but reproducible) seed each epoch:
        # random_state = cfg.SEED + epoch. Val/test datasets keep epoch=0,
        # giving deterministic sampling across all evaluations.
        self.epoch = 0

        self.patients = df["patient_id"].unique().tolist()                 # ordered list of unique patient IDs

        # Pre-build lookup dicts to avoid re-filtering the DataFrame on every __getitem__
        self.patient2rows = {
            pid: df[df["patient_id"] == pid].reset_index(drop=True)
            for pid in self.patients
        }
        self.labels = {
            pid: int(rows["label"].iloc[0])                                # all rows for a patient share the same label
            for pid, rows in self.patient2rows.items()
        }

    def __len__(self) -> int:
        return len(self.patients)                                          # one item per patient

    def __getitem__(self, idx: int):
        pid   = self.patients[idx]
        label = self.labels[pid]

        # Per-patient seed: mix epoch and patient index so two patients with the
        # same bag size don't always draw the same image positions.
        # Multiply epoch by a large constant so epoch=1,idx=0 != epoch=0,idx=10000.
        pseed = cfg.SEED + self.epoch * 10000 + idx

        # ── Tabular features ─────────────────────────────────────────────
        tab_feats = None
        if self.tab_features:
            pid_str = str(pid)
            if pid_str not in self.tab_features:
                raise KeyError(
                    f"Patient '{pid_str}' not found in tab_features. "
                    f"Ensure all patients in this split are in TAB_FEATURES_CSV."
                )
            tab_arr = self.tab_features[pid_str].copy()

            if self.tab_full_dropout_prob > 0 or self.tab_dropout > 0:
                rng_tab = np.random.default_rng(pseed + 20000)
                if (self.tab_full_dropout_prob > 0
                        and rng_tab.random() < self.tab_full_dropout_prob):
                    tab_arr[:] = 0.0
                elif self.tab_dropout > 0:
                    mask = rng_tab.random(tab_arr.shape) > self.tab_dropout
                    tab_arr *= mask.astype(np.float32)

            tab_feats = torch.from_numpy(tab_arr)

        # ── Fast path: apply sampling on pre-extracted feature vectors ───
        if self.feature_cache and pid in self.feature_cache:
            feats    = self.feature_cache[pid]             # [N_all, D]
            n        = len(feats)
            feat_idx = np.arange(n)

            if self.max_imgs and n > self.max_imgs:
                feat_idx = np.random.default_rng(pseed).choice(
                    n, self.max_imgs, replace=False)
                n = self.max_imgs

            if self.bag_dropout > 0:
                max_drop = int(n * self.bag_dropout)
                if max_drop > 0:
                    rng    = np.random.default_rng(pseed)
                    n_drop = int(rng.integers(0, max_drop + 1))
                    n_keep = max(1, n - n_drop)
                    n_keep = min(n_keep, n)
                    if n_keep < n:
                        keep = np.random.default_rng(pseed + 1).choice(
                            n, n_keep, replace=False)
                        feat_idx = feat_idx[keep]

            result = {"images": feats[feat_idx], "label": label, "patient_id": pid}
            if tab_feats is not None:
                result["tab_feats"] = tab_feats
            return result

        # ── Slow path: load and preprocess images from disk ──────────────
        rows = self.patient2rows[pid]

        if self.max_imgs and len(rows) > self.max_imgs:
            rows = rows.sample(self.max_imgs, random_state=pseed)

        if self.bag_dropout > 0:
            max_drop = int(len(rows) * self.bag_dropout)
            if max_drop > 0:
                rng    = np.random.default_rng(pseed)
                n_drop = int(rng.integers(0, max_drop + 1))
                n_keep = max(1, len(rows) - n_drop)
                n_keep = min(n_keep, len(rows))
                if n_keep < len(rows):
                    rows = rows.sample(n_keep, random_state=pseed + 1)

        imgs = []
        for _, row in rows.iterrows():
            try:
                img = Image.open(row["full_path"]).convert("RGB")
            except Exception:
                self.corrupted_files.add(row["full_path"])
                print(f"  WARNING: skipping corrupted file: {row['full_path']}")
                continue
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        if len(imgs) == 0:
            imgs_tensor = torch.zeros(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE)
        else:
            imgs_tensor = torch.stack(imgs)                                # [N_imgs, 3, 224, 224]

        result = {"images": imgs_tensor, "label": label, "patient_id": pid}
        if tab_feats is not None:
            result["tab_feats"] = tab_feats
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. DATALOADER BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _make_image_loader(df: pd.DataFrame, train: bool) -> DataLoader:
    """
    Builds a DataLoader over EMImageDataset for image-level training or validation.
    Used internally by build_dataloaders().

    Args:
        df    : image-level DataFrame (train or val split)
        train : True → shuffle + augmentation; False → no shuffle, val transforms
    """
    ds = EMImageDataset(df, transform=get_transforms(train=train))         # image-level dataset
    return DataLoader(
        ds,
        batch_size  = cfg.BATCH_SIZE,                                      # images per batch
        shuffle     = train,                                               # shuffle only during training
        num_workers = cfg.NUM_WORKERS,                                     # parallel loading workers
        pin_memory  = True,                                                # faster CPU→GPU transfer
        drop_last   = train,                                               # drop last incomplete batch only during training
    )


def _make_patient_loader(df: pd.DataFrame, train: bool,
                         tab_features: dict = None) -> PatientMILDataset:
    """
    Builds a PatientMILDataset for patient-level training or validation.
    Returns the Dataset directly (not a DataLoader) because each patient bag
    has a variable number of images and must be iterated manually.

    Args:
        df           : image-level DataFrame (train or val split)
        train        : True → augmentation transforms + dropout; False → val transforms
        tab_features : optional {patient_id: np.ndarray} of ternary-encoded markers
    """
    if tab_features is not None:
        split_name = "train" if train else "val"
        missing = [str(pid) for pid in df["patient_id"].unique()
                   if str(pid) not in tab_features]
        if missing:
            raise ValueError(
                f"TAB: {len(missing)} {split_name} patients not found in TAB_FEATURES_CSV "
                f"(TAB_ID_COL='{cfg.TAB_ID_COL}'). "
                f"First missing: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        print(f"  TAB {split_name}: all {df['patient_id'].nunique()} patients found in tab_features.")

    return PatientMILDataset(
        df,
        transform             = get_transforms(train=train),
        max_imgs              = cfg.MAX_IMGS_PER_PATIENT if train else None,
        bag_dropout           = cfg.BAG_DROPOUT if train else 0.0,
        tab_features          = tab_features,
        tab_dropout           = getattr(cfg, "TAB_DROPOUT", 0.0) if train else 0.0,
        tab_full_dropout_prob = getattr(cfg, "TAB_FULL_DROPOUT_PROB", 0.0) if train else 0.0,
    )


def _compute_class_weights(train_df: pd.DataFrame, device: torch.device,
                           mode: str = "image") -> torch.Tensor:
    """
    Computes inverse-frequency class weights for the loss function.

    mode = "image"   : weights by image count — correct for image-level training
                        where every image contributes one gradient step
    mode = "patient" : weights by patient count — correct for patient-level MIL
                        where every patient contributes one gradient step regardless
                        of how many images they have
    """
    if mode == "patient":
        # One row per patient — count unique patients per class
        patient_df    = train_df.drop_duplicates(subset="patient_id")
        class_counts  = patient_df["label"].value_counts().sort_index().values
        count_label   = "patient"
    else:
        # Every image row counts — use raw image counts
        class_counts  = train_df["label"].value_counts().sort_index().values
        count_label   = "image"

    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float)
    class_weights = (class_weights / class_weights.sum()).to(device)

    print(f"\nClass weights for loss function (by {count_label} count):")
    for idx, w in enumerate(class_weights.cpu().numpy()):
        print(f"  [{idx}] {IDX_TO_CLASS[idx]}: {w:.4f}  ({class_counts[idx]} {count_label}s)")

    return class_weights


def build_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, device: torch.device,
                      tab_features: dict = None) -> dict:
    """
    Builds exactly the datasets and loaders needed for the active TRAINING_MODE.
    Reads cfg.TRAINING_MODE to decide what to build — nothing unused is created.

    Returns a dict with keys depending on the mode:

    TRAINING_MODE = "image":
        "train_loader"     – DataLoader[EMImageDataset] for image-level training
        "val_img_loader"   – DataLoader[EMImageDataset] for image-level validation
        "class_weights"    – [NUM_CLASSES] inverse-frequency weight tensor

    TRAINING_MODE = "patient":
        "train_patient_ds" – PatientMILDataset for patient-level training
        "val_patient_ds"   – PatientMILDataset for patient-level validation
        "class_weights"    – [NUM_CLASSES] inverse-frequency weight tensor

    Class weights are always computed from training image counts regardless of mode.
    """
    weight_mode   = "patient" if cfg.TRAINING_MODE == "patient" else "image"
    class_weights = _compute_class_weights(train_df, device, mode=weight_mode)

    if cfg.TRAINING_MODE == "image":
        # Approach 1: image-level throughout — no patient datasets needed at all
        print("\n  [DataLoader] Approach 1 (image-level): building image train + val loaders")
        return {
            "train_loader":   _make_image_loader(train_df, train=True),    # shuffled + augmented image batches
            "val_img_loader": _make_image_loader(val_df,   train=False),   # deterministic image batches
            "class_weights":  class_weights,
        }

    elif cfg.TRAINING_MODE == "patient":
        # Approach 2: patient-level throughout — no image-level loaders needed
        print("\n  [DataLoader] Approach 2 (patient-level): building patient train + val datasets")
        return {
            "train_patient_ds": _make_patient_loader(train_df, train=True,  tab_features=tab_features),
            "val_patient_ds":   _make_patient_loader(val_df,   train=False, tab_features=tab_features),
            "class_weights":    class_weights,
        }

    else:
        raise ValueError(
            f"Unknown TRAINING_MODE '{cfg.TRAINING_MODE}'. Choose: 'image', 'patient'."
        )