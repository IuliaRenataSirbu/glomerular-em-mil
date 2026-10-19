# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration.  Defines all defaults; a YAML file can override any
# subset of them.  All other modules import `cfg` from here.
#
# Usage:
#     python main.py                          # use defaults from this file
#     python main.py --config configs/dino_frozen.yml   # override with YAML
#
# YAML format — only include the keys you want to change:
#
#     BACKBONE: dinov2_vits14
#     VIT_CHECKPOINT: checkpoints/dino/epoch_500.pt
#     FREEZE_BACKBONE: true
#     TRAINING_MODE: patient
#     FOLD: 0
#     PROJECT_NAME: kidney_dino_frozen_fold0
#
# Keys not present in the YAML keep their default values from this file.
# Path-type fields (DATA_ROOT, CSV_PATH, SAVE_DIR, VIT_CHECKPOINT) are
# automatically converted to Path objects after loading.
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import torch
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS  — edit these for your environment; YAML files override per-run
# ─────────────────────────────────────────────────────────────────────────────

class Config:

    # ── Paths ─────────────────────────────────────────────────────────────────
    DATA_ROOT = Path("/path/to/EMimages")                        # ← CHANGE THIS (or set PROCESSED_JSON_PATH)
    CSV_PATH  = None                                             # legacy scan path; set in configs/paths.yml if needed
    SAVE_DIR  = Path("checkpoints")
    FOLDS_PATH = None

    # Set this to use a pre-built JSON index instead of scanning DATA_ROOT.
    # When set, build_dataframe_v2(PROCESSED_JSON_PATH) is used in main.py.
    PROCESSED_JSON_PATH = None                                     # set in configs/paths.yml

    # ── Class Definitions ─────────────────────────────────────────────────────
    CLASSES = [
        "IgA nephropathy",
        "Alport syndrome/thin membrane disease",
        "Membranous nephropathy",
        "Diabetic kidney disease",
        "Minimal change disease (MCD)",
        "Lupus nephritis",
    ]
    NUM_CLASSES = len(CLASSES)                         # always in sync with CLASSES

    # ── Data ──────────────────────────────────────────────────────────────────
    IMG_SIZE             = 224
    # Augmentation preset:
    #   "v1"  — original: squish to 256×256, RandomCrop(224), HFlip, VFlip, ±15° rotation, ColorJitter(0.2)
    #   "v1a" — v1 with aspect-preserving resize (shorter edge → 256) instead of squish
    #   "v2"  — improved: aspect-preserving resize+crop, VFlip, discrete 90° rotation,
    #           ±15° jitter, GaussianBlur(p=0.5), ColorJitter(0.4)
    AUG_MODE             = "v2"
    MAX_IMGS_PER_PATIENT = 50                          # None = use all images
    # Bag dropout: randomly drop up to this fraction of a patient's images each
    # training epoch. Draws n_drop ~ Uniform(0, floor(n * BAG_DROPOUT)) so the
    # actual drop is an integer and small bags are never reduced below 1 image.
    # Applied only during training; val/test always use the full (capped) bag.
    # 0.0 = disabled (default).
    BAG_DROPOUT          = 0.0

    # ── Tabular Fusion ────────────────────────────────────────────────────────
    # Mid-level fusion: per-patient marker features are projected and concatenated
    # to the MIL patient embedding before the classification head.
    #
    # TAB_FUSION            : enable mid-level fusion (default False)
    # TAB_FEATURES_CSV      : path to markers CSV (required when TAB_FUSION=True)
    # TAB_ID_COL            : patient ID column in the markers CSV
    # TAB_MARKER_COLS       : columns to use; None = auto-detect *bin_corrected
    # TAB_PROJ_DIM          : projection layer output dim before concatenation
    # TAB_DROPOUT           : per-marker zero-out probability during training
    # TAB_FULL_DROPOUT_PROB : probability of zeroing all markers for a patient
    TAB_FUSION            = False
    TAB_FEATURES_CSV      = None
    TAB_ID_COL            = "ID"
    TAB_MARKER_COLS       = None
    TAB_PROJ_DIM          = 64
    TAB_DROPOUT           = 0.0
    TAB_FULL_DROPOUT_PROB = 0.0

    # ── Training Mode ─────────────────────────────────────────────────────────
    #   "image"   — image-level training + image-level val
    #   "patient" — patient-level MIL training + patient-level val
    TRAINING_MODE = "patient"

    # ── General Training ──────────────────────────────────────────────────────
    PRETRAINED   = True                                # ImageNet weights for ResNet50

    # ── Loss function ─────────────────────────────────────────────────────────
    #   "ce"    — standard weighted CrossEntropyLoss (default)
    #   "focal" — FocalLoss, reduces weight on easy examples (good for imbalance)
    LOSS        = "ce"
    FOCAL_GAMMA = 2.0                                  # focusing parameter (only used when LOSS="focal")
    EPOCHS       = 30
    BATCH_SIZE   = 32
    LR           = 1e-4
    # Per-component backbone LR override.
    #   None (default) = auto-scale: ViT/DINOv2 → LR×0.01, ResNet50 → LR×0.1
    #   Set to a float to override, e.g. BACKBONE_LR: 3e-4 for uniform LRs or
    #   BACKBONE_LR: 3e-6 to pin an exact value regardless of LR.
    # Has no effect when FREEZE_BACKBONE=True (backbone gradients are disabled).
    BACKBONE_LR  = None
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP        = 1.0
    # Metric used to select the best checkpoint during training.
    #   "auc"      — macro-averaged one-vs-rest AUC (default)
    #   "acc"      — accuracy
    #   "bacc"     — balanced accuracy (recommended for class imbalance)
    #   "f1_macro" — macro F1
    CHECKPOINT_METRIC = "auc"
    # Accumulate gradients over this many patients before each optimizer step.
    # 1 = step every patient (default, no accumulation).
    # 4–8 = effective batch of 4–8 patients, stabilises noisy per-patient gradients.
    GRAD_ACCUM_STEPS = 1
    AMP              = True

    # ── Backbone ──────────────────────────────────────────────────────────────
    #   "resnet50"      — ImageNet-pretrained ResNet50
    #   "dinov2_vits14" — DINOv2 ViT-S/14 (main model, requires VIT_CHECKPOINT)
    BACKBONE        = "resnet50"
    # VIT_CHECKPOINT controls weight init for dinov2_vits14:
    #   Path to *.pt   — custom pretrained weights (lightly export)
    #   "facebook"     — official DINOv2 SSL weights from Facebook (LVD-142M)
    #   "random"       — random init, no pretraining at all  (ablation)
    VIT_CHECKPOINT  = None                                       # set in configs/paths.yml
    FREEZE_BACKBONE = False                            # True = frozen encoder, False = fine-tuned

    # ── MIL Aggregation ───────────────────────────────────────────────────────
    #   "mean" | "max" | "attention"
    MIL_MODE = "max"

    # ── Cross-Validation ──────────────────────────────────────────────────────
    N_FOLDS = 5
    FOLD    = 0
    SEED    = 42

    # ── Logging ───────────────────────────────────────────────────────────────
    #   "wandb" | "tensorboard" | "none"
    LOGGER       = "wandb"
    PROJECT_NAME = "kidney_em_mil"                     # shared across experiments — groups runs in wandb
    RUN_NAME     = "default_run"                       # unique per experiment — used as wandb run name
                                                       # and checkpoint subdir: SAVE_DIR / RUN_NAME

    # ── Hardware ──────────────────────────────────────────────────────────────
    NUM_WORKERS = 4
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# YAML LOADING  — overrides any subset of the defaults above
# ─────────────────────────────────────────────────────────────────────────────

# Fields that should be converted to Path objects after loading
_PATH_FIELDS = {"DATA_ROOT", "CSV_PATH", "SAVE_DIR", "VIT_CHECKPOINT", "PROCESSED_JSON_PATH",
                "TAB_FEATURES_CSV"}


def _load_yaml(path: str) -> dict:
    """Load a YAML file and return its contents as a flat dict."""
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for YAML config files.  "
            "Install it with:  pip install pyyaml"
        )
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data


def _apply_overrides(cfg_obj: Config, overrides: dict) -> Config:
    """
    Apply a dict of overrides onto a Config instance.
    Unknown keys raise an error so typos are caught immediately.
    Path fields are converted to Path objects automatically.
    """
    valid_keys = {k for k in vars(Config) if not k.startswith("_")}
    # Also include class-level attributes (not just instance attrs)
    valid_keys |= {k for k in dir(Config)
                   if not k.startswith("_") and not callable(getattr(Config, k))}

    for key, value in overrides.items():
        if key not in valid_keys:
            raise ValueError(
                f"Unknown config key '{key}' in YAML file.\n"
                f"Valid keys: {sorted(valid_keys)}"
            )
        if key in _PATH_FIELDS and value is not None:
            # VIT_CHECKPOINT can be a keyword string — don't convert to Path:
        
            #   "facebook"  — Facebook DINOv2 SSL weights via torch.hub (dinov2_vits14 only)
            #   "random"    — random init (any backbone)
            _vit_keywords = {"imagenet", "facebook", "random"}
            if not (key == "VIT_CHECKPOINT" and str(value) in _vit_keywords):
                value = Path(value)

        # YAML parses scientific notation like 3e-4 as a string.
        # If the default value for this key is a float/int, coerce it.
        default_val = getattr(Config, key, None)
        if isinstance(default_val, float) and isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        elif isinstance(default_val, int) and isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                pass
        elif default_val is None and isinstance(value, str):
            # Coerce string values for None-defaulted fields that may be numeric (e.g. BACKBONE_LR)
            try:
                value = float(value)
            except ValueError:
                pass

        setattr(cfg_obj, key, value)

    # Keep NUM_CLASSES in sync if CLASSES was overridden
    if "CLASSES" in overrides:
        cfg_obj.NUM_CLASSES = len(cfg_obj.CLASSES)

    if cfg_obj.NUM_CLASSES != len(cfg_obj.CLASSES):
        raise ValueError(
            f"NUM_CLASSES={cfg_obj.NUM_CLASSES} does not match len(CLASSES)={len(cfg_obj.CLASSES)}. "
            f"Do not set NUM_CLASSES directly — it is derived from CLASSES automatically."
        )

    return cfg_obj


def _parse_config_arg():
    """
    Extract --config <path> from sys.argv without consuming other arguments.
    Returns the path string, or None if --config was not provided.

    We parse only --config here so that main.py (and other scripts) can still
    define their own argparse arguments freely.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None)
    known, _ = parser.parse_known_args()
    return known.config


def _build_cfg() -> Config:
    """Build the global cfg object, applying any YAML overrides."""
    cfg_obj = Config()

    config_path = _parse_config_arg()
    if config_path is not None:
        overrides = _load_yaml(config_path)
        cfg_obj   = _apply_overrides(cfg_obj, overrides)
        print(f"  Config loaded from: {config_path}")
        if overrides:
            max_key_len = max(len(k) for k in overrides)
            print(f"  Overrides ({len(overrides)}):")
            for k, v in overrides.items():
                print(f"    {k:<{max_key_len}} = {v}")
        print()

    return cfg_obj


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL INSTANCE  — imported by every other module
# ─────────────────────────────────────────────────────────────────────────────

cfg = _build_cfg()