# Flexible MIL for Glomerular Disease Classification from EM Images

Code for the paper:

> **Flexible multiple instance learning for classification of glomerular diseases based on kidney biopsy electron microscopy images**

## Overview

This repository contains:
- **`code/backbone_pretraining/`** — self-supervised DINOv2 ViT-S/14 pretraining on EM images
- **`code/mil_classifier/`** — full-image MIL classification pipeline with optional immunofluorescence (IF) marker fusion
- **`data/`** — placeholder data structure (see `data/README.md`; no patient data included)

## Requirements

**Local install:**
```bash
pip install -r requirements.txt
```

**HPC / Singularity (recommended for exact reproducibility):**

Two containers were used — one per stage:
```bash
# Backbone pretraining  (PyTorch 2.5.1, CUDA 12.4)
apptainer build environment_pretraining.sif environment_pretraining.def

# MIL classifier  (PyTorch 2.1.0, CUDA 12.1)
apptainer build environment_mil.sif environment_mil.def
```
Then run scripts with `apptainer exec --nv <sif> python ...`

Key dependencies: lightly-train ≥ 0.14, timm, scikit-learn, wandb.

## Setup

**1. Data.** Prepare your EM images and metadata files as described in `data/README.md`.

**2. Paths.** Copy and fill in the paths config:
```bash
cp code/mil_classifier/configs/paths.yml.example code/mil_classifier/configs/paths.yml
# edit paths.yml with your actual file paths
```

**3. Environment.** Copy and fill in `.env` for WandB and optional SLURM settings:
```bash
cp .env.example .env
# edit .env
source .env
```

## Backbone Pretraining

Pretrain DINOv2 ViT-S/14 on your EM images:
```bash
cd code/backbone_pretraining
# edit configs/dinov2_morgagni.yml — set 'data' to your images directory
python pretrain_dino_lightly.py --config configs/dinov2_morgagni.yml
```
The checkpoint at `epoch_500.pt` is what `VIT_CHECKPOINT` in `paths.yml` should point to.

Three pretraining configs are provided, corresponding to the ablation study in the paper:
- `dinov2_morgagni.yml` — Morgagni images only (main model)
- `dinov2_morgagni_ext.yml` — Morgagni + Tecnai (other disease classes)
- `dinov2_morgagni_all.yml` — Morgagni + Tecnai (all disease classes)

## MIL Classifier

### Training

Train one fold (replace `main_model` with any config name, `0` with fold index 0–4):
```bash
cd code/mil_classifier
python main.py \
    --config configs/paths.yml \
    --config configs/main_model_fold0.yml
```

Or use the shell script:
```bash
bash train.sh configs/paths.yml configs/main_model_fold0.yml 0
```

### Inference

Run inference on a trained fold:
```bash
python test.py --run-dir checkpoints/main_model_fold0 --out-dir test_results
```

### Ensemble evaluation

After training and testing all 5 folds:
```bash
python ensemble_cv_evaluation.py \
    --models checkpoints/main_model_fold0 \
             checkpoints/main_model_fold1 \
             checkpoints/main_model_fold2 \
             checkpoints/main_model_fold3 \
             checkpoints/main_model_fold4 \
    --out ensemble_results/main_model \
    --test-subdir test_results
```

### Confidence intervals

```bash
python bootstrap_ci.py \
    --models checkpoints/main_model_fold{0..4} \
    --out ensemble_results/main_model \
    --test-subdir test_results
```

### Explainability

Patient-level explainability (image importance, DINOv2 patch attention maps, marker sensitivity):
```bash
python explain.py \
    --run-dir checkpoints/main_model_fold0 \
    --folds data/folds.json \
    --split test \
    --top-k 5 \
    --levels 1 2 3a 3b \
    --out explain_results/main_model_fold0
```
Levels 3a and 3b (marker sensitivity / image-vs-marker split) are skipped automatically when `TAB_FUSION=false`.

### Composite attention figure

Multi-image attention composite plot (requires `explain.py` output):
```bash
python explain_attention_figure.py \
    --run-dir checkpoints/main_model_fold0 \
    --images images.json \
    --out figures/attention.jpg \
    --dpi 300
```
`images.json` is a list of `{"path": "...", "disease": "..."}` entries.

## Configs

Six experiment configs are provided (5 folds each), corresponding to the ablation table in the paper:

| Config | Description |
|--------|-------------|
| `main_model` | DINOv2 pretrained on Morgagni, finetuned, max-pool MIL, with IF |
| `ablation_no_if` | Same as main model without IF marker fusion |
| `ablation_frozen_encoder` | Same as main model with frozen image encoder |
| `ablation_facebook_weights` | DINOv2 with stock Facebook weights (no domain pretraining) |
| `ablation_random_init` | DINOv2 with random weight initialisation |
| `ablation_resnet50` | ResNet-50 backbone with random initialisation |

Configs for the pretraining dataset ablation (`dinov2_morgagni_ext`, `dinov2_morgagni_all`) will be added once the corresponding checkpoints are released.

All paths (`PROCESSED_JSON_PATH`, `FOLDS_PATH`, `TAB_FEATURES_CSV`, `VIT_CHECKPOINT`) are set in `configs/paths.yml` and are absent from the per-experiment configs so they do not need editing.

## Citation

This work is currently under review. Citation information will be added upon publication.
