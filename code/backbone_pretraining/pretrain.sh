#!/bin/bash
# Pretrain DINOv2 ViT-S/14 on EM images using lightly-train.
#
# Usage:
#   source ../../.env
#   bash pretrain.sh configs/dinov2_morgagni.yml
#
# Arguments:
#   $1  pretraining config YAML (default: configs/dinov2_morgagni.yml)
#
# Hardware: pretraining used 8 GPUs with 500 GB RAM.
# Adapt #SBATCH directives for your cluster.

CONFIG="${1:-configs/dinov2_morgagni.yml}"

# ── Direct execution ──────────────────────────────────────────────────────────
python pretrain_dino_lightly.py --config "$CONFIG"

# ── SLURM multi-GPU (uncomment and adapt) ────────────────────────────────────
# sbatch \
#     -p "$SLURM_PARTITION" \
#     --mem=500G \
#     --cpus-per-task=128 \
#     --gres=gpu:8 \
#     --job-name=dino_pretrain \
#     --output="logs/pretrain_%j.log" \
#     --wrap="python pretrain_dino_lightly.py --config $CONFIG"
