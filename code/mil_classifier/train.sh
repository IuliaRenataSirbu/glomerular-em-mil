#!/bin/bash
# Train one fold of a MIL experiment.
#
# Usage:
#   source ../../.env
#   bash train.sh configs/paths.yml configs/main_model.yml 0
#
# Arguments:
#   $1  paths config  (e.g. configs/paths.yml)
#   $2  experiment config (e.g. configs/main_model.yml)
#   $3  fold index 0-4
#
# SLURM users: adjust #SBATCH directives below for your cluster and
# uncomment the sbatch block instead of the direct python block.

PATHS_CFG="${1:-configs/paths.yml}"
EXP_CFG="${2:-configs/main_model.yml}"
FOLD="${3:-0}"

# ── Direct execution ──────────────────────────────────────────────────────────
python main.py --config "$PATHS_CFG" --config "$EXP_CFG" --fold "$FOLD"

# ── SLURM (uncomment and adapt) ───────────────────────────────────────────────
# RUN_NAME="$(basename "$EXP_CFG" .yml)_${FOLD}"
# sbatch \
#     -p "$SLURM_PARTITION" \
#     --mem=64G \
#     --gres=gpu:1 \
#     --job-name=mil \
#     --output="logs/train_%j_${RUN_NAME}.log" \
#     --wrap="python main.py --config $PATHS_CFG --config $EXP_CFG --fold $FOLD"
