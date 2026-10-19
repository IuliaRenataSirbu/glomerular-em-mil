#!/bin/bash
# Generate DINOv2 attention maps and the attention figure for one fold.
#
# Usage:
#   source ../../.env
#   bash test_explain.sh checkpoints/main_model_0 images.json explain_results/main_model_0
#
# Arguments:
#   $1  run directory containing the checkpoint and config .yml
#   $2  JSON file listing images to explain (see explain_attention_figure.py)
#   $3  output directory for the attention figure

RUN_DIR="${1:?Usage: test_explain.sh <run-dir> <images-json> <out-dir>}"
IMAGES_JSON="${2:?}"
OUT_DIR="${3:-explain_results}"

# ── Direct execution ──────────────────────────────────────────────────────────
python explain_attention_figure.py \
    --run-dir  "$RUN_DIR" \
    --images   "$IMAGES_JSON" \
    --out      "${OUT_DIR}/figure_attention.jpg" \
    --dpi      300 \
    --font-size 9

# ── SLURM (uncomment and adapt) ───────────────────────────────────────────────
# sbatch \
#     -p "$SLURM_PARTITION" \
#     --mem=32G \
#     --gres=gpu:1 \
#     --job-name=mil_explain \
#     --output="logs/explain_%j.log" \
#     --wrap="python explain_attention_figure.py \
#         --run-dir $RUN_DIR --images $IMAGES_JSON \
#         --out ${OUT_DIR}/figure_attention.jpg --dpi 300 --font-size 9"
