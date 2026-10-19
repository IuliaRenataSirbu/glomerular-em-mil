#!/bin/bash
# Run inference for one trained fold checkpoint.
#
# Usage:
#   source ../../.env
#   bash test.sh checkpoints/main_model_0 test_results
#
# Arguments:
#   $1  run directory containing the checkpoint and config .yml
#   $2  output subdirectory name (created inside the run directory)

RUN_DIR="${1:?Usage: test.sh <run-dir> <out-subdir>}"
OUT_SUBDIR="${2:-test_results}"

# ── Direct execution ──────────────────────────────────────────────────────────
python test.py --run-dir "$RUN_DIR" --out-dir "$OUT_SUBDIR"

# ── SLURM (uncomment and adapt) ───────────────────────────────────────────────
# sbatch \
#     -p "$SLURM_PARTITION" \
#     --mem=32G \
#     --gres=gpu:1 \
#     --job-name=mil_test \
#     --output="logs/test_%j.log" \
#     --wrap="python test.py --run-dir $RUN_DIR --out-dir $OUT_SUBDIR"
