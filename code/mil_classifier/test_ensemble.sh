#!/bin/bash
# Aggregate 5-fold checkpoints into ensemble predictions and figures.
#
# Usage:
#   source ../../.env
#   bash test_ensemble.sh main_model test_results
#
# Arguments:
#   $1  common checkpoint name prefix (folds 0-4 appended automatically)
#   $2  test output subdirectory name inside each fold checkpoint

CKPT_NAME_COMMON="${1:?Usage: test_ensemble.sh <ckpt-name-common> <test-subdir>}"
TEST_SUBDIR="${2:-test_results}"

MODELS=$(for i in 0 1 2 3 4; do echo "checkpoints/${CKPT_NAME_COMMON}_${i}"; done)
OUT_DIR="ensemble_results/${CKPT_NAME_COMMON}"

# ── Direct execution ──────────────────────────────────────────────────────────
python ensemble_cv_evaluation.py \
    --models $MODELS \
    --out "$OUT_DIR" \
    --test-subdir "$TEST_SUBDIR"

# ── SLURM (uncomment and adapt) ───────────────────────────────────────────────
# sbatch \
#     -p "$SLURM_PARTITION" \
#     --mem=32G \
#     --gres=gpu:1 \
#     --job-name=mil_ensemble \
#     --output="logs/ensemble_%j.log" \
#     --wrap="python ensemble_cv_evaluation.py --models $MODELS --out $OUT_DIR --test-subdir $TEST_SUBDIR"
