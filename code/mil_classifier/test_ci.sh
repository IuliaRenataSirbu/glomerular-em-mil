#!/bin/bash
# Compute bootstrap confidence intervals for an ensemble result.
#
# Usage:
#   source ../../.env
#   bash test_ci.sh main_model test_results
#
# Arguments:
#   $1  common checkpoint name prefix (folds 0-4 appended automatically)
#   $2  test output subdirectory name inside each fold checkpoint
#
# Run AFTER test_ensemble.sh — requires ensemble_test_predictions.csv to exist.

CKPT_NAME_COMMON="${1:?Usage: test_ci.sh <ckpt-name-common> <test-subdir>}"
TEST_SUBDIR="${2:-test_results}"

MODELS=$(for i in 0 1 2 3 4; do echo "checkpoints/${CKPT_NAME_COMMON}_${i}"; done)
OUT_DIR="ensemble_results/${CKPT_NAME_COMMON}"

# ── Direct execution ──────────────────────────────────────────────────────────
python bootstrap_ci.py \
    --models $MODELS \
    --out "$OUT_DIR" \
    --test-subdir "$TEST_SUBDIR"

# ── SLURM (uncomment and adapt) ───────────────────────────────────────────────
# sbatch \
#     -p "$SLURM_PARTITION" \
#     --mem=32G \
#     --gres=gpu:1 \
#     --job-name=mil_ci \
#     --output="logs/ci_%j.log" \
#     --wrap="python bootstrap_ci.py --models $MODELS --out $OUT_DIR --test-subdir $TEST_SUBDIR"
