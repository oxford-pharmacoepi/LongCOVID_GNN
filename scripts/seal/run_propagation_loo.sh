#!/usr/bin/env bash
# LOO validation on PROPAGATED graph (fill-gaps, max 50 descendants)
# Compares propagated graph performance vs baseline (no propagation)
#
# Run with: caffeinate -s bash scripts/seal/run_propagation_loo.sh 2>&1 | tee results/seal_propagation_loo.log
set -euo pipefail

BASE="uv run python -u scripts/seal/train_loo.py --conv sage --use-jk --hidden 32 --epochs 50 --hops 2 --neg-ratio 3 --neg-strategy mixed --hard-ratio 0.5 --seed 42 --no-mlflow"

echo "=============================================="
echo " PROPAGATION LOO STUDY started at $(date)"
echo "=============================================="

# 1. Osteoporosis — 54 drugs
echo ""
echo ">>> Osteoporosis LOO (propagated graph) at $(date)"
$BASE --target-disease EFO_0003854

# 2. Ankylosing Spondylitis — 44 drugs
echo ""
echo ">>> Ankylosing Spondylitis LOO (propagated graph) at $(date)"
$BASE --target-disease EFO_0003898

# 3. Cluster Headache — 0 → 34 drugs from propagation 
echo ""
echo ">>> Cluster Headache LOO (propagated graph) at $(date)"
$BASE --target-disease HP_0012199

echo ""
echo "=============================================="
echo " PROPAGATION LOO STUDY COMPLETE at $(date)"
echo "=============================================="
