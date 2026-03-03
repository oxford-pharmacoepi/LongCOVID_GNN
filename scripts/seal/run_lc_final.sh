#!/usr/bin/env bash
# Final 5-seed SEAL production run for Long COVID
# NARROW gene config (8 GWAS leads) — confirmed winner from gene config comparison
# SAGEConv + JK, h=32, hops=2 — confirmed winner from architecture sweep
#
# Estimated runtime: ~5–8 hours total (~1–1.5 hours per seed)
# Run with: caffeinate -s bash scripts/seal/run_lc_final.sh 2>&1 | tee results/seal_final_5seed.log
set -euo pipefail

BASE="uv run python -u scripts/seal/predict_long_covid.py --conv sage --use-jk --hidden 32 --epochs 50 --hops 2 --neg-ratio 3 --neg-strategy mixed --hard-ratio 0.5 --max-drug-per-gene 5 --top-k 100 --gene-categories gwas"

echo "=============================================="
echo " FINAL 5-SEED SEAL PRODUCTION RUN at $(date)"
echo " Config: SAGEConv + JK, h=32, NARROW (gwas)"
echo "=============================================="

for SEED in 7 42 123 456 789; do
    echo ""
    echo ">>> SEAL NARROW seed=${SEED} at $(date)"
    echo "=============================================="
    $BASE --seed $SEED
done

echo ""
echo "=============================================="
echo " FINAL 5-SEED RUN COMPLETE at $(date)"
echo "=============================================="
echo ""
echo "Next step: aggregate results with"
echo "  uv run python scripts/seal/aggregate_lc_results.py"
