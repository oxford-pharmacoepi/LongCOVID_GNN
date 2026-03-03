#!/usr/bin/env bash
# SEAL Ablation Studies — Edge types & Node features
# Run with: caffeinate -s bash scripts/seal/run_ablation.sh 2>&1 | tee results/seal_ablation.log
set -euo pipefail

BASE="uv run python -u scripts/seal/train_loo.py --target-disease EFO_0003854 --conv sage --use-jk --hidden 32 --epochs 50 --hops 2 --neg-ratio 3 --neg-strategy mixed --hard-ratio 0.5 --seed 42 --no-mlflow"

echo "=============================================="
echo " SEAL ABLATION STUDIES started at $(date)"
echo "=============================================="

# ─── A. Edge-type ablation ───────────────────────────────────────────
# A1: Baseline (full graph)

echo ""
echo ">>> A2: No PPI edges (gene-gene) at $(date)"
$BASE --exclude-edge-types ppi

echo ""
echo ">>> A3: No disease similarity edges at $(date)"
$BASE --exclude-edge-types disease-similarity

echo ""
echo ">>> A4: No drug-gene edges (MoA signal) at $(date)"
$BASE --exclude-edge-types drug-gene

echo ""
echo ">>> A5: No disease-gene edges at $(date)"
$BASE --exclude-edge-types disease-gene

# ─── B. Node feature ablation ───────────────────────────────────────
echo ""
echo ">>> B2: DRNL only — no node features, no node types at $(date)"
$BASE --no-node-features --no-node-types

echo ""
echo ">>> B3: DRNL + node types only — no original features at $(date)"
$BASE --no-node-features

echo "=============================================="
echo " SEAL ABLATION COMPLETE at $(date)"
echo "=============================================="
