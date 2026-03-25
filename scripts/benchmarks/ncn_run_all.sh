#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
# NCN FULL PIPELINE — Run Everything in One Go
# ══════════════════════════════════════════════════════════════════════════
#
# This script runs ALL NCN experiments for the paper:
#   Phase 1: Hyperparameter sweep (Osteoporosis)
#   Phase 2: Best config across all diseases × 3 seeds (tournament)
#   Phase 3: Ablation studies (edge-type + node-feature)
#   Phase 4: COVID-19 RCT failed drug analysis
#   Phase 5: Long COVID predictions (gene configs + 5-seed final)
#
# Estimated total runtime: ~30-60 minutes (vs SEAL's ~20+ hours)
#
# Usage:
#   chmod +x scripts/benchmarks/ncn_run_all.sh
#   caffeinate -ims bash scripts/benchmarks/ncn_run_all.sh 2>&1 | tee results/ncn_full_pipeline.log
#
# ══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# Auto-detect environment
if [ -n "${VIRTUAL_ENV:-}" ]; then
    PYTHON="python"
else
    PYTHON="uv run python"
fi

OSTEO="EFO_0003854"
MS="EFO_0003929"
DEPRESSION="MONDO_0002009"
DEMENTIA="HP_0000726"
COVID="MONDO_0100096"

DISEASES=("$OSTEO" "$MS" "$DEPRESSION" "$DEMENTIA")
DISEASE_NAMES=("Osteoporosis" "Multiple_Sclerosis" "Depression" "Dementia")
SEEDS=(42 123 7)

LOO_SCRIPT="scripts/benchmarks/ncnc_loo.py"
LC_SCRIPT="scripts/benchmarks/ncn_predict_long_covid.py"

mkdir -p results/ncnc_results results/ncn_long_covid results/ncn_logs

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           NCN FULL PIPELINE — ALL EXPERIMENTS               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Start: $(date)"
echo "Python: $PYTHON"
echo ""

run_exp() {
    local name=$1
    shift
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$(date +%H:%M:%S)] $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    PYTHONUNBUFFERED=1 "$@"
    echo "[$(date +%H:%M:%S)] ✓ DONE: $name"
}


# ══════════════════════════════════════════════════════════════════════════
# PHASE 1: HYPERPARAMETER SWEEP ON OSTEOPOROSIS
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "########## PHASE 1: HYPERPARAMETER SWEEP (Osteoporosis) ##########"

COMMON="--target-disease $OSTEO --epochs 50 --patience 10 --seed 42"

# A. Hidden dimension
for hidden in 32 64 128; do
    run_exp "Sweep hidden=$hidden" \
        $PYTHON $LOO_SCRIPT $COMMON --hidden $hidden --gnn-layers 1
done

# B. GNN layers
for layers in 1 2 3; do
    run_exp "Sweep layers=$layers" \
        $PYTHON $LOO_SCRIPT $COMMON --hidden 64 --gnn-layers $layers
done

# C. Beta (CN weight)
for beta in 0.5 1.0 2.0; do
    run_exp "Sweep beta=$beta" \
        $PYTHON $LOO_SCRIPT $COMMON --hidden 64 --gnn-layers 1 --beta $beta
done

# D. JK and residual
run_exp "Sweep JK" \
    $PYTHON $LOO_SCRIPT $COMMON --hidden 64 --gnn-layers 1 --use-jk

run_exp "Sweep no-res" \
    $PYTHON $LOO_SCRIPT $COMMON --hidden 64 --gnn-layers 1 --no-res

# E. Learning rates
for lr in 0.001 0.003 0.01; do
    run_exp "Sweep lr=$lr" \
        $PYTHON $LOO_SCRIPT $COMMON --hidden 64 --gnn-layers 1 --gnn-lr $lr --pred-lr $lr
done

# F. Dropout
for dp in 0.1 0.3 0.5; do
    run_exp "Sweep dropout=$dp" \
        $PYTHON $LOO_SCRIPT $COMMON --hidden 64 --gnn-layers 1 --dropout $dp
done

# G. Neg ratio
for neg in 1 3 5; do
    run_exp "Sweep neg_ratio=$neg" \
        $PYTHON $LOO_SCRIPT $COMMON --hidden 64 --gnn-layers 1 --neg-ratio $neg
done

echo ""
echo "=== PHASE 1 COMPLETE at $(date) ==="
echo ">>> Review results/ncnc_results/ncn_EFO_0003854_*.json to pick best config"
echo ""


# ══════════════════════════════════════════════════════════════════════════
# PHASE 2: DISEASE TOURNAMENT — BEST CONFIG × ALL DISEASES × 3 SEEDS
# ══════════════════════════════════════════════════════════════════════════
echo "########## PHASE 2: DISEASE TOURNAMENT ##########"
# Using hidden=64, layers=1, beta=1.0 as default best config
# (if sweep reveals different best, update these values and rerun)

BEST="--hidden 64 --gnn-layers 1 --beta 1.0 --epochs 50 --patience 10"

for i in "${!DISEASES[@]}"; do
    disease="${DISEASES[$i]}"
    dname="${DISEASE_NAMES[$i]}"
    for seed in "${SEEDS[@]}"; do
        logfile="results/ncn_logs/tournament_${dname}_seed${seed}.log"
        run_exp "Tournament: $dname seed=$seed" \
            $PYTHON $LOO_SCRIPT --target-disease "$disease" $BEST --seed "$seed" \
            > "$logfile" 2>&1 || true
        grep "Hits@100:" "$logfile" 2>/dev/null | head -1 || echo "  (check log)"
    done
done

echo ""
echo "=== PHASE 2 COMPLETE at $(date) ==="


# ══════════════════════════════════════════════════════════════════════════
# PHASE 3: ABLATION STUDIES (OSTEOPOROSIS, SEED=42)
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "########## PHASE 3: ABLATION STUDIES ##########"

ABLATION_BASE="--target-disease $OSTEO $BEST --seed 42"

# A. Edge-type ablation
run_exp "Ablation: No PPI" \
    $PYTHON $LOO_SCRIPT $ABLATION_BASE --exclude-edge-types ppi

run_exp "Ablation: No drug-gene" \
    $PYTHON $LOO_SCRIPT $ABLATION_BASE --exclude-edge-types drug-gene

run_exp "Ablation: No disease-gene" \
    $PYTHON $LOO_SCRIPT $ABLATION_BASE --exclude-edge-types disease-gene

run_exp "Ablation: No disease-similarity" \
    $PYTHON $LOO_SCRIPT $ABLATION_BASE --exclude-edge-types disease-similarity

# B. Node feature ablation
run_exp "Ablation: No node features (types only)" \
    $PYTHON $LOO_SCRIPT $ABLATION_BASE --no-node-features

run_exp "Ablation: No features, no types" \
    $PYTHON $LOO_SCRIPT $ABLATION_BASE --no-node-features --no-node-types

echo ""
echo "=== PHASE 3 COMPLETE at $(date) ==="


# ══════════════════════════════════════════════════════════════════════════
# PHASE 4: COVID-19 RCT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "########## PHASE 4: COVID-19 RCT ANALYSIS ##########"

for seed in "${SEEDS[@]}"; do
    run_exp "RCT COVID seed=$seed" \
        $PYTHON $LOO_SCRIPT --target-disease "$COVID" $BEST --seed "$seed" \
        --failed-rcts failed_rcts.txt
done

echo ""
echo "=== PHASE 4 COMPLETE at $(date) ==="


# ══════════════════════════════════════════════════════════════════════════
# PHASE 5: LONG COVID PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "########## PHASE 5: LONG COVID PREDICTIONS ##########"

LC_BASE="--hidden 64 --gnn-layers 1 --epochs 50 --patience 10 --top-k 100"

# A. Gene configuration comparison (seed=123, matching SEAL analysis)
run_exp "LC NARROW genes" \
    $PYTHON $LC_SCRIPT $LC_BASE --seed 123 \
    --gene-categories gwas --max-drug-per-gene 5

run_exp "LC BROAD genes" \
    $PYTHON $LC_SCRIPT $LC_BASE --seed 123 \
    --gene-categories gwas,cat1,cat2,cat3 --max-drug-per-gene 5

run_exp "LC FULL genes" \
    $PYTHON $LC_SCRIPT $LC_BASE --seed 123 \
    --gene-categories all

# B. Hub threshold sweep (NARROW, seed=123)
for cap in 3 5 10; do
    run_exp "LC NARROW hub_cap=$cap" \
        $PYTHON $LC_SCRIPT $LC_BASE --seed 123 \
        --gene-categories gwas --max-drug-per-gene $cap
done

run_exp "LC NARROW no_hub_cap" \
    $PYTHON $LC_SCRIPT $LC_BASE --seed 123 \
    --gene-categories gwas

# C. 5-seed production run (NARROW, best hub cap)
echo ""
echo "--- 5-SEED FINAL PRODUCTION RUN ---"
for seed in 7 42 123 456 789; do
    run_exp "LC FINAL seed=$seed" \
        $PYTHON $LC_SCRIPT $LC_BASE --seed $seed \
        --gene-categories gwas --max-drug-per-gene 5
done

echo ""
echo "=== PHASE 5 COMPLETE at $(date) ==="


# ══════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ALL NCN EXPERIMENTS COMPLETE at $(date)  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Results locations:"
echo "  LOO results:     results/ncnc_results/"
echo "  LC predictions:  results/ncn_long_covid/"
echo "  Run logs:        results/ncn_logs/"
echo ""
echo "Next: compile results into RESULTS_ALL_NCN.md"
