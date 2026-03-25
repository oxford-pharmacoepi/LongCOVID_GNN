#!/bin/bash
# ==========================================================
# FULL TOURNAMENT — Multi-disease, multi-seed, multi-model
# ==========================================================
# Run overnight with: caffeinate -ims bash scripts/tournament.sh 2>&1 | tee results/tournament.log
#
# This runs:
# 1. SEAL (sage+JK, h=32) — best config — on 3 diseases × 3 seeds
# 2. GNN LOO (SAGE + Transformer) — on 3 diseases × 1 seed each
# 3. GNN LOO with MLP decoder — fairer comparison
# 4. Heuristics — on remaining 2 diseases (Osteo already done)
#
# Estimated runtime: ~20-24 hours
# ==========================================================

set -euo pipefail

# Auto-detect environment (Docker venv vs local uv)
if [ -n "${VIRTUAL_ENV:-}" ]; then
    PYTHON="python"
else
    PYTHON="uv run python"
fi

# Diseases
OSTEO="EFO_0003854"       # Osteoporosis (27 true drugs)
MS="EFO_0003929"          # Multiple Sclerosis (59 true drugs)
DEPRESSION="MONDO_0002009" # Depression (108 true drugs)

DISEASES=("$OSTEO" "$MS" "$DEPRESSION")
DISEASE_NAMES=("Osteoporosis" "Multiple_Sclerosis" "Depression")

SEEDS=(42 123 7)

echo "=========================================================="
echo "FULL TOURNAMENT"
echo "Started at: $(date -u)"
echo "=========================================================="
echo ""
echo "Diseases: ${DISEASE_NAMES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Python: $PYTHON"
echo ""

# Helper function
run_experiment() {
    local name=$1
    shift
    echo ""
    echo "============================================================"
    echo "[$(date -u '+%H:%M:%S')] $name"
    echo "  CMD: $@"
    echo "============================================================"
    PYTHONUNBUFFERED=1 "$@"
    echo "[$(date -u '+%H:%M:%S')] DONE: $name"
}

# ==========================================================
# PART 1: SEAL sage+JK on all diseases × all seeds
# ==========================================================
# NOTE: Osteoporosis seed=42 already done (sweep result: seal_EFO_0003854_20260218_235021.json)
echo ""
echo "########## PART 1: SEAL (sage+JK, h=32) ##########"

# Osteo — seeds 123 and 7 only (seed 42 already done)
for seed in 123 7; do
    run_experiment "SEAL Osteoporosis seed=${seed}" \
        $PYTHON scripts/seal/train_loo.py \
        --target-disease "$OSTEO" \
        --epochs 50 --hops 2 --workers 0 \
        --conv sage --pooling mean+max \
        --neg-ratio 3 --neg-strategy mixed --hard-ratio 0.5 \
        --use-jk --seed "$seed"
done

# MS and Depression — all 3 seeds
for i in 1 2; do
    disease="${DISEASES[$i]}"
    dname="${DISEASE_NAMES[$i]}"
    for seed in "${SEEDS[@]}"; do
        run_experiment "SEAL ${dname} seed=${seed}" \
            $PYTHON scripts/seal/train_loo.py \
            --target-disease "$disease" \
            --epochs 50 --hops 2 --workers 0 \
            --conv sage --pooling mean+max \
            --neg-ratio 3 --neg-strategy mixed --hard-ratio 0.5 \
            --use-jk --seed "$seed"
    done
done

# ==========================================================
# PART 2: GNN LOO — systematic hyperparameter exploration
# ==========================================================
# We give the GNN its best shot across:
#   - Models: SAGE, Transformer, GAT
#   - Decoders: mlp_interaction (default), dot, mlp_neighbor
#   - Hidden channels: 128 (default), 256
#   - Layers: 2 (default), 3
#   - Epochs: 50 (default), 100
# Not full Bayesian, but enough to be defensible in a paper.
echo ""
echo "########## PART 2: GNN LOO ##########"

for i in "${!DISEASES[@]}"; do
    disease="${DISEASES[$i]}"
    dname="${DISEASE_NAMES[$i]}"

    # --- A. Architecture comparison (default settings) ---
    run_experiment "GNN-SAGE ${dname}" \
        $PYTHON scripts/leave_one_out_validation.py \
        --target-node "$disease" --model SAGEModel --epochs 200 --no-mlflow

    run_experiment "GNN-Transformer ${dname}" \
        $PYTHON scripts/leave_one_out_validation.py \
        --target-node "$disease" --model TransformerModel --epochs 200 --no-mlflow

    run_experiment "GNN-GAT ${dname}" \
        $PYTHON scripts/leave_one_out_validation.py \
        --target-node "$disease" --model GAT --epochs 200 --no-mlflow

    # --- B. Decoder comparison (SAGE, best arch from Round 1) ---
    run_experiment "GNN-SAGE-dot ${dname}" \
        $PYTHON scripts/leave_one_out_validation.py \
        --target-node "$disease" --model SAGEModel --decoder-type dot --epochs 200 --no-mlflow

    run_experiment "GNN-SAGE-mlp_neighbor ${dname}" \
        $PYTHON scripts/leave_one_out_validation.py \
        --target-node "$disease" --model SAGEModel --decoder-type mlp_neighbor --epochs 200 --no-mlflow

    # --- C. Capacity & training budget ---
    run_experiment "GNN-SAGE-h256 ${dname}" \
        $PYTHON scripts/leave_one_out_validation.py \
        --target-node "$disease" --model SAGEModel --epochs 200 --no-mlflow \
        --override-config model_config.hidden_channels=256

    run_experiment "GNN-SAGE-3layers ${dname}" \
        $PYTHON scripts/leave_one_out_validation.py \
        --target-node "$disease" --model SAGEModel --layers 3 --epochs 200 --no-mlflow

    run_experiment "GNN-SAGE-300ep ${dname}" \
        $PYTHON scripts/leave_one_out_validation.py \
        --target-node "$disease" --model SAGEModel --epochs 300 --no-mlflow
done

# ==========================================================
# PART 3: Heuristics on remaining diseases
# ==========================================================
echo ""
echo "########## PART 3: Heuristics ##########"

# Osteoporosis already done — run MS and Depression
for i in 1 2; do
    disease="${DISEASES[$i]}"
    dname="${DISEASE_NAMES[$i]}"

    run_experiment "Heuristic ${dname}" \
        $PYTHON scripts/heuristic_baselines.py \
        --target-node "$disease" --no-mlflow
done

echo ""
echo "=========================================================="
echo "ALL TOURNAMENT EXPERIMENTS COMPLETE at $(date -u)"
echo "=========================================================="
