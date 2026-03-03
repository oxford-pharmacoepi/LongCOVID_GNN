#!/bin/bash
# ============================================================
# SEAL Hyperparameter Sweep — Osteoporosis (EFO_0003854)
# Runs 10 experiments sequentially, saves JSON results.
# ============================================================
set -e

DISEASE="EFO_0003854"
COMMON="--target-disease $DISEASE --epochs 50 --hops 2 --workers 0"

# Use bare python if a venv is active (Docker), otherwise uv run
if [ -n "$VIRTUAL_ENV" ]; then
    PYTHON="python"
else
    PYTHON="uv run python"
fi

echo "=========================================================="
echo "SEAL HYPERPARAMETER SWEEP — Osteoporosis"
echo "Started at: $(date -u)"
echo "=========================================================="

run_experiment() {
    local name=$1
    shift
    echo ""
    echo "=========================================================="
    echo "[$(date -u '+%H:%M:%S')] Experiment: $name"
    echo "  Args: $COMMON $@"
    echo "=========================================================="
    PYTHONUNBUFFERED=1 $PYTHON scripts/seal/train_loo.py $COMMON "$@"
    echo "[$(date -u '+%H:%M:%S')] DONE: $name"
}

# Baseline 
# run_experiment "baseline_sage" --conv sage --pooling mean+max --neg-ratio 3 --neg-strategy mixed --hard-ratio 0.5

# Round 1: Architecture
run_experiment "gat" --conv gat --pooling mean+max --neg-ratio 3 --neg-strategy mixed --hard-ratio 0.5
run_experiment "gin" --conv gin --pooling mean+max --neg-ratio 3 --neg-strategy mixed --hard-ratio 0.5
run_experiment "sage_jk" --conv sage --pooling mean+max --neg-ratio 3 --neg-strategy mixed --hard-ratio 0.5 --use-jk
run_experiment "gat_jk" --conv gat --pooling mean+max --neg-ratio 3 --neg-strategy mixed --hard-ratio 0.5 --use-jk

# Round 2: Training parameters
run_experiment "hidden64" --conv sage --pooling mean+max --neg-ratio 3 --neg-strategy mixed --hard-ratio 0.5 --hidden 64
run_experiment "neg5" --conv sage --pooling mean+max --neg-ratio 5 --neg-strategy mixed --hard-ratio 0.5
run_experiment "hard_only" --conv sage --pooling mean+max --neg-ratio 3 --neg-strategy hard
run_experiment "no_nodetypes" --conv sage --pooling mean+max --neg-ratio 3 --neg-strategy mixed --hard-ratio 0.5 --no-node-types

# Round 3: Subgraph config
run_experiment "hops3" --conv sage --pooling mean+max --neg-ratio 3 --neg-strategy mixed --hard-ratio 0.5 --hops 3
run_experiment "hops1" --conv sage --pooling mean+max --neg-ratio 3 --neg-strategy mixed --hard-ratio 0.5 --hops 1

echo ""
echo "=========================================================="
echo "ALL EXPERIMENTS COMPLETE at $(date -u)"
echo "=========================================================="
