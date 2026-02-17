#!/bin/bash
# ============================================================
# Entrypoint for LongCOVID GNN Docker container
# Routes named commands to the correct Python scripts
# ============================================================
set -e

# Activate the uv virtual environment
source /app/.venv/bin/activate

case "$1" in
    # ── SEAL workflows ──────────────────────────────────────
    seal-loo)
        shift
        echo "Running SEAL Leave-One-Out validation..."
        python scripts/seal/train_loo.py "$@"
        ;;
    seal-lc)
        shift
        echo "Running SEAL Long COVID scoring..."
        python scripts/seal/score_long_covid.py "$@"
        ;;

    # ── Global GNN workflows ────────────────────────────────
    gnn-train)
        shift
        echo "Running GNN training..."
        python scripts/2_train_models.py "$@"
        ;;
    gnn-test)
        shift
        echo "Running GNN evaluation..."
        python scripts/3_test_evaluate.py "$@"
        ;;
    gnn-loo)
        shift
        echo "Running GNN Leave-One-Out validation..."
        python scripts/leave_one_out_validation.py "$@"
        ;;
    gnn-explain)
        shift
        echo "Running GNN explainability..."
        python scripts/4_explain_predictions.py "$@"
        ;;
    gnn-optimise)
        shift
        echo "Running hyperparameter optimisation..."
        python scripts/5_optimise_hyperparameters.py "$@"
        ;;
    gnn-repurpose)
        shift
        echo "Running Long COVID repurposing..."
        python scripts/6_long_covid_repurposing.py "$@"
        ;;

    # ── Benchmarks ──────────────────────────────────────────
    benchmark-heuristics)
        shift
        echo "Running GNN vs heuristics benchmark..."
        python scripts/benchmarks/gnn_vs_heuristics.py "$@"
        ;;
    benchmark-ablation)
        shift
        echo "Running GNN ablation study..."
        python scripts/benchmarks/gnn_ablation.py "$@"
        ;;

    # ── Data pipeline ───────────────────────────────────────
    create-graph)
        shift
        echo "Running graph creation..."
        python scripts/1_create_graph.py "$@"
        ;;

    # ── Direct Python execution ─────────────────────────────
    python)
        shift
        python "$@"
        ;;

    # ── Help ────────────────────────────────────────────────
    --help|-h|help)
        echo ""
        echo "LongCOVID GNN Docker Container"
        echo "=============================="
        echo ""
        echo "Usage: docker run <image> <command> [args...]"
        echo ""
        echo "Commands:"
        echo "  seal-loo         SEAL Leave-One-Out validation"
        echo "  seal-lc          SEAL Long COVID scoring"
        echo "  gnn-train        Train Global GNN models"
        echo "  gnn-test         Evaluate GNN models"
        echo "  gnn-loo          GNN Leave-One-Out validation"
        echo "  gnn-explain      GNN explainability (GNNExplainer)"
        echo "  gnn-optimise     Hyperparameter optimisation (Optuna)"
        echo "  gnn-repurpose    Long COVID drug repurposing"
        echo "  benchmark-*      Run benchmark scripts"
        echo "  create-graph     Create/process knowledge graph"
        echo "  python <script>  Run any Python script directly"
        echo ""
        echo "Examples:"
        echo "  docker run -v ./results:/app/results <image> seal-loo \\"
        echo "    --target-disease EFO_0003854 --epochs 50 --hops 2"
        echo ""
        echo "  docker run -v ./results:/app/results <image> gnn-loo"
        echo ""
        ;;

    # ── Fallback: try running as a command ──────────────────
    *)
        exec "$@"
        ;;
esac
