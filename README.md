# Long COVID Drug Repurposing with Graph Neural Networks

Drug repurposing framework combining **SEAL** (Subgraph Extraction and Link Prediction) and **Global GNN** architectures for drug–disease link prediction on biomedical knowledge graphs. Special focus on Long COVID treatment discovery using Open Targets data.

## Overview

This project systematically evaluates SEAL against global GNN architectures (GAT, GraphSAGE, Transformer) and heuristic baselines (Common Neighbours, Jaccard, Adamic-Adar) for drug repurposing, using leave-one-out validation, temporal validation, and ablation studies.

**Key Capabilities:**
- **SEAL model** with SAGEConv+JK for local subgraph-based link prediction
- **Global GNNs**: GAT, GraphSAGE, GCN, Transformer (8 architecture variants)
- Temporal validation using time-stamped Open Targets releases (2021, 2023, 2024)
- Leave-one-out (LOO) validation across multiple diseases
- Multi-seed consensus predictions (5 seeds) for Long COVID
- Multiple negative sampling strategies (random, hard, mixed)
- Bayesian hyperparameter optimisation with Optuna
- Experiment tracking with MLflow
- Dockerised for reproducible deployment on GCP

## Key Results

- **SEAL** excels for focused diseases (Osteoporosis: Hits@50 = 78%, median rank 24/2,471)
- **GAT** outperforms SEAL for broadly-treated diseases (Depression: Hits@100 = 34% vs 16%)
- Graph topology (drug–gene–disease pathway) is more predictive than node features or PPI
- Temporal validation: future drug–disease links placed in top 13% on average
- Long COVID: 96 consensus drug candidates from 5-seed ensemble

## Project Structure

```
LongCOVID_GNN/
├── src/                              # Core framework modules
│   ├── config.py                     # Centralised configuration
│   ├── models.py                     # Global GNN architectures (GCN, SAGE, GAT, Transformer)
│   ├── models_seal.py                # SEAL model (SAGEConv/GATConv/GINConv + JK)
│   ├── negative_sampling.py          # Negative sampling strategies
│   ├── data_processing.py            # Data loading and preprocessing
│   ├── data/                         # Data loaders, filters, mappers, storage
│   ├── features/                     # Node and edge feature engineering
│   ├── graph/                        # Graph construction, edge extraction, splitting
│   ├── training/                     # Loss functions, optimiser, MLflow tracker
│   └── utils/                        # Evaluation, graph, feature, edge utilities
│
├── scripts/                          # Pipeline scripts
│   ├── 1_create_graph.py             # Knowledge graph construction
│   ├── 2_train_models.py             # Global GNN model training
│   ├── 3_test_evaluate.py            # Testing and evaluation
│   ├── 4_explain_predictions.py      # GNNExplainer predictions
│   ├── 5_optimise_hyperparameters.py # Bayesian hyperparameter optimisation
│   ├── 6_long_covid_repurposing.py   # Global GNN Long COVID predictions
│   ├── create_paper_plots.py         # Publication figure generation
│   ├── heuristic_baselines.py        # Heuristic baseline evaluation
│   ├── leave_one_out_validation.py   # Global GNN LOO validation
│   ├── tournament.sh                 # Full model tournament runner
│   ├── seal/                         # SEAL pipeline
│   │   ├── train_loo.py              # SEAL LOO validation
│   │   ├── train_global.py           # SEAL global training
│   │   ├── train_temporal.py         # SEAL temporal validation
│   │   ├── predict_long_covid.py     # SEAL Long COVID predictions
│   │   ├── aggregate_lc_results.py   # Multi-seed consensus aggregation
│   │   ├── score_long_covid.py       # RCT drug scoring
│   │   └── visualise.py              # SEAL result visualisation
│   ├── benchmarks/                   # Cross-model benchmarking scripts
│   ├── tools/                        # Utility scripts (enrichment, embeddings)
│   └── setup/                        # Data download and folder creation
│
├── docker/                           # Docker deployment
│   ├── Dockerfile                    # Container configuration (CUDA 12.1, Python 3.12)
│   └── entrypoint.sh                 # Container entry point
│
├── tests/                            # Unit tests (409 tests, 81% coverage)
├── processed_data/                   # Pre-processed Open Targets datasets
├── results/                          # Outputs (models, figures, evaluation)
├── gwas_genes_long_covid.txt         # Long COVID GWAS gene list (57 genes, 8 categories)
├── run_pipeline.py                   # Full pipeline orchestrator
└── pyproject.toml                    # Dependencies (uv)
```

### Two Pipelines

The repository contains **two complementary pipelines**:

1. **Global GNN Pipeline** (numbered scripts `1–6`): Traditional message-passing GNNs that learn node embeddings over the entire graph and predict links via inner-product decoders.
2. **SEAL Pipeline** (`scripts/seal/`): Subgraph-based approach that extracts enclosing subgraphs around candidate (drug, disease) pairs, labels nodes with Double-Radius Node Labelling (DRNL), and classifies subgraphs directly.

Both pipelines share the same graph (`scripts/1_create_graph.py`) and source modules in `src/`.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

### Prerequisites
- Python 3.12+
- uv package manager

### Setup

```bash
git clone <repository-url>
cd LongCOVID_GNN
uv sync
```

### Docker (GPU)

```bash
docker build -f docker/Dockerfile -t longcovid-gnn .
docker run --gpus all longcovid-gnn
```

## Quick Start

### Graph Construction

```bash
# Build knowledge graph from processed Open Targets data
uv run python scripts/1_create_graph.py
```

**Graph statistics:** 31,902 nodes (2,471 drugs, ~5,000 diseases, ~20,000 genes) and 1,068,588 edges across 6 edge types (drug–gene MoA, disease–gene, gene–gene PPI, drug–disease indication, disease similarity, drug–drug similarity).

### SEAL Leave-One-Out Validation

```bash
# Run SEAL LOO for Osteoporosis
uv run python scripts/seal/train_loo.py --disease EFO_0003854 --seed 42
```

### Long COVID Drug Repurposing (SEAL)

```bash
# Predict drug candidates for Long COVID (multi-seed consensus)
for seed in 7 42 123 456 789; do
    uv run python scripts/seal/predict_long_covid.py \
        --gene-categories gwas --max-drug-per-gene 5 --seed $seed
done

# Aggregate into consensus predictions (≥3/5 seeds)
uv run python scripts/seal/aggregate_lc_results.py \
    --seeds 7 42 123 456 789 --include-gat --min-seeds 3
```

### Global GNN Training & Evaluation

```bash
# Train all Global GNN models
uv run python scripts/2_train_models.py

# Evaluate on test set
uv run python scripts/3_test_evaluate.py

# Long COVID predictions with Global GNN
uv run python scripts/6_long_covid_repurposing.py --top-k 100 --lookup-names
```

## Model Architectures

### SEAL
- Subgraph extraction with Double-Radius Node Labelling (DRNL)
- SAGEConv with Jumping Knowledge (JK) aggregation
- Mean + max graph pooling for subgraph classification
- Best for: Focused diseases with specific drug–gene pathways

### GAT (Graph Attention Network)
- Multi-head attention over node neighbourhoods
- Strongest Global GNN in the tournament
- Best for: Broadly-treated diseases with diverse pharmacology

### GraphSAGE / GCN / Transformer
- Additional Global GNN variants evaluated in the model tournament
- See `results/FULL_TOURNAMENT_REPORT.md` for complete comparison

## Temporal Validation

Three Open Targets versions ensure models generalise to future discoveries:

1. **Training (21.06):** 9,425 drug–disease associations
2. **Validation (23.06):** 336 associations for hyperparameter tuning
3. **Test (24.06):** 83 associations across 56 diseases for final evaluation

SEAL achieves test AUC = 0.915 and places future links in the top 13% of drugs on average.

## Configuration

All settings are in `src/config.py`. Key parameters:

| Parameter | Default | Description |
|:---|:---|:---|
| `hidden_channels` | 32 | SEAL hidden dimension |
| `num_layers` | 3 | Convolution layers |
| `model_choice` | `'SAGE'` | GNN architecture |
| `epochs` | 50 | Training epochs |
| `negative_sampling_strategy` | `'mixed'` | Sampling strategy |
| `train_neg_ratio` | 3 | Negatives per positive |

## Testing

The project includes a comprehensive test suite with **409 unit tests** covering **81% of the codebase**.

```bash
# Run full test suite
uv run python -m pytest tests/ -q

# Run with coverage report
uv run python -m pytest tests/ --cov=src --cov-report=term-missing
```

Key module coverage:

| Module | Coverage |
|:---|---:|
| `models.py` | 96% |
| `eval_utils.py` | 94% |
| `losses.py` | 89% |
| `config.py` | 88% |
| `negative_sampling.py` | 88% |
| `training/optimiser.py` | 82% |

Tests cover model architectures, evaluation metrics, loss functions, configuration, negative sampling strategies, data loading, edge/node feature extraction, graph construction, MLflow tracking, LOO protocol correctness, and Long COVID prediction pipeline (gene parsing, category filtering, gene wiring).

## MLflow Experiment Tracking

```bash
uv run mlflow ui
# Navigate to http://localhost:5000
```

## License

See [License.md](License.md) for details.

## Acknowledgements

- [Open Targets Platform](https://www.opentargets.org/) — Biomedical knowledge graph data
- [PyTorch Geometric](https://pyg.org/) — GNN framework

## Contact

For questions or issues, please open a GitHub issue or contact [kim.lopez@spc.ox.ac.uk].
