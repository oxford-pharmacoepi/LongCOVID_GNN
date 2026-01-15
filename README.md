# Long COVID Drug Repurposing with Graph Neural Networks

Built on the project "KG-Bench: Benchmarking Graph Neural Network Algorithms for Drug Repurposing", this Graph Neural Network (GNN) framework is a direct application for drug repurposing with specialised focus on Long COVID treatment discovery. This system uses temporal knowledge graphs from Open Targets data to predict novel drug-disease associations through link prediction.

## Overview

Drug repurposing accelerates treatment discovery by identifying new therapeutic uses for existing drugs. This framework provides a systematic approach to evaluate GNN architectures on drug-disease association prediction tasks using biomedical knowledge graphs, with built-in temporal validation to ensure robust predictive performance.

**Key Capabilities:**
- Temporal validation using time-stamped Open Targets releases (2021, 2023, 2024)
- Multiple negative sampling strategies (random, hard, degree-matched, feature-similarity, mixed)
- Bayesian hyperparameter optimisation with Optuna
- Comprehensive experiment tracking with MLflow
- Specific Long COVID drug candidate identification
- Model explainability via GNNExplainer

## Project Structure

```
LongCOVID_GNN/
├── src/                              # Core framework modules
│   ├── models.py                     # GNN architectures (GCN, GraphSAGE, Transformer)
│   ├── utils.py                      # Evaluation metrics and utilities
│   ├── config.py                     # Centralized configuration
│   ├── data_processing.py            # Data loading and preprocessing
│   ├── negative_sampling.py          # Negative sampling strategies
│   ├── bayesian_optimiser.py         # Bayesian hyperparameter optimisation
│   └── mlflow_tracker.py             # Experiment tracking
│
├── scripts/                          # Pipeline components
│   ├── 1_create_graph.py             # Knowledge graph construction
│   ├── 2_train_models.py             # Model training with validation
│   ├── 3_test_evaluate.py            # Testing and evaluation
│   ├── 4_explain_predictions.py      # Prediction explanation
│   └── 5_optimise_hyperparameters.py # Standalone hyperparameter optimisation
│
# This is retrievable if the user runs the data extraction script
├── processed_data/                   # Pre-processed datasets (included)
│   ├── tables/                       # Entity tables (drugs, diseases, genes)
│   ├── mappings/                     # Node index mappings
│   └── edges/                        # Pre-computed graph edges
│
# This is obtainable if the user runs the whole pipeline
├── results/                          # Output directory
│   ├── models/                       # Trained model checkpoints
│   ├── evaluation/                   # Performance metrics
│   ├── figures/                      # Visualisations
│   ├── predictions/                  # Model predictions
│   └── experiments/                  # Experiment results
│   └── long_covid/                   # Long COVID drug repurposing results
│
├── long_covid_drug_repurposing.py    # Long COVID analysis pipeline
├── run_pipeline.py                   # Full pipeline orchestrator
└── gwas_genes_long_covid.txt         # Long COVID GWAS genes
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

### Prerequisites

- Python 3.12+
- uv package manager

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### Setup

```bash
# Clone repository
git clone <repository-url>
cd LongCOVID_GNN

# Install dependencies and project
uv sync
```

## Quick Start

### Using Pre-processed Data (Recommended)

The repository includes pre-processed Open Targets data (v21.06, v23.06, v24.06) for immediate use:

```bash
# Run complete pipeline
uv run python run_pipeline.py
```

This executes:
1. Graph construction from processed data
2. Model training (GCN, GraphSAGE, Transformer)
3. Temporal validation and testing
4. Evaluation and visualization

### Long COVID Drug Repurposing

```bash
# Predict drug candidates for Long COVID
uv run python long_covid_drug_repurposing.py --top-k 100 --lookup-names --visualise

# Options:
#   --top-k N           Return top N candidates (default: 50)
#   --all-drugs         Predict all drugs (~1,900 compounds)
#   --lookup-names      Query ChEMBL for drug names
#   --lookup-top-n N    Limit name lookup to top N drugs
#   --visualise         Generate distribution plots
```

**Example output:**
```
Top 10 Drug Candidates:
1. Dexamethasone         Probability: 0.8234 | Confidence: High
2. Tocilizumab          Probability: 0.7891 | Confidence: High
3. Baricitinib          Probability: 0.7654 | Confidence: High
...
```

## Pipeline Components

### 1. Graph Construction

```bash
# Create knowledge graph
uv run python scripts/1_create_graph.py

# With options
uv run python scripts/1_create_graph.py \
    --force-mode processed \
    --analyze \
    --experiment-name graph_creation_v1
```

**Graph structure:**
- Nodes: ~60,000 (drugs, genes, diseases, pathways, therapeutic areas)
- Edges: ~350,000 (6 edge types)
- Features: Drug properties, gene biotypes, disease classifications

### 2. Model Training

```bash
# Train all models
uv run python scripts/2_train_models.py

# Train specific model
uv run python scripts/2_train_models.py --model Transformer --epochs 200

# Train with automatic hyperparameter optimisation (recommended)
uv run python scripts/2_train_models.py --optimise-first --n-trials 100 --model Transformer
```

**Training features:**
- Early stopping on validation APR
- Negative sampling strategies
- Mixed precision training
- Automatic hyperparameter logging
- Optional Bayesian hyperparameter optimisation

**Hyperparameter Optimisation Workflows:**

**Option 1: Integrated optimisation (one command)**
```bash
# Optimise hyperparameters, then train immediately with best params
uv run python scripts/2_train_models.py --optimise-first --n-trials 50 --model Transformer
```

**Option 2: Sequential optimisation**
```bash
# Step 1: Run optimisation separately
uv run python scripts/5_optimise_hyperparameters.py --model Transformer --n-trials 100

# Step 2: Review results in results/bayesian_optimisation/
# - best_params_*.json contains optimal hyperparameters
# - optimisation_plots_*.png shows convergence and importance

# Step 3: Update src/config.py with best parameters (should be automatic, but check manually)

# Step 4: Train with optimised hyperparameters
uv run python scripts/2_train_models.py --model Transformer
```

### 3. Evaluation

```bash
# Evaluate on test set
uv run python scripts/3_test_evaluate.py

# With custom settings
uv run python scripts/3_test_evaluate.py \
    --graph results/graph_*.pt \
    --models results/models/ \
    --export-fp \
    --fp-threshold 0.7
```

**Metrics computed:**
- AUC-ROC (area under ROC curve)
- APR (average precision-recall)
- F1 score at optimal threshold
- Precision, Recall, Specificity
- Confusion matrices

### 4. Explanation

```bash
# Explain model predictions
uv run python scripts/4_explain_predictions.py \
    --graph results/graph_*.pt \
    --predictions results/predictions/TransformerModel_predictions.csv \
    --top-k 20
```

### 5. Hyperparameter Optimisation

```bash
# Optimise hyperparameters
uv run python scripts/5_optimise_hyperparameters.py \
    --model Transformer \
    --trials 50 \
    --timeout 3600
```

## Configuration

All settings are managed in `src/config.py`:

```python
class Config:
    # Data versions (temporal validation)
    training_version = "21.06"      # Training data
    validation_version = "23.06"    # Validation data  
    test_version = "24.06"          # Test data
    
    # Model architecture
    model_choice = 'Transformer'    # GCN | SAGE | Transformer
    hidden_channels = 128
    num_layers = 3
    dropout_rate = 0.2
    
    # Training parameters
    learning_rate = 0.001
    epochs = 100
    batch_size = 1024
    patience = 15
    primary_metric = 'apr'          # Metric for early stopping
    
    # Negative sampling
    negative_sampling_strategy = 'hard'  # random | hard | degree_matched
    train_neg_ratio = 10            # Training negatives per positive
    pos_neg_ratio = 10              # Val/test negatives per positive
    
    # Paths
    paths = {
        'processed': 'processed_data/',
        'results': 'results/',
        'raw': 'raw_data/'
    }
```

## Model Architectures

### GCN (Graph Convolutional Network)
- Spectral convolution using normalized adjacency matrix
- Fast inference, good baseline performance
- Best for: Homogeneous graphs with uniform node types

### GraphSAGE (Sample and Aggregate)
- Inductive learning via neighborhood sampling
- Handles large graphs efficiently
- Best for: Scalability and generalization

### Graph Transformer
- Multi-head attention over graph structure
- Captures long-range dependencies
- Best for: Complex relational patterns (recommended)

## Negative Sampling Strategies

### Random Sampling
- Uniform random selection from non-edges
- Fast, simple baseline
- May include trivial negatives

### Hard Negative Sampling
- Prioritizes high-similarity but disconnected pairs
- More challenging training examples
- Improves model discrimination

### Degree-Matched Sampling
- Matches degree distribution of positive edges
- Controls for structural biases
- Better calibrated predictions

## Temporal Validation

The framework uses three Open Targets versions for robust temporal validation:

1. **Training (21.06):** Historical drug-disease associations
2. **Validation (23.06):** Recent associations for hyperparameter tuning
3. **Test (24.06):** Latest associations for final evaluation

This ensures models generalize to future discoveries, not just interpolate known associations.

## MLflow Experiment Tracking

View all experiments:

```bash
uv run mlflow ui
```

Then navigate to `http://localhost:5000`

**Tracked information:**
- Hyperparameters (learning rate, architecture, sampling strategy)
- Metrics (AUC, APR, F1 per epoch)
- Artifacts (trained models, visualizations, predictions)
- System info (runtime, GPU usage, Python environment)

## Output Files

### Graph Files
`results/graph_{version}_{mode}_{timestamp}.pt`
- PyTorch Geometric Data object with node features and edge indices
- Includes train/val/test splits
- Metadata: node counts, edge types, configuration

### Model Checkpoints
`results/models/{ModelName}_best_model_{timestamp}.pt`
- State dict of trained model parameters
- Optimizer state for resuming training
- Training metrics history

### Evaluation Results
`results/evaluation/test_results_summary_{timestamp}.csv`
- Performance metrics for all models
- Statistical comparisons
- Best model selection criteria

### Visualisations
- `results/figures/test_roc_curves_{timestamp}.png` - ROC curves
- `results/figures/test_pr_curves_{timestamp}.png` - Precision-recall curves
- `results/figures/test_confusion_matrices_{timestamp}.png` - Confusion matrices
- `results/figures/model_comparison_{timestamp}.png` - Performance comparison

### Predictions
`results/predictions/{ModelName}_predictions_{timestamp}.csv`
```csv
drug_id,disease_id,probability,predicted_label,true_label
CHEMBL123,MONDO:0001234,0.8234,1,1
CHEMBL456,MONDO:0005678,0.7891,1,0
...
```

## Data Processing (Optional)

To use raw Open Targets data instead of pre-processed files:

### Download Raw Data

```bash
# Automated download
uv run python download_parquet_files.py

# Or manually from: https://platform.opentargets.org/downloads/
```

Required versions:
- 21.06: indication, molecule, diseases, targets, associationByOverallDirect
- 23.06: indication
- 24.06: indication

### Process Raw Data

```bash
# Force processing from raw files
uv run python scripts/1_create_graph.py --force-mode raw
```

This generates processed files in `processed_data/` for future use.

## Long COVID Analysis Details

The Long COVID pipeline (`long_covid_drug_repurposing.py`):

1. **GWAS Integration:** Uses Long COVID GWAS genes from `gwas_genes_long_covid.txt`
2. **Graph Augmentation:** Adds Long COVID node connected to GWAS genes
3. **Feature Initialisation:** Averages features from similar diseases (CFS, COPD, fibromyalgia)
4. **Drug Prediction:** Ranks all drugs by predicted association strength
5. **ChEMBL Lookup:** Enriches predictions with drug names and approval status
6. **Visualisation:** Generates distribution plots and confidence metrics


## License

See [License.md](License.md) for details.

## Acknowledgments

- [Open Targets Platform](https://www.opentargets.org/) - Biomedical knowledge graph data

## Contact

For questions or issues, please open a GitHub issue or contact [kim.lopez@spc.ox.ac.uk].
