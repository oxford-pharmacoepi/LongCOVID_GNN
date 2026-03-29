# LongCOVID_GNN Tests

## Overview

Test suite covering the graph creation pipeline, model architectures, training, evaluation, and prediction pipelines.

## Running Tests

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest tests/ -q

# Run with coverage report
uv run pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_models_seal.py -v
```

## Test Summary

| Category | File(s) | Tests | Description |
|----------|---------|------:|-------------|
| Config | `test_config.py` | 17 | Configuration paths, defaults, validation |
| Data Loading | `test_data_loaders.py`, `test_loaders.py` | 22 | Data loading from parquet/JSON files |
| Data Filtering | `test_data_filters.py` | 8 | Drug filtering, score thresholds |
| Data Mapping | `test_data_mappers.py`, `test_mappers.py` | 16 | ID resolution, node index mapping |
| Data Storage | `test_data_storage.py` | 6 | Serialisation and loading |
| Data Processing | `test_data_processing.py` | 8 | End-to-end processing pipeline |
| Graph Construction | `test_graph_builder.py`, `test_graph_construction.py` | 12 | Builder, edge extraction, splits |
| Graph Utilities | `test_graph_utils.py`, `test_graph_utils_extended.py` | 20 | Adjacency, degree, subgraph ops |
| Graph Splits | `test_graph_split.py` | 3 | Train/val/test splitting |
| Edge Extraction | `test_edge_extraction.py` | 5 | Raw data → graph edge validation |
| Edge Features | `test_edge_features.py` | 10 | MoA encoding, edge feature dims |
| Edge Utilities | `test_edge_utils.py` | 12 | Edge indexing, type detection |
| Node Features | `test_features.py`, `test_feature_utils.py` | 14 | Node feature encoding, normalisation |
| Models (Global GNN) | `test_models.py` | 24 | Forward pass, gradient flow, all architectures |
| Models (SEAL) | `test_models_seal.py` | 9 | DRNL labelling, subgraph extraction, SEALModel |
| SEAL Dataset | `test_seal_dataset.py` | 8 | Dataset creation, caching |
| Negative Sampling | `test_negative_sampling_unit.py` | 16 | Random, hard, mixed strategies |
| Loss Functions | `test_losses.py` | 18 | BCE, focal, PU, ranking losses |
| Training | `test_training.py` | 20 | Training loop, early stopping |
| Optimiser | `test_optimiser.py`, `test_optimiser_extended.py`, `test_optimiser_init.py` | 28 | Bayesian hyperparameter optimisation |
| Evaluation | `test_eval_utils.py` | 16 | Hits@K, MRR, NDCG, ranking metrics |
| Tracker | `test_tracker.py` | 14 | MLflow experiment tracking |
| Common Utilities | `test_common.py` | 3 | Reproducibility, seeding |
| Data Faithfulness | `test_data_faithfulness.py` | 8 | Graph vs raw data validation |
| Data Invariants | `test_data_invariants.py` | 5 | Structural integrity (no self-loops, symmetry) |
| Processed Graph | `test_processed_graph.py` | 5 | Serialised graph validation |
| **LOO Protocol** | `test_loo_protocol.py` | 5 | Edge holdout, negative isolation, scoring |
| **LC Prediction** | `test_predict_long_covid.py` | 11 | Gene parsing, category filtering, wiring |
| **Total** | **37 files** | **409** | **81% coverage** |

## Architecture

### Shared Graph Fixture

Integration tests share a single graph built once per session via `conftest.py`:

```python
@pytest.fixture(scope="session")
def shared_graph():
    """Build graph once per test session and reuse across all tests."""
    builder = GraphBuilder(config, force_mode='processed', tracker=None)
    builder.load_or_create_data()
    builder.create_node_features()
    builder.create_edges()
    builder.create_train_val_test_splits()
    graph = builder.build_graph()
    return {'graph': graph, 'builder': builder, 'config': config}
```

### Test Categories

- **Unit tests** (`test_models*.py`, `test_losses.py`, etc.): Fast, no data dependencies
- **Integration tests** (`test_data_faithfulness.py`, `test_edge_extraction.py`): Require processed data, marked with `@pytest.mark.integration`
- **Pipeline tests** (`test_loo_protocol.py`, `test_predict_long_covid.py`): Validate script logic using synthetic data
