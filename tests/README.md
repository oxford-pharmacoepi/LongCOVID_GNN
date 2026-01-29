# LongCOVID_GNN Tests

## Overview

Test suite for the graph creation pipeline ensuring data processing correctness and graph integrity.

## Running Tests

### Install Dependencies
```bash
uv sync
```

### Run All Tests
```bash
uv run pytest tests/ -v
```

This will:
1. Build the graph once at the start
2. Run all 34 tests using the shared graph

### Run Specific Test Files
```bash
uv run pytest tests/test_data_faithfulness.py -v
uv run pytest tests/test_edge_extraction.py -v
uv run pytest tests/test_graph_construction.py -v
```

### Coverage Report
```bash
uv run pytest tests/ --cov=src --cov-report=term-missing
```

## Test Files

### Data Faithfulness Tests (`test_data_faithfulness.py`)
Validates that the graph accurately represents OpenTargets data.

| Test | Description |
|------|-------------|
| `test_drug_disease_edge_count_is_reasonable` | Edge counts match at least 5% of raw indication data |
| `test_drug_gene_edges_from_mechanism_of_action` | Drug-gene edges match mechanismOfAction parquet files |
| `test_val_test_edges_are_from_future_versions` | Val/test positives come from future OT versions (temporal separation) |
| `test_negative_samples_are_true_negatives` | Negative samples have less than 1% false negative rate |
| `test_mapping_indices_match_graph` | All mapping indices are valid node indices |
| `test_node_type_ranges_are_disjoint` | No index overlap between node types |
| `test_drug_count_matches_features` | Mapping counts match metadata counts |
| `test_feature_dimensions_sensible` | Features are non-zero and reasonably sized |

### Edge Extraction Tests (`test_edge_extraction.py`)
Validates edges against raw parquet files.

| Test | Description |
|------|-------------|
| `test_known_edges_from_raw_data` | Samples random drug-disease pairs from raw data and verifies in graph |
| `test_self_loop_removal` | Confirms no self-loops exist in the graph |

### Graph Construction Tests (`test_graph_construction.py`)
End-to-end graph building validation.

| Test | Description |
|------|-------------|
| `test_graph_builder_creates_valid_graph` | Valid graph structure with nodes and edges |
| `test_train_val_test_splits_exist` | Splits exist with no overlap |
| `test_graph_metadata_present` | Metadata includes node and edge info |

### Data Invariants Tests (`test_data_invariants.py`)
Graph structural invariants.

| Test | Description |
|------|-------------|
| `test_all_edge_indices_within_node_range` | Edge indices less than node count |
| `test_no_duplicate_edges` | No duplicate edges in the graph |
| `test_bidirectional_edges_symmetric` | Undirected graph symmetry |
| `test_feature_dimensions_match` | Feature matrix matches node count |
| `test_edge_features_present` | Edge features match edge count |

### Processed Graph Tests (`test_processed_graph.py`)
Validation of serialised graph data.

| Test | Description |
|------|-------------|
| `test_processed_graph_loads` | Graph loads from processed_data/graph.pt |
| `test_no_self_loops` | No self-loops in saved graph |
| `test_edge_indices_valid` | All edge indices within valid range |
| `test_graph_is_undirected` | Symmetric edges |
| `test_splits_exist` | Train/val/test splits present |

### Data Loaders Tests (`test_data_loaders.py`)
Unit tests for data loading functions.

| Test | Description |
|------|-------------|
| `test_load_molecule_data_returns_table` | Molecule data loading |
| `test_load_disease_data_filters_therapeutic_areas` | Disease filtering |
| `test_load_gene_data_version_21_06` | Gene data with version handling |
| `test_load_indication_data_structure` | Indication data structure |

### Data Filters Tests (`test_data_filters.py`)
Unit tests for filtering logic.

| Test | Description |
|------|-------------|
| `test_filter_removes_children` | Child molecules filtered out |
| `test_filter_keeps_valid_drugs` | Valid approved drugs kept |
| `test_filter_by_score` | Score-based filtering |
| `test_filter_by_genes_and_diseases` | Gene/disease membership filtering |

### Data Mappers Tests (`test_data_mappers.py`)
Unit tests for ID mapping and node indexing.

| Test | Description |
|------|-------------|
| `test_resolves_redundant_ids` | Redundant drug ID resolution |
| `test_apply_drug_mappings` | Drug mapping application |
| `test_create_node_mappings` | Node index mapping creation |

## How It Works

### Shared Graph Fixture
The `conftest.py` file defines a `shared_graph` fixture with `scope="session"`:

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

All integration tests receive this fixture as a parameter and use the pre-built graph.

## Test Summary

| Category | Tests | Description |
|----------|-------|-------------|
| Data Faithfulness | 8 | Validates graph represents raw data correctly |
| Edge Extraction | 2 | Raw data to graph edge validation |
| Graph Construction | 3 | End-to-end build validation |
| Data Invariants | 5 | Structural integrity checks |
| Processed Graph | 5 | Serialised graph validation |
| Data Loaders | 4 | Unit tests for loading |
| Data Filters | 4 | Unit tests for filtering |
| Data Mappers | 3 | Unit tests for mapping |
| **Total** | **34** | |
