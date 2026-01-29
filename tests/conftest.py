"""
Pytest configuration and shared fixtures for LongCOVID_GNN tests.
"""
import pytest
import pandas as pd
import pyarrow as pa
import torch
import tempfile
import shutil
from pathlib import Path
from src.config import Config, get_config
from src.graph.builder import GraphBuilder


# ============================================================================
# SHARED GRAPH FIXTURE (create once and reuse for all tests)
# ============================================================================

@pytest.fixture(scope="session")
def shared_graph():
    """
    Build graph once per test session and reuse across all tests.
    """
    print("\n" + "="*80)
    print("BUILDING SHARED GRAPH FOR ALL TESTS (this happens once)")
    print("="*80)
    
    config = get_config()
    
    # Build graph using processed data if available, otherwise raw
    builder = GraphBuilder(config, force_mode='processed', tracker=None)
    builder.load_or_create_data()
    builder.create_node_features()
    builder.create_edges()
    builder.create_train_val_test_splits()
    graph = builder.build_graph()
    
    print(f"✓ Shared graph built: {graph.x.size(0):,} nodes, {graph.edge_index.size(1):,} edges")
    print("="*80 + "\n")
    
    # Return both graph and builder (for mappings, etc.)
    return {
        'graph': graph,
        'builder': builder,
        'config': config
    }


# ============================================================================
# BASIC FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration with minimal settings."""
    config = Config()
    config.seed = 42
    config.data_version = 21.06
    config.network_config = {
        'use_ppi_network': True,
        'ppi_score_threshold': 0.7,
        'use_disease_similarity': True,
        'disease_similarity_max_children': 10,
        'disease_similarity_min_shared': 1,
        'trial_edges': True
    }
    return config


# ============================================================================
# SAMPLE DATA FIXTURES (for unit tests)
# ============================================================================

@pytest.fixture
def sample_molecule_data():
    """Create sample molecule data for testing."""
    data = {
        'id': ['CHEMBL1', 'CHEMBL2', 'CHEMBL3', 'CHEMBL4', 'CHEMBL5'],
        'name': ['Drug A', 'Drug B', 'Drug C', 'Drug D', 'Drug E'],
        'drugType': ['Small molecule', 'Antibody', 'Small molecule', 'Protein', 'Small molecule'],
        'maximumClinicalTrialPhase': [4.0, 3.0, 4.0, 2.0, 4.0],
        'parentId': [None, None, None, None, None],
        'childChemblIds': [[], [], [], [], []],
        'linkedDiseases': [
            [{'id': 'EFO_0000001'}],
            [],
            [{'id': 'EFO_0000002'}, {'id': 'EFO_0000003'}],
            [],
            [{'id': 'EFO_0000001'}]
        ]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_disease_data():
    """Create sample disease data for testing."""
    data = {
        'id': ['EFO_0000001', 'EFO_0000002', 'EFO_0000003', 'EFO_0000004', 'EFO_0000005'],
        'name': ['Disease A', 'Disease B', 'Disease C', 'Disease D', 'Disease E'],
        'therapeuticAreas': [
            [{'id': 'EFO_0000001'}],
            [{'id': 'EFO_0000001'}],
            [{'id': 'EFO_0000002'}],
            [{'id': 'EFO_0000002'}],
            [{'id': 'EFO_0000003'}]
        ],
        'ancestors': [
            ['PARENT_1'],
            ['PARENT_1'],
            ['PARENT_2'],
            ['PARENT_2', 'PARENT_3'],
            ['PARENT_3']
        ]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_gene_data():
    """Create sample gene data for testing."""
    data = {
        'id': ['ENSG00000001', 'ENSG00000002', 'ENSG00000003', 'ENSG00000004', 'ENSG00000005'],
        'approvedSymbol': ['GENE1', 'GENE2', 'GENE3', 'GENE4', 'GENE5'],
        'biotype': ['protein_coding', 'protein_coding', 'lncRNA', 'protein_coding', 'protein_coding']
    }
    return pd.DataFrame(data)


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
