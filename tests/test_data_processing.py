"""
Unit tests for src/data_processing.py

Tests the DataProcessor compatibility wrapper and utility functions.
"""

import pytest
import os
import tempfile
import pyarrow as pa
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.data_processing import DataProcessor, detect_data_mode
from src.config import Config


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def processor(config):
    return DataProcessor(config)


class TestDataProcessorInit:
    def test_has_loader(self, processor):
        assert processor.loader is not None

    def test_has_id_mapper(self, processor):
        assert processor.id_mapper is not None

    def test_has_node_mapper(self, processor):
        assert processor.node_mapper is not None

    def test_has_molecule_filter(self, processor):
        assert processor.molecule_filter is not None

    def test_has_storage(self, processor):
        assert processor.storage is not None

    def test_has_redundant_id_mapping(self, processor):
        assert processor.redundant_id_mapping is not None


class TestDelegatedMethods:
    """Ensure wrapper methods delegate correctly."""

    def test_resolve_mapping(self, processor):
        result = processor.resolve_mapping('A', {'A': 'B', 'B': 'C'})
        assert result == 'C'

    def test_safe_list_conversion_list(self, processor):
        result = processor.safe_list_conversion([1, 2, 3])
        assert isinstance(result, list)

    def test_safe_list_conversion_none(self, processor):
        result = processor.safe_list_conversion(None)
        assert isinstance(result, list)

    def test_update_approved_indications(self, processor):
        result = processor.update_approved_indications(
            ['D1', 'D2'], {'D1': 'D1_new'}
        )
        assert 'D1_new' in result or 'D1' in result


class TestCreateGeneReactomeMapping:
    def test_old_version_with_reactome(self, processor):
        gene_table = pa.table({
            'id': ['G1', 'G2'],
            'reactome': [['R1', 'R2'], ['R3']],
        })
        result = processor.create_gene_reactome_mapping(gene_table, 21.06)
        assert isinstance(result, pa.Table)

    def test_old_version_no_reactome(self, processor):
        gene_table = pa.table({
            'id': ['G1', 'G2'],
            'biotype': ['protein_coding', 'lncRNA'],
        })
        result = processor.create_gene_reactome_mapping(gene_table, 21.06)
        assert isinstance(result, pa.Table)

    def test_new_version_with_pathways(self, processor):
        gene_table = pa.table({
            'id': ['G1'],
            'pathways': [[{'pathwayId': 'R-HSA-123'}]],
        })
        result = processor.create_gene_reactome_mapping(gene_table, 24.06)
        assert isinstance(result, pa.Table)
        assert len(result) >= 1

    def test_new_version_no_pathways(self, processor):
        gene_table = pa.table({
            'id': ['G1'],
            'biotype': ['protein_coding'],
        })
        result = processor.create_gene_reactome_mapping(gene_table, 24.06)
        assert isinstance(result, pa.Table)


class TestDetectDataMode:
    def test_force_mode(self):
        config = Config()
        assert detect_data_mode(config, force_mode='raw') == 'raw'
        assert detect_data_mode(config, force_mode='processed') == 'processed'

    def test_auto_detect_raw(self):
        config = Config()
        config.paths['processed_path'] = '/nonexistent/path'
        result = detect_data_mode(config)
        assert result == 'raw'

    def test_auto_detect_processed(self):
        config = Config()
        with tempfile.TemporaryDirectory() as tmpdir:
            tables_dir = Path(tmpdir) / 'tables'
            mappings_dir = Path(tmpdir) / 'mappings'
            tables_dir.mkdir()
            mappings_dir.mkdir()
            config.paths['processed_path'] = tmpdir
            result = detect_data_mode(config)
            assert result == 'processed'
