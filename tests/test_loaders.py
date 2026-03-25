"""
Tests for src/data/loaders.py — OpenTargetsLoader.
Uses temporary parquet files to test loading and preprocessing.
"""

import pytest
import tempfile
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.loaders import OpenTargetsLoader


@pytest.fixture
def loader():
    return OpenTargetsLoader()


@pytest.fixture
def indication_dir(tmp_path):
    """Create a temp directory with a minimal indication parquet file."""
    df = pd.DataFrame({
        'id': ['M1', 'M2', 'M3'],
        'approvedIndications': [['D1', 'D2'], ['D3'], []],
        'indication': ['ind1', 'ind2', 'ind3'],
    })
    table = pa.Table.from_pandas(df)
    pq.write_table(table, tmp_path / 'part-0.parquet')
    return str(tmp_path)


@pytest.fixture
def molecule_dir(tmp_path):
    """Create a temp directory with a minimal molecule parquet file."""
    df = pd.DataFrame({
        'id': ['M1', 'M2'],
        'name': ['Drug1', 'Drug2'],
        'drugType': ['SmallMolecule', None],
        'blackBoxWarning': [True, False],
        'yearOfFirstApproval': [2010, 2015],
        'parentId': [None, 'M1'],
        'childChemblIds': [['M2'], None],
        'linkedDiseases': [{'count': 2, 'rows': ['D1', 'D2']}, {'count': 1, 'rows': ['D1']}],
        'hasBeenWithdrawn': [False, False],
        'linkedTargets': [{'count': 1, 'rows': ['G1']}, {'count': 1, 'rows': ['G2']}],
    })
    table = pa.Table.from_pandas(df)
    pq.write_table(table, tmp_path / 'part-0.parquet')
    return str(tmp_path)


@pytest.fixture 
def gene_dir(tmp_path):
    """Create gene parquet files for old format."""
    df = pd.DataFrame({
        'id': ['G1', 'G2'],
        'approvedName': ['Gene1', 'Gene2'],
        'bioType': ['protein_coding', 'lncRNA'],
        'proteinAnnotations': [{'functions': ['func1']}, {'functions': []}],
        'reactome': [['R1'], ['R2']],
    })
    table = pa.Table.from_pandas(df)
    pq.write_table(table, tmp_path / 'part-0.parquet')
    return str(tmp_path)


class TestLoadIndicationData:
    def test_loads_and_filters(self, loader, indication_dir):
        result = loader.load_indication_data(indication_dir)
        # Only rows with non-empty approvedIndications (M1 and M2)
        assert len(result) == 2

    def test_missing_dir_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_indication_data('/nonexistent/path')


class TestLoadMoleculeData:
    def test_loads(self, loader, molecule_dir):
        result = loader.load_molecule_data(molecule_dir)
        assert len(result) == 2
        # drugType 'unknown' should be replaced with 'Unknown'
        df = result.to_pandas()
        assert 'drugType' in df.columns

    def test_missing_dir_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_molecule_data('/nonexistent/path')


class TestLoadGeneData:
    def test_loads_old_version(self, loader, gene_dir):
        result = loader.load_gene_data(gene_dir, 21.06)
        assert len(result) == 2
        assert 'id' in result.column_names

    def test_missing_dir_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_gene_data('/nonexistent/path', 21.06)


class TestLoadDiseaseData:
    def test_loads(self, loader, tmp_path):
        df = pd.DataFrame({
            'id': ['EFO_0001', 'UBERON_001', 'EFO_0002', 'EFO_0000544'],
            'name': ['Disease1', 'NotADisease', 'Disease2', 'SkippedDisease'],
            'description': ['d1', 'd2', 'd3', 'd4'],
            'ancestors': [[], [], [], []],
            'descendants': [[], [], [], []],
            'children': [[], [], [], []],
            'therapeuticAreas': [['TA1'], ['TA2'], ['TA1'], ['TA1']],
        })
        table = pa.Table.from_pandas(df)
        pq.write_table(table, tmp_path / 'part-0.parquet')
        
        result = loader.load_disease_data(str(tmp_path))
        df_result = result.to_pandas()
        # UBERON should be filtered out, EFO_0000544 should be filtered
        assert 'UBERON_001' not in df_result['id'].values
        assert 'EFO_0000544' not in df_result['id'].values

    def test_missing_dir_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_disease_data('/nonexistent/path')


class TestLoadAssociationsData:
    def test_loads(self, loader, tmp_path):
        df = pd.DataFrame({
            'targetId': ['G1', 'G2'],
            'diseaseId': ['D1', 'D2'],
            'score': [0.8, 0.3],
        })
        table = pa.Table.from_pandas(df)
        pq.write_table(table, tmp_path / 'part-0.parquet')
        
        result, score_column = loader.load_associations_data(str(tmp_path), 21.06)
        assert score_column == 'score'
        assert len(result) == 2

    def test_newer_version(self, loader, tmp_path):
        df = pd.DataFrame({
            'targetId': ['G1'],
            'diseaseId': ['D1'],
            'datasourceScores.overall': [0.5],
        })
        table = pa.Table.from_pandas(df)
        pq.write_table(table, tmp_path / 'part-0.parquet')
        
        result, score_column = loader.load_associations_data(str(tmp_path), 24.06)
        assert score_column == 'datasourceScores.overall'

    def test_missing_dir_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_associations_data('/nonexistent/path', 21.06)


class TestLoadKnownDrugs:
    def test_loads(self, loader, tmp_path):
        df = pd.DataFrame({'drugId': ['M1'], 'phase': [3]})
        table = pa.Table.from_pandas(df)
        pq.write_table(table, tmp_path / 'part-0.parquet')
        
        result = loader.load_known_drugs_aggregated(str(tmp_path))
        assert len(result) == 1

    def test_missing_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_known_drugs_aggregated('/nonexistent/path')


class TestLoadDrugWarnings:
    def test_loads(self, loader, tmp_path):
        df = pd.DataFrame({'chemblIds': [['M1']], 'warningType': ['Withdrawn'], 'toxicityClass': ['Hepatotoxicity']})
        table = pa.Table.from_pandas(df)
        pq.write_table(table, tmp_path / 'part-0.parquet')
        
        result = loader.load_drug_warnings(str(tmp_path))
        assert len(result) == 1

    def test_missing_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_drug_warnings('/nonexistent/path')


class TestLoadMoA:
    def test_loads(self, loader, tmp_path):
        df = pd.DataFrame({'chemblIds': [['M1']], 'targets': [['G1']], 'actionType': ['inhibitor']})
        table = pa.Table.from_pandas(df)
        pq.write_table(table, tmp_path / 'part-0.parquet')
        
        result = loader.load_mechanism_of_action(str(tmp_path))
        assert len(result) == 1


class TestLoadInteraction:
    def test_loads(self, loader, tmp_path):
        df = pd.DataFrame({'targetA': ['G1'], 'targetB': ['G2'], 'scoring': [0.9]})
        table = pa.Table.from_pandas(df)
        pq.write_table(table, tmp_path / 'part-0.parquet')
        
        result = loader.load_interaction(str(tmp_path))
        assert len(result) == 1

    def test_missing_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_interaction('/nonexistent/path')
