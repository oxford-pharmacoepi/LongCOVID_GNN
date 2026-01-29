"""
Unit tests for OpenTargets data loading functions.
"""
import pytest
import pandas as pd
import pyarrow as pa
from pathlib import Path
from src.data.loaders import OpenTargetsLoader


class TestOpenTargetsLoader:
    """Test OpenTargetsLoader data loading and initial filtering."""
    
    def test_load_molecule_data_returns_table(self, test_config, temp_dir):
        """Test that load_molecule_data returns a PyArrow table with correct structure."""
        # Create mock data that matches OpenTargets structure for struct fields
        molecule_data = pd.DataFrame({
            'id': ['CHEMBL1', 'CHEMBL2'],
            'name': ['Drug A', 'Drug B'],
            'drugType': ['Small molecule', 'Antibody'],
            'blackBoxWarning': [False, True],
            'yearOfFirstApproval': [2000, 2010],
            'parentId': [None, None],
            'childChemblIds': [[], []],
            'linkedDiseases': [{'count': 0, 'rows': []}, {'count': 0, 'rows': []}],
            'hasBeenWithdrawn': [False, False],
            'linkedTargets': [{'count': 0, 'rows': []}, {'count': 0, 'rows': []}]
        })
        
        molecule_path = temp_dir / "molecule"
        molecule_path.mkdir()
        molecule_data.to_parquet(molecule_path / "part-0.parquet")
    
        loader = OpenTargetsLoader(test_config)
        result = loader.load_molecule_data(str(molecule_path))
        
        assert isinstance(result, pa.Table)
        assert 'id' in result.column_names
        # 'linkedTargets.count' should have been dropped by the loader
        assert 'linkedTargets.count' not in result.column_names
        assert result.num_rows == 2

    def test_load_disease_data_filters_therapeutic_areas(self, test_config, temp_dir):
        """Test that load_disease_data excludes therapeutic area roots and specific IDs."""
        disease_data = pd.DataFrame({
            'id': ['EFO_0000001', 'EFO_0001444', 'EFO_0000544'],
            'name': ['Disease A', 'Root Area', 'Excluded ID'],
            'description': ['Desc A', 'Desc Root', 'Desc Excl'],
            'therapeuticAreas': [['TA1'], ['EFO_0001444'], ['TA3']],
            'ancestors': [[], [], []],
            'descendants': [[], [], []],
            'children': [[], [], []]
        })
    
        disease_path = temp_dir / "disease"
        disease_path.mkdir()
        disease_data.to_parquet(disease_path / "part-0.parquet")
    
        loader = OpenTargetsLoader(test_config)
        result = loader.load_disease_data(str(disease_path))
        
        # Should only keep EFO_0000001
        assert result.num_rows == 1
        assert result.column('id').to_pylist() == ['EFO_0000001']

    def test_load_gene_data_version_21_06(self, test_config, temp_dir):
        """Test that load_gene_data returns correct columns for version 21.06."""
        # version 21.06 uses: ['id', 'approvedName','bioType', 'proteinAnnotations.functions', 'reactome']
        gene_data = pd.DataFrame({
            'id': ['ENSG1', 'ENSG2'],
            'approvedName': ['Gene 1', 'Gene 2'],
            'bioType': ['protein_coding', 'protein_coding'],
            'proteinAnnotations': [{'functions': []}, {'functions': []}],
            'reactome': [[], []]
        })
    
        gene_path = temp_dir / "targets"
        gene_path.mkdir()
        gene_data.to_parquet(gene_path / "part-0.parquet")
    
        loader = OpenTargetsLoader(test_config)
        result = loader.load_gene_data(str(gene_path), 21.06)
        
        assert 'bioType' in result.column_names
        assert result.num_rows == 2

    def test_load_indication_data_structure(self, test_config, temp_dir):
        """Test that load_indication_data returns table with required columns."""
        indication_data = pd.DataFrame({
            'id': ['CHEMBL1', 'CHEMBL2'],
            'approvedIndications': [['DIS1'], []] 
        })
        
        indication_path = temp_dir / "indication"
        indication_path.mkdir()
        indication_data.to_parquet(indication_path / "part-0.parquet")
    
        loader = OpenTargetsLoader(test_config)
        result = loader.load_indication_data(str(indication_path))
        
        assert result.num_rows == 1
        assert result.column('id').to_pylist() == ['CHEMBL1']
