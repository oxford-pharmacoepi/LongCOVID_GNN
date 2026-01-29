"""
Unit tests for data filtering classes.
"""
import pytest
import pandas as pd
import pyarrow as pa
from src.data.filters import MoleculeFilter, AssociationFilter


class TestMoleculeFilter:
    """Test MoleculeFilter logic."""
    
    def test_filter_removes_children(self):
        """Test that molecules with parentId are removed."""
        molecule_data = pd.DataFrame({
            'id': ['CHEMBL1', 'CHEMBL2', 'CHEMBL3'],
            'parentId': [None, 'CHEMBL1', None],
            'linkedDiseases': [
                {'count': 1}, 
                {'count': 0}, 
                {'count': 1}
            ]
        })
        # Mock indication_df and known_drugs_df
        indication_df = pd.DataFrame({'id': ['CHEMBL1', 'CHEMBL3']})
        
        filter_obj = MoleculeFilter()
        result = filter_obj.filter_linked_molecules(molecule_data, indication_df)
        
        assert 'CHEMBL2' not in result['id'].values
        assert 'CHEMBL1' in result['id'].values
        assert 'CHEMBL3' in result['id'].values

    def test_filter_keeps_valid_drugs(self):
        """Test that drugs with Phase 3+ are retained from known_drugs_df."""
        molecule_data = pd.DataFrame({
            'id': ['CHEMBL1', 'CHEMBL2', 'CHEMBL3'],
            'parentId': [None, None, None]
        })
        # Molecule linked only via known drugs phase 3
        known_drugs_df = pd.DataFrame({
            'drugId': ['CHEMBL1', 'CHEMBL2'],
            'phase': [4, 2] # Phase 4 (keep), Phase 2 (filtered unless linked elsewhere)
        })
        # Empty indication and metadata links
        indication_df = pd.DataFrame({'id': []})
        
        # Add metadata links to mock rows
        def mock_has_linked_diseases(row):
            return False
        
        filter_obj = MoleculeFilter()
        # We need to mock the has_linked_diseases in the test or provide data that matches the expected structure
        molecule_data['linkedDiseases'] = [{'count': 0}, {'count': 0}, {'count': 0}]
        
        result = filter_obj.filter_linked_molecules(molecule_data, indication_df, known_drugs_df)
        
        assert 'CHEMBL1' in result['id'].values
        assert 'CHEMBL2' not in result['id'].values # Phase 2 and no other links


class TestAssociationFilter:
    """Test AssociationFilter logic."""
    
    def test_filter_by_score(self):
        """Test that associations are filtered by score threshold."""
        association_data = pa.Table.from_pandas(pd.DataFrame({
            'targetId': ['GENE1', 'GENE2', 'GENE3', 'GENE4'],
            'diseaseId': ['DIS1', 'DIS2', 'DIS3', 'DIS4'],
            'score': [0.9, 0.7, 0.5, 0.3]
        }))
        
        filter_obj = AssociationFilter()
        result = filter_obj.filter_by_score(association_data, 'score', threshold=0.6)
        
        assert result.num_rows == 2
        assert 'GENE1' in result.column('targetId').to_pylist()
        assert 'GENE2' in result.column('targetId').to_pylist()

    def test_filter_by_genes_and_diseases(self):
        """Test that associations are filtered by gene and disease membership."""
        association_data = pa.Table.from_pandas(pd.DataFrame({
            'targetId': ['GENE1', 'GENE2', 'GENE3', 'GENE4'],
            'diseaseId': ['DIS1', 'DIS2', 'DIS3', 'DIS4'],
            'score': [0.9, 0.8, 0.7, 0.6]
        }))
        
        valid_genes = ['GENE1', 'GENE2', 'GENE3']
        valid_diseases = ['DIS1', 'DIS2']
        
        filter_obj = AssociationFilter()
        result = filter_obj.filter_by_genes_and_diseases(
            association_data, valid_genes, valid_diseases, 'score', threshold=0.0
        )
        
        assert result.num_rows == 2
        assert 'GENE1' in result.column('targetId').to_pylist()
        assert 'GENE2' in result.column('targetId').to_pylist()
        assert 'GENE3' not in result.column('targetId').to_pylist() # Disease DIS3 is missing
