"""
Unit tests for ID mapping and node indexing.
"""
import pytest
import pandas as pd
import pyarrow as pa
from src.data.mappers import IdMapper, NodeIndexMapper


class TestIdMapper:
    """Test IdMapper logic."""
    
    def test_resolves_redundant_ids(self):
        """Test that IdMapper correctly resolves redundant drug IDs."""
        # IdMapper now has predefined mappings in __init__
        mapper = IdMapper()
        
        # Test one of the predefined mappings (e.g., CHEMBL1200538 -> CHEMBL632)
        # We can check the dictionary directly or call resolve_mapping
        mappings = mapper.redundant_id_mapping['drug_mappings']
        
        resolved = mapper.resolve_mapping('CHEMBL1200538', mappings)
        assert resolved == 'CHEMBL632'
        
        # Test recursive resolution if any exists, or just direct
        # In the real code, most are direct mappings.
        
    def test_apply_drug_mappings(self):
        """Test applying drug mappings to a dataframe."""
        molecule_df = pd.DataFrame({
            'id': ['CHEMBL1200538', 'CHEMBL3184512', 'CHOOSE_ME'],
            'name': ['Drug 1', 'Drug 2', 'Drug 3']
        })
        
        mapper = IdMapper()
        result = mapper.apply_drug_mappings(molecule_df)
        
        assert 'CHEMBL632' in result['id'].values # Resolved from CHEMBL1200538
        assert 'CHEMBL1753' in result['id'].values # Resolved from CHEMBL3184512
        assert 'CHOOSE_ME' in result['id'].values # Unchanged


class TestNodeIndexMapper:
    """Test NodeIndexMapper logic."""
    
    def test_create_node_mappings(self):
        """Test that NodeIndexMapper creates correct index mappings."""
        molecule_df = pd.DataFrame({
            'id': ['DRUG1', 'DRUG2'],
            'drugType': ['Small molecule', 'Antibody']
        })
        disease_df = pd.DataFrame({
            'id': ['DIS1', 'DIS2'],
            'therapeuticAreas': [str(['AREA1']), str(['AREA2'])]
        })
        gene_table = pa.Table.from_pandas(pd.DataFrame({
            'id': ['GENE1', 'GENE2'],
            'pathways': [str(['PATH1']), str(['PATH2'])]
        }))
        
        mapper = NodeIndexMapper()
        mappings = mapper.create_node_mappings(molecule_df, disease_df, gene_table, 23.06)
        
        # Check that we have all expected keys in the result
        assert 'drug_key_mapping' in mappings
        assert 'gene_key_mapping' in mappings
        assert 'disease_key_mapping' in mappings
        
        # Check counts
        assert len(mappings['drug_key_mapping']) == 2
        assert len(mappings['gene_key_mapping']) == 2
        assert len(mappings['disease_key_mapping']) == 2
        
        # Check that indices are unique across all types
        all_indices = []
        for key in ['drug_key_mapping', 'drug_type_key_mapping', 'gene_key_mapping', 
                    'reactome_key_mapping', 'disease_key_mapping', 'therapeutic_area_key_mapping']:
            all_indices.extend(mappings[key].values())
        
        assert len(all_indices) == len(set(all_indices))
