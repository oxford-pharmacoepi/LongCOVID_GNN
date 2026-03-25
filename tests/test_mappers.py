"""
Tests for src/data/mappers.py — IdMapper and NodeIndexMapper.
"""

import pytest
import numpy as np
import pandas as pd
import pyarrow as pa

from src.data.mappers import IdMapper, NodeIndexMapper


class TestIdMapper:
    @pytest.fixture
    def mapper(self):
        return IdMapper()

    def test_init(self, mapper):
        assert 'drug_mappings' in mapper.redundant_id_mapping
        assert 'disease_mappings' in mapper.redundant_id_mapping

    def test_resolve_direct(self, mapper):
        assert mapper.resolve_mapping('X', {'X': 'Y'}) == 'Y'

    def test_resolve_chain(self, mapper):
        assert mapper.resolve_mapping('A', {'A': 'B', 'B': 'C'}) == 'C'

    def test_resolve_no_mapping(self, mapper):
        assert mapper.resolve_mapping('Z', {'A': 'B'}) == 'Z'

    def test_safe_list_none(self, mapper):
        assert mapper.safe_list_conversion(None) == []

    def test_safe_list_na(self, mapper):
        assert mapper.safe_list_conversion(float('nan')) == []

    def test_safe_list_string(self, mapper):
        result = mapper.safe_list_conversion("['a', 'b']")
        assert result == ['a', 'b']

    def test_safe_list_bad_string(self, mapper):
        assert mapper.safe_list_conversion("not a list") == []

    def test_safe_list_list(self, mapper):
        assert mapper.safe_list_conversion([1, 2]) == [1, 2]

    def test_safe_list_tuple(self, mapper):
        assert mapper.safe_list_conversion((1, 2)) == [1, 2]

    def test_safe_list_ndarray(self, mapper):
        result = mapper.safe_list_conversion(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_safe_list_other(self, mapper):
        assert mapper.safe_list_conversion(42) == []

    def test_update_approved_indications(self, mapper):
        result = mapper.update_approved_indications(['D1', 'D2'], {'D1': 'D1_new'})
        assert 'D1_new' in result or result == ['D1_new', 'D2']

    def test_apply_drug_mappings(self, mapper):
        df = pd.DataFrame({'id': ['CHEMBL1200538', 'CHEMBL999']})
        result = mapper.apply_drug_mappings(df)
        assert result.loc[0, 'id'] == 'CHEMBL632'
        assert result.loc[1, 'id'] == 'CHEMBL999'

    def test_apply_disease_mappings(self, mapper):
        df = pd.DataFrame({
            'id': ['X'],
            'approvedIndications': [['EFO_1000905', 'EFO_0000000']]
        })
        result = mapper.apply_disease_mappings(df)
        assert 'EFO_0004228' in result.loc[0, 'approvedIndications']

    def test_apply_disease_mappings_no_column(self, mapper):
        df = pd.DataFrame({'id': ['X']})
        result = mapper.apply_disease_mappings(df)
        assert len(result) == 1


class TestNodeIndexMapper:
    def test_create_node_mappings(self):
        mapper = NodeIndexMapper()
        molecule_df = pd.DataFrame({
            'id': ['M1', 'M2'],
            'drugType': ['SmallMolecule', 'Antibody'],
        })
        disease_df = pd.DataFrame({
            'id': ['D1', 'D2', 'D3'],
            'therapeuticAreas': [['TA1'], ['TA1', 'TA2'], ['TA2']],
        })
        gene_table = pa.table({
            'id': ['G1', 'G2'],
            'reactome': [['R1'], ['R2', 'R3']],
        })

        result = mapper.create_node_mappings(molecule_df, disease_df, gene_table, 21.06)
        
        assert 'drug_key_mapping' in result
        assert 'gene_key_mapping' in result
        assert 'disease_key_mapping' in result
        assert len(result['approved_drugs_list']) == 2
        assert len(result['gene_list']) == 2
        assert len(result['disease_list']) == 3
        assert len(result['reactome_list']) > 0

    def test_node_indices_dont_overlap(self):
        mapper = NodeIndexMapper()
        molecule_df = pd.DataFrame({
            'id': ['M1'],
            'drugType': ['SmallMolecule'],
        })
        disease_df = pd.DataFrame({
            'id': ['D1', 'D2'],
            'therapeuticAreas': [['TA1'], ['TA1']],
        })
        gene_table = pa.table({
            'id': ['G1'],
            'reactome': [['R1']],
        })
        result = mapper.create_node_mappings(molecule_df, disease_df, gene_table, 21.06)
        
        all_indices = set()
        for mapping_name in ['drug_key_mapping', 'drug_type_key_mapping', 'gene_key_mapping',
                             'reactome_key_mapping', 'disease_key_mapping', 'therapeutic_area_key_mapping']:
            for idx in result[mapping_name].values():
                assert idx not in all_indices, f"Duplicate index {idx} in {mapping_name}"
                all_indices.add(idx)
