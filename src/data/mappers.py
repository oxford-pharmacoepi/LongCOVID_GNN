"""
ID mapping and node indexing module.
Handles redundant ID resolution and node-to-index mappings for graph construction.
"""

import pandas as pd
import pyarrow as pa
import ast
import numpy as np


class IdMapper:
    """Handles redundant ID mappings and resolution.
    
    Single Responsibility: Resolve and apply ID mappings for data consistency.
    """
    
    def __init__(self):
        """Initialize with predefined redundant ID mappings."""
        self.redundant_id_mapping = self._get_redundant_id_mappings()
    
    def _get_redundant_id_mappings(self):
        """Get predefined redundant ID mappings for data cleaning."""
        drug_mappings = {
            'CHEMBL1200538': 'CHEMBL632',
            'CHEMBL1200376': 'CHEMBL632',
            'CHEMBL1200384': 'CHEMBL632',
            'CHEMBL1201207': 'CHEMBL632',
            'CHEMBL1497': 'CHEMBL632',
            'CHEMBL1201661': 'CHEMBL3989767',
            'CHEMBL1506': 'CHEMBL130',
            'CHEMBL1201281': 'CHEMBL130',
            'CHEMBL1201289': 'CHEMBL1753',
            'CHEMBL3184512': 'CHEMBL1753',
            'CHEMBL1530428': 'CHEMBL384467',
            'CHEMBL1201302': 'CHEMBL384467',
            'CHEMBL1511': 'CHEMBL135',
            'CHEMBL4298187': 'CHEMBL2108597',
            'CHEMBL4298110': 'CHEMBL2108597',
            'CHEMBL1200640': 'CHEMBL2108597',
            'CHEMBL989': 'CHEMBL1501',
            'CHEMBL1201064': 'CHEMBL1200600',
            'CHEMBL1473': 'CHEMBL1676',
            'CHEMBL1201512': 'CHEMBL1201688',
            'CHEMBL1201657': 'CHEMBL1201513',
            'CHEMBL1091': 'CHEMBL389621',
            'CHEMBL1549': 'CHEMBL389621',
            'CHEMBL3989663': 'CHEMBL389621',
            'CHEMBL1641': 'CHEMBL389621',
            'CHEMBL1200562': 'CHEMBL389621',
            'CHEMBL1201544': 'CHEMBL2108597',
            'CHEMBL1200823': 'CHEMBL2108597',
            'CHEMBL2021423': 'CHEMBL1200572',
            'CHEMBL1364144':'CHEMBL650',
            'CHEMBL1200844': 'CHEMBL650',
            'CHEMBL1201265': 'CHEMBL650',
            'CHEMBL1140': 'CHEMBL573',
            'CHEMBL1152': 'CHEMBL131',
            'CHEMBL1201231': 'CHEMBL131',
            'CHEMBL1200909': 'CHEMBL131',
            'CHEMBL635': 'CHEMBL131',
            'CHEMBL1200335': 'CHEMBL386630',
            'CHEMBL1504': 'CHEMBL1451',
            'CHEMBL1200449': 'CHEMBL1451',
            'CHEMBL1200878': 'CHEMBL1451',
            'CHEMBL1200929': 'CHEMBL3988900'
        }
        
        disease_mappings = {
            'EFO_1000905': 'EFO_0004228',
            'EFO_0005752': 'EFO_1001888',
            'EFO_0007512': 'EFO_0007510'
        }
        
        return {
            'drug_mappings': drug_mappings,
            'disease_mappings': disease_mappings
        }
    
    def resolve_mapping(self, entity_id, mapping_dict):
        """Recursively resolve ID mappings to the final target."""
        visited = set()
        while entity_id in mapping_dict and entity_id not in visited:
            visited.add(entity_id)
            entity_id = mapping_dict[entity_id]
        return entity_id
    
    def safe_list_conversion(self, value):
        """Safely convert various formats to lists."""
        # Handle None first
        if value is None:
            return []
        
        # Handle pandas NA/None - wrap in try/except for arrays
        try:
            if pd.isna(value):
                return []
        except (ValueError, TypeError):
            # value might be an array, continue
            pass
        
        # Handle strings
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except:
                return []
        
        # Handle lists and tuples
        elif isinstance(value, (list, tuple)):
            return list(value)
            
        # Handle numpy arrays
        elif isinstance(value, np.ndarray):
            return value.tolist()
        
        # Default
        else:
            return []
    
    def update_approved_indications(self, disease_list, mapping_dict):
        """Update disease IDs inside lists using mapping dictionary."""
        disease_list = self.safe_list_conversion(disease_list)
        return [self.resolve_mapping(disease_id, mapping_dict) for disease_id in disease_list]
    
    def apply_drug_mappings(self, molecule_df):
        """Apply drug ID mappings to molecule dataframe."""
        drug_mappings = self.redundant_id_mapping['drug_mappings']
        
        # Apply to main ID column
        molecule_df['id'] = molecule_df['id'].apply(
            lambda x: self.resolve_mapping(x, drug_mappings)
        )
        
        return molecule_df
    
    def apply_disease_mappings(self, indication_df):
        """Apply disease ID mappings to indication dataframe."""
        disease_mappings = self.redundant_id_mapping['disease_mappings']
        
        # Apply to approvedIndications lists
        if 'approvedIndications' in indication_df.columns:
            indication_df['approvedIndications'] = indication_df['approvedIndications'].apply(
                lambda x: self.update_approved_indications(x, disease_mappings)
            )
        
        return indication_df


class NodeIndexMapper:
    """Creates node-to-index mappings for graph construction.
    
    Single Responsibility: Map node IDs to integer indices for PyTorch Geometric.
    """
    
    def __init__(self):
        """Initialize node index mapper."""
        pass
    
    def create_node_mappings(self, molecule_df, disease_df, gene_table, version):
        """Create node index mappings for all node types."""
        print("Creating node index mappings...")
        
        # Extract unique node lists
        approved_drugs_list = list(molecule_df['id'].unique())
        drug_type_list = list(molecule_df['drugType'].dropna().unique())
        gene_list = list(gene_table.column('id').unique().to_pylist())
        
        if isinstance(disease_df, pd.DataFrame):
            disease_list = list(disease_df['id'].unique())
        else:
            disease_list = disease_df.column('id').to_pylist()
        
        print(f"Diseases (core set with approved drugs): {len(disease_list)}")
        print("Note: Ancestor diseases will be used for similarity edges but NOT added as nodes")
        
        # Create reactome pathway list from gene data
        reactome_list = []
        pathways_column = None
        
        if 'pathways' in gene_table.column_names:
            pathways_column = gene_table.column('pathways').to_pylist()
        elif 'reactome' in gene_table.column_names:
            pathways_column = gene_table.column('reactome').to_pylist()
            
        if pathways_column:
            for pathways in pathways_column:
                if pathways:
                    if isinstance(pathways, str):
                        try:
                            pathways = ast.literal_eval(pathways)
                        except:
                            continue
                    if isinstance(pathways, list):
                        reactome_list.extend(pathways)
        
        reactome_list = list(set(reactome_list))
        
        # Create therapeutic area list from disease data
        therapeutic_area_list = []
        if isinstance(disease_df, pd.DataFrame) and 'therapeuticAreas' in disease_df.columns:
            for areas in disease_df['therapeuticAreas'].dropna():
                if isinstance(areas, str):
                    try:
                        areas = ast.literal_eval(areas)
                    except:
                        continue
                if isinstance(areas, list) or isinstance(areas, np.ndarray):
                    therapeutic_area_list.extend(areas)
        
        therapeutic_area_list = list(set(therapeutic_area_list))
        
        # Create index mappings
        drug_key_mapping = {drug: idx for idx, drug in enumerate(approved_drugs_list)}
        drug_type_key_mapping = {dtype: idx + len(approved_drugs_list) 
                                 for idx, dtype in enumerate(drug_type_list)}
        gene_key_mapping = {gene: idx + len(approved_drugs_list) + len(drug_type_list) 
                           for idx, gene in enumerate(gene_list)}
        reactome_key_mapping = {pathway: idx + len(approved_drugs_list) + len(drug_type_list) + len(gene_list)
                               for idx, pathway in enumerate(reactome_list)}
        disease_key_mapping = {disease: idx + len(approved_drugs_list) + len(drug_type_list) + len(gene_list) + len(reactome_list)
                              for idx, disease in enumerate(disease_list)}
        therapeutic_key_mapping = {area: idx + len(approved_drugs_list) + len(drug_type_list) + len(gene_list) + len(reactome_list) + len(disease_list)
                                  for idx, area in enumerate(therapeutic_area_list)}
        
        print(f"Created mappings for {len(approved_drugs_list)} drugs, {len(gene_list)} genes, {len(disease_list)} diseases")
        
        return {
            'drug_key_mapping': drug_key_mapping,
            'drug_type_key_mapping': drug_type_key_mapping,
            'gene_key_mapping': gene_key_mapping,
            'reactome_key_mapping': reactome_key_mapping,
            'disease_key_mapping': disease_key_mapping,
            'therapeutic_area_key_mapping': therapeutic_key_mapping,
            'approved_drugs_list': approved_drugs_list,
            'drug_type_list': drug_type_list,
            'gene_list': gene_list,
            'reactome_list': reactome_list,
            'disease_list': disease_list,
            'therapeutic_area_list': therapeutic_area_list
        }
