"""
Data processing module for drug-disease prediction.
Handles all data loading, preprocessing, and transformation operations.
"""

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pandas as pd
import numpy as np
import ast
import os
import json
import pickle
from pathlib import Path
from tqdm import tqdm
import polars as pl

from .utils import set_seed, extract_edges


class DataProcessor:
    """Main class for data loading and preprocessing operations."""
    
    def __init__(self, config):
        self.config = config
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
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except:
                return []
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, list):
            return value
        return [value] if value is not None else []
    
    def update_approved_indications(self, disease_list, mapping_dict):
        """Update disease IDs inside lists using mapping dictionary."""
        if not isinstance(disease_list, list):
            return disease_list
        return [mapping_dict.get(str(d), str(d)) for d in disease_list]
    
    def load_indication_data(self, path):
        """Load and preprocess indication data."""
        print(f"Loading indication data from {path}")
        
        # Use glob pattern to only read parquet files, excluding index.html
        import glob
        parquet_files = glob.glob(f"{path}/*.parquet")
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {path}")
        
        indication_dataset = ds.dataset(parquet_files, format="parquet")
        indication_table = indication_dataset.to_table()
        
        # Filter for drugs with approved indications
        expr = pc.list_value_length(pc.field("approvedIndications")) > 0 
        filtered_indication_table = indication_table.filter(expr)
        
        return filtered_indication_table
    
    def load_molecule_data(self, path):
        """Load and preprocess molecule data."""
        print(f"Loading molecule data from {path}")
        
        # Use glob pattern to only read parquet files, excluding index.html
        import glob
        parquet_files = glob.glob(f"{path}/*.parquet")
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {path}")
        
        molecule_dataset = ds.dataset(parquet_files, format="parquet")
        molecule_table = molecule_dataset.to_table()
        
        # Clean drug type data
        drug_type_column = pc.replace_substring(
            molecule_table.column('drugType'), 'unknown', 'Unknown'
        )
        fill_value = pa.scalar('Unknown', type=pa.string())
        molecule_table = molecule_table.drop_columns("drugType").add_column(
            3, "drugType", drug_type_column.fill_null(fill_value)
        )
        
        # Select relevant columns
        filtered_molecule_table = molecule_table.select([
            'id', 'name', 'drugType', 'blackBoxWarning', 'yearOfFirstApproval',
            'parentId', 'childChemblIds', 'linkedDiseases', 'hasBeenWithdrawn', 'linkedTargets'
        ]).flatten().drop_columns(['linkedTargets.count', 'linkedDiseases.count'])
        
        return filtered_molecule_table
    
    def load_disease_data(self, path):
        """Load and preprocess disease data."""
        print(f"Loading disease data from {path}")
        
        # Use glob pattern to only read parquet files, excluding index.html
        import glob
        parquet_files = glob.glob(f"{path}/*.parquet")
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {path}")
        
        disease_dataset = ds.dataset(parquet_files, format="parquet")
        disease_table = disease_dataset.to_table()
        
        # Filter out diseases without therapeutic areas
        disease_table = disease_table.filter(
            pc.list_value_length(pc.field("therapeuticAreas")) > 0
        )
        
        # Filter out specific therapeutic area
        df = disease_table.to_pandas()
        filtered_df = df[~df['therapeuticAreas'].apply(lambda x: 'EFO_0001444' in x)]
        disease_table = pa.Table.from_pandas(filtered_df)
        
        # Select relevant columns
        disease_table = disease_table.select([
            'id', 'name', 'description', 'ancestors', 'descendants', 'children', 'therapeuticAreas'
        ])
        
        # Filter out unwanted prefixes
        prefixes_to_remove = ["UBERON", "ZFA", "CL", "GO", "FBbt", "FMA"]
        filter_conditions = [
            pc.starts_with(disease_table.column('id'), prefix) 
            for prefix in prefixes_to_remove
        ]
        
        combined_filter = filter_conditions[0]
        for condition in filter_conditions[1:]:
            combined_filter = pc.or_(combined_filter, condition)
        
        negated_filter = pc.invert(combined_filter)
        filtered_disease_table = disease_table.filter(negated_filter)
        
        # Additional filtering
        filtered_disease_table = filtered_disease_table.filter(
            pc.list_value_length(pc.field("descendants")) == 0
        )
        filtered_disease_table = filtered_disease_table.filter(
            pc.field("id") != "EFO_0000544"
        )
        
        return filtered_disease_table
    
    def load_gene_data(self, path, version):
        """Load and preprocess gene/target data."""
        print(f"Loading gene data from {path}")
        
        # Use glob pattern to only read parquet files, excluding index.html
        import glob
        parquet_files = glob.glob(f"{path}/*.parquet")
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {path}")
        
        gene_dataset = ds.dataset(parquet_files, format="parquet")
        gene_table = gene_dataset.to_table().flatten().flatten()
        
        # Version-specific column selection
        if version == 21.04 or version == 21.06:
            filtered_gene_table = gene_table.select([
                'id', 'approvedName','bioType', 'proteinAnnotations.functions', 'reactome'
            ]).flatten()
        else:
            filtered_gene_table = gene_table.select([
                'id', 'approvedName','biotype', 'functionDescriptions', 'proteinIds', 'pathways'
            ]).flatten()
        
        return filtered_gene_table
    
    def load_associations_data(self, path, version):
        """Load and preprocess associations data."""
        print(f"Loading associations data from {path}")
        
        # Use glob pattern to only read parquet files, excluding index.html
        import glob
        parquet_files = glob.glob(f"{path}/*.parquet")
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {path}")
        
        associations_dataset = ds.dataset(parquet_files, format="parquet")
        associations_table = associations_dataset.to_table()
        
        # Find score column
        score_column = None
        for col in associations_table.column_names:
            if "Score" in col or "score" in col:
                score_column = col
                break
        
        if score_column is None:
            raise ValueError("No score column found in associations data")
        
        # Select relevant columns
        if version == 21.04:
            associations_table = associations_table.select(['diseaseId', 'targetId', score_column])
        else:
            associations_table = associations_table.select(['diseaseId', 'targetId', score_column])
        
        return associations_table, score_column
    
    def load_mechanism_of_action(self, path):
        """
        Load mechanismOfAction dataset for drug-gene edge features.
        
        Returns:
            DataFrame with columns: [chemblIds, actionType, mechanismOfAction, targetName, targets]
        """
        print(f"Loading mechanismOfAction data from {path}")
        
        import glob
        parquet_files = glob.glob(f"{path}/*.parquet")
        
        if not parquet_files:
            print(f"Warning: No mechanismOfAction data found in {path}")
            return pd.DataFrame()
        
        moa_dataset = ds.dataset(parquet_files, format="parquet")
        moa_table = moa_dataset.to_table()
        
        # Select relevant columns
        # chemblIds: drug ID, actionType: mechanism type (inhibitor, etc.), targets: gene IDs
        moa_df = moa_table.to_pandas()
        
        print(f"Loaded {len(moa_df)} mechanism of action records")
        return moa_df
    
    def load_drug_warnings(self, path):
        """
        Load drugWarnings dataset for enhanced drug node features.
        
        Returns:
            DataFrame with drug warnings beyond black box warnings
        """
        print(f"Loading drugWarnings data from {path}")
        
        import glob
        parquet_files = glob.glob(f"{path}/*.parquet")
        
        if not parquet_files:
            print(f"Warning: No drugWarnings data found in {path}")
            return pd.DataFrame()
        
        warnings_dataset = ds.dataset(parquet_files, format="parquet")
        warnings_table = warnings_dataset.to_table()
        warnings_df = warnings_table.to_pandas()
        
        print(f"Loaded {len(warnings_df)} drug warning records")
        return warnings_df
    
    def load_interaction(self, path):
        """
        Load interaction dataset for drug-drug edges.
        
        Returns:
            DataFrame with drug-drug interaction information
        """
        print(f"Loading interaction data from {path}")
        
        import glob
        parquet_files = glob.glob(f"{path}/*.parquet")
        
        if not parquet_files:
            print(f"Warning: No interaction data found in {path}")
            return pd.DataFrame()
        
        interaction_dataset = ds.dataset(parquet_files, format="parquet")
        interaction_table = interaction_dataset.to_table()
        interaction_df = interaction_table.to_pandas()
        
        print(f"Loaded {len(interaction_df)} drug-drug interaction records")
        return interaction_df
    
    def load_known_drugs_aggregated(self, path):
        """
        Load knownDrugsAggregated dataset for clinical trial phase information.
        
        Returns:
            DataFrame with clinical trial phases and approval status
        """
        print(f"Loading knownDrugsAggregated data from {path}")
        
        import glob
        parquet_files = glob.glob(f"{path}/*.parquet")
        
        if not parquet_files:
            print(f"Warning: No knownDrugsAggregated data found in {path}")
            return pd.DataFrame()
        
        kda_dataset = ds.dataset(parquet_files, format="parquet")
        kda_table = kda_dataset.to_table()
        kda_df = kda_table.to_pandas()
        
        print(f"Loaded {len(kda_df)} known drug records with clinical phase info")
        return kda_df
    
    def apply_id_mappings(self, molecule_df, indication_df):
        """Apply redundant ID mappings to clean the data."""
        print("Applying ID mappings for data consistency...")
        
        # Apply drug ID mappings
        drug_mappings = self.redundant_id_mapping['drug_mappings']
        id_to_parentid_mapping = {
            k: self.resolve_mapping(v, drug_mappings) 
            for k, v in drug_mappings.items()
        }
        
        # Update drug IDs
        indication_df['id'] = indication_df['id'].apply(
            lambda x: self.resolve_mapping(x, id_to_parentid_mapping) 
            if x in id_to_parentid_mapping else x
        )
        
        # Apply disease ID mappings
        disease_mappings = self.redundant_id_mapping['disease_mappings']
        indication_df['approvedIndications'] = indication_df['approvedIndications'].apply(
            self.safe_list_conversion
        )
        indication_df['approvedIndications'] = indication_df['approvedIndications'].apply(
            lambda x: self.update_approved_indications(x, disease_mappings)
        )
        
        return molecule_df, indication_df
    
    def filter_linked_molecules(self, molecule_df, indication_df, known_drugs_df=None):
        """Filter molecules to only include those with validated connections."""
        print("Filtering molecules with validated connections...")
        
        # Remove molecules with parent IDs (keep only parent molecules)
        molecule_df = molecule_df[pd.isna(molecule_df['parentId'])]
        
        # 1. Molecules from approved indications
        unique_indication_ids = set(indication_df['id'].unique())
        
        # 2. Molecules from known drugs (Phase 3 or 4)
        unique_known_ids = set()
        if known_drugs_df is not None:
            # Filter for Phase 3 and 4
            valid_known = known_drugs_df[known_drugs_df['phase'] >= 3]
            unique_known_ids = set(valid_known['drugId'].unique())
        
        # 3. Molecules with pre-linked diseases in metadata
        def has_linked_diseases(row):
            if 'linkedDiseases' in row and isinstance(row['linkedDiseases'], dict):
                return row['linkedDiseases'].get('count', 0) > 0
            return False
            
        unique_metadata_ids = set(molecule_df[molecule_df.apply(has_linked_diseases, axis=1)]['id'].unique())
        
        # Combine all valid IDs
        all_valid_ids = unique_indication_ids | unique_known_ids | unique_metadata_ids
        
        print(f"  Molecules in Indications: {len(unique_indication_ids)}")
        print(f"  Molecules in Known Drugs (Ph 3+): {len(unique_known_ids)}")
        print(f"  Molecules with Metadata Links: {len(unique_metadata_ids)}")
        print(f"  Total Unique Valid Molecules: {len(all_valid_ids)}")
        
        # Filter molecule_df
        filtered_molecule_df = molecule_df[molecule_df['id'].isin(all_valid_ids)]
        
        print(f"Filtered to {len(filtered_molecule_df)} molecules with validated connections")
        return filtered_molecule_df
    
    def create_gene_reactome_mapping(self, gene_table, version):
        """Create gene-reactome pathway mappings based on version."""
        print("Creating gene-reactome mappings...")
        
        if version == 21.04 or version == 21.06:
            gene_reactome_table = gene_table.select(['id', 'reactome']).flatten()
        else:
            # Handle newer format with pathway dictionaries
            gene_reactome_df = gene_table.select(['id', 'pathways']).flatten().to_pandas()
            exploded = gene_reactome_df.explode('pathways')
            exploded['pathwayId'] = exploded['pathways'].apply(
                lambda x: x['pathwayId'] if pd.notnull(x) and isinstance(x, dict) else None
            )
            final_df = exploded[['id', 'pathwayId']].dropna()
            gene_reactome_table = pa.Table.from_pandas(final_df)
        
        return gene_reactome_table
    
    def filter_associations_by_genes_and_diseases(self, associations_table, gene_ids, disease_ids, score_column, threshold=0.1):
        """Filter associations by linked genes and diseases with score threshold."""
        print(f"Filtering associations by genes and diseases (threshold >= {threshold})...")
        
        # Filter for genes linked with approved drugs
        gene_filter_mask = pc.is_in(
            associations_table.column('targetId'), 
            value_set=pa.array(gene_ids)
        )
        gene_filtered_associations = associations_table.filter(gene_filter_mask)
        
        # Filter for diseases with approved drugs
        disease_filter_mask = pc.is_in(
            gene_filtered_associations.column('diseaseId'), 
            value_set=pa.array(disease_ids)
        )
        filtered_associations = gene_filtered_associations.filter(disease_filter_mask)
        
        # Apply score threshold
        score_threshold = pc.field(score_column) >= threshold
        filtered_associations = filtered_associations.filter(score_threshold)
        
        print(f"Filtered to {len(filtered_associations)} high-quality associations (threshold >= {threshold})")
        return filtered_associations
    
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
        
        # Extract reactome pathways
        if version == 21.04 or version == 21.06:
            reactome = gene_table.column('reactome').combine_chunks().flatten()
        else:
            pathways_column = gene_table.column('pathways').combine_chunks().flatten()
            reactome = pathways_column.field(0) if len(pathways_column) > 0 else pa.array([])
        
        reactome_list = list(reactome.unique().to_pylist()) if len(reactome) > 0 else []
        
        # Extract therapeutic areas - handle pandas DataFrame
        if isinstance(disease_df, pd.DataFrame):
            # Convert pandas DataFrame to PyArrow Table for consistent processing
            disease_table = pa.Table.from_pandas(disease_df)
            therapeutic_area = disease_table.column('therapeuticAreas').combine_chunks().flatten()
        else:
            # Already a PyArrow Table
            therapeutic_area = disease_df.column('therapeuticAreas').combine_chunks().flatten()
        
        therapeutic_area_list = list(therapeutic_area.unique().to_pylist())
        
        # Create sequential index mappings
        current_index = 0
        
        drug_key_mapping = {approved_drugs_list[i]: i for i in range(len(approved_drugs_list))}
        current_index += len(drug_key_mapping)
        
        drug_type_key_mapping = {drug_type_list[i]: i + current_index for i in range(len(drug_type_list))}
        current_index += len(drug_type_key_mapping)
        
        gene_key_mapping = {gene_list[i]: i + current_index for i in range(len(gene_list))}
        current_index += len(gene_key_mapping)
        
        reactome_key_mapping = {reactome_list[i]: i + current_index for i in range(len(reactome_list))}
        current_index += len(reactome_key_mapping)
        
        disease_key_mapping = {disease_list[i]: i + current_index for i in range(len(disease_list))}
        current_index += len(disease_key_mapping)
        
        therapeutic_area_key_mapping = {therapeutic_area_list[i]: i + current_index for i in range(len(therapeutic_area_list))}
        
        mappings = {
            'drug_key_mapping': drug_key_mapping,
            'drug_type_key_mapping': drug_type_key_mapping,
            'gene_key_mapping': gene_key_mapping,
            'reactome_key_mapping': reactome_key_mapping,
            'disease_key_mapping': disease_key_mapping,
            'therapeutic_area_key_mapping': therapeutic_area_key_mapping,
            'approved_drugs_list': approved_drugs_list,
            'drug_type_list': drug_type_list,
            'gene_list': gene_list,
            'reactome_list': reactome_list,
            'disease_list': disease_list,
            'therapeutic_area_list': therapeutic_area_list
        }
        
        print(f"Created mappings for {len(approved_drugs_list)} drugs, {len(gene_list)} genes, {len(disease_list)} diseases")
        return mappings
    
    def generate_validation_test_splits(self, config, mappings, train_edges_set):
        """Generate validation and test edge splits using different OpenTargets versions."""
        print("Generating validation and test edge splits...")
        
        # Load validation data
        val_indication_table = self.load_indication_data(config.paths['val_indication'])
        
        # Filter for approved drugs
        approved_drugs_array = pa.array(mappings['approved_drugs_list'])
        expr1 = pc.is_in(val_indication_table.column('id'), value_set=approved_drugs_array)
        val_filtered_indication_table = val_indication_table.filter(expr1)
        val_molecule_disease_table = val_filtered_indication_table.select(['id', 'approvedIndications']).flatten()
        
        # Extract validation edges
        all_val_md_edges_set = extract_edges(
            val_molecule_disease_table, 
            mappings['drug_key_mapping'], 
            mappings['disease_key_mapping'], 
            return_edge_set=True
        )
        
        # Find new validation edges (not in training)
        new_val_edges_set = all_val_md_edges_set - train_edges_set
        
        # Load test data
        test_indication_table = self.load_indication_data(config.paths['test_indication'])
        expr2 = pc.is_in(test_indication_table.column('id'), value_set=approved_drugs_array)
        test_filtered_indication_table = test_indication_table.filter(expr2)
        test_molecule_disease_table = test_filtered_indication_table.select(['id', 'approvedIndications']).flatten()
        
        # Extract test edges
        all_test_md_edges_set = extract_edges(
            test_molecule_disease_table, 
            mappings['drug_key_mapping'], 
            mappings['disease_key_mapping'], 
            return_edge_set=True
        )
        
        # Find new test edges (not in training or validation)
        new_test_edges_set = all_test_md_edges_set - train_edges_set - all_val_md_edges_set
        
        print(f"Validation set: {len(new_val_edges_set)} new edges")
        print(f"Test set: {len(new_test_edges_set)} new edges")
        
        return new_val_edges_set, new_test_edges_set
    
    def save_mappings(self, mappings, output_dir):
        """Save mappings to files for later use."""
        print(f"Saving mappings to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON files - include both mappings and lists
        mapping_files = [
            'drug_key_mapping', 'drug_type_key_mapping', 'gene_key_mapping',
            'reactome_key_mapping', 'disease_key_mapping', 'therapeutic_area_key_mapping'
        ]
        
        for mapping_name in mapping_files:
            if mapping_name in mappings:
                filepath = os.path.join(output_dir, f"{mapping_name}.json")
                with open(filepath, 'w') as f:
                    json.dump(mappings[mapping_name], f, indent=2)
        
        # Generate and save the lists from the mappings
        if 'drug_key_mapping' in mappings:
            mappings['approved_drugs_list'] = list(mappings['drug_key_mapping'].keys())
        if 'drug_type_key_mapping' in mappings:
            mappings['drug_type_list'] = list(mappings['drug_type_key_mapping'].keys())
        if 'gene_key_mapping' in mappings:
            mappings['gene_list'] = list(mappings['gene_key_mapping'].keys())
        if 'reactome_key_mapping' in mappings:
            mappings['reactome_list'] = list(mappings['reactome_key_mapping'].keys())
        if 'disease_key_mapping' in mappings:
            mappings['disease_list'] = list(mappings['disease_key_mapping'].keys())
        if 'therapeutic_area_key_mapping' in mappings:
            mappings['therapeutic_area_list'] = list(mappings['therapeutic_area_key_mapping'].keys())
        
        # Save the lists as JSON files
        list_files = [
            'approved_drugs_list', 'drug_type_list', 'gene_list',
            'reactome_list', 'disease_list', 'therapeutic_area_list'
        ]
        
        for list_name in list_files:
            if list_name in mappings:
                filepath = os.path.join(output_dir, f"{list_name}.json")
                with open(filepath, 'w') as f:
                    json.dump(mappings[list_name], f, indent=2)
        
        # Save complete mappings as pickle
        filepath = os.path.join(output_dir, "all_mappings.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(mappings, f)
        
        print("Mappings saved successfully")
    
    def load_mappings(self, mappings_path):
        """Load mappings from files."""
        print(f"Loading mappings from {mappings_path}")
        
        if os.path.isfile(mappings_path) and mappings_path.endswith('.pkl'):
            # Load from pickle file
            with open(mappings_path, 'rb') as f:
                mappings = pickle.load(f)
        else:
            # Load from JSON files in directory
            mappings = {}
            mapping_files = [
                'drug_key_mapping', 'drug_type_key_mapping', 'gene_key_mapping',
                'reactome_key_mapping', 'disease_key_mapping', 'therapeutic_area_key_mapping',
                'approved_drugs_list', 'drug_type_list', 'gene_list',
                'reactome_list', 'disease_list', 'therapeutic_area_list'
            ]
            
            for mapping_name in mapping_files:
                filepath = os.path.join(mappings_path, f"{mapping_name}.json")
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        mappings[mapping_name] = json.load(f)
        
        print("Mappings loaded successfully")
        return mappings
    
    def save_processed_data(self, data_dict, output_dir):
        """Save processed data tables for quick loading."""
        print(f"Saving processed data to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for name, data in data_dict.items():
            if isinstance(data, pd.DataFrame):
                filepath = os.path.join(output_dir, f"{name}.csv")
                data.to_csv(filepath, index=False)
            elif isinstance(data, pa.Table):
                filepath = os.path.join(output_dir, f"{name}.parquet")
                data.to_pandas().to_parquet(filepath)
        
        print("Processed data saved successfully")

def detect_data_mode(config, force_mode=None):
    """
    Detect whether to use raw or processed data mode.
    
    Parameters:
        config: Configuration object with paths
        force_mode: Optional string to force mode - either "raw" or "processed"
    """
    if force_mode:
        if force_mode == "raw":
            print("Forced raw data mode - using Option 1 workflow")
            return "raw"
        elif force_mode == "processed":
            print("Forced processed data mode - using Option 2 workflow") 
            return "processed"
        else:
            raise ValueError(f"Invalid force_mode: {force_mode}. Must be 'raw' or 'processed'")
    
    # Default auto-detection logic (prioritise raw data for temporal validation)
    processed_path = config.paths['processed']
    
    # Check if pre-processed data exists
    processed_files_exist = (
        os.path.exists(f"{processed_path}tables/processed_molecules.csv") and
        os.path.exists(f"{processed_path}mappings/drug_key_mapping.json") and
        os.path.exists(f"{processed_path}edges/1_molecule_drugType_edges.pt")
    )
    
    # Check if raw data exists
    raw_files_exist = (
        os.path.exists(config.paths['indication']) and
        os.path.exists(config.paths['molecule']) and
        os.path.exists(config.paths['diseases']) and
        os.path.exists(config.paths['targets']) and
        os.path.exists(config.paths['associations'])
    )
    
    # Prioritise raw data when both exist (for temporal validation)
    if raw_files_exist:
        print("Raw OpenTargets data detected - using Option 1 workflow")
        return "raw"
    elif processed_files_exist:
        print("Pre-processed data detected - using Option 2 workflow")
        return "processed"
    else:
        raise FileNotFoundError(
            "Neither pre-processed nor raw data found. "
            "Please follow the README instructions for data preparation."
        )


def create_full_dataset(config):
    """Main function to create complete processed dataset."""
    print("Creating full processed dataset...")
    
    # Initialise processor
    processor = DataProcessor(config)
    
    # Load raw data
    indication_table = processor.load_indication_data(config.paths['indication'])
    molecule_table = processor.load_molecule_data(config.paths['molecule'])
    disease_table = processor.load_disease_data(config.paths['disease'])
    gene_table = processor.load_gene_data(config.paths['gene'], config.training_version)
    associations_table, score_column = processor.load_associations_data(
        config.paths['associations'], config.training_version
    )
    
    # Convert to dataframes for processing
    indication_df = indication_table.to_pandas()
    molecule_df = molecule_table.to_pandas()
    disease_df = disease_table.to_pandas()
    
    # Load other drug-disease datasets
    known_drugs_df = processor.load_known_drugs_aggregated(config.paths['knownDrugsAggregated']) if 'knownDrugsAggregated' in config.paths else None

    # Apply ID mappings
    molecule_df, indication_df = processor.apply_id_mappings(molecule_df, indication_df)
    
    # Filter molecules (now more inclusive)
    molecule_df = processor.filter_linked_molecules(molecule_df, indication_df, known_drugs_df)
    
    # Create node mappings
    mappings = processor.create_node_mappings(
        molecule_df, disease_df, gene_table, config.training_version
    )
    
    # Save processed data and mappings
    processed_data = {
        'processed_molecules': molecule_df,
        'processed_indications': indication_df,
        'processed_diseases': disease_df,
        'processed_genes': gene_table.to_pandas(),
        'processed_associations': associations_table.to_pandas(),
        'processed_known_drugs': known_drugs_df if known_drugs_df is not None else pd.DataFrame()
    }
    
    processor.save_processed_data(processed_data, f"{config.paths['processed']}tables/")
    processor.save_mappings(mappings, f"{config.paths['processed']}mappings/")
    
    print("Full dataset processing completed!")
    return processed_data, mappings
