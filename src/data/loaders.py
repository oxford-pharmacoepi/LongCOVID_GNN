"""
OpenTargets data loading module.
Handles loading and preprocessing of raw Parquet data files.
"""

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pandas as pd
import glob


class OpenTargetsLoader:
    """Loads and preprocesses OpenTargets Parquet data.
    
    Single Responsibility: Load raw data files and apply basic preprocessing.
    """
    
    def __init__(self, config=None):
        """Initialize the loader."""
        self.config = config
    
    def load_indication_data(self, path):
        """Load and preprocess indication data."""
        print(f"Loading indication data from {path}")
        
        # EXACT COPY from original lines 119-132
        # Use glob pattern to only read parquet files, excluding index.html
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
        
        # EXACT COPY from original lines 139-163
        # Use glob pattern to only read parquet files, excluding index.html
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
        
        # EXACT COPY from original lines 170-216
        # Use glob pattern to only read parquet files, excluding index.html
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
        
        # EXACT COPY from original lines 223-242
        # Use glob pattern to only read parquet files, excluding index.html
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
        
        # EXACT COPY from original lines 245-263
        # Use glob pattern to only read parquet files, excluding index.html
        parquet_files = glob.glob(f"{path}/*.parquet")
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {path}")
        
        associations_dataset = ds.dataset(parquet_files, format="parquet")
        associations_table = associations_dataset.to_table()
        
        # Version-specific score column
        if version == 21.04 or version == 21.06:
            score_column = 'score'
        else:
            score_column = 'datasourceScores.overall'
        
        return associations_table, score_column
    
    def load_known_drugs_aggregated(self, path):
        """Load known drugs aggregated data."""
        print(f"Loading knownDrugsAggregated data from {path}")
        
        # EXACT COPY from original lines 266-280
        # Use glob pattern to only read parquet files, excluding index.html
        parquet_files = glob.glob(f"{path}/*.parquet")
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {path}")
        
        known_drugs_dataset = ds.dataset(parquet_files, format="parquet")
        known_drugs_table = known_drugs_dataset.to_table()
        
        # Convert to pandas for easier manipulation
        known_drugs_df = known_drugs_table.to_pandas()
        print(f"Loaded {len(known_drugs_df)} known drug records with clinical phase info")
        
        return known_drugs_df
    
    def load_mechanism_of_action(self, path):
        """Load mechanism of action data."""
        print(f"Loading mechanism of action data from {path}")
        
        # EXACT COPY from original lines 283-296
        # Use glob pattern to only read parquet files, excluding index.html
        parquet_files = glob.glob(f"{path}/*.parquet")
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {path}")
        
        moa_dataset = ds.dataset(parquet_files, format="parquet")
        moa_table = moa_dataset.to_table()
        
        # Convert to pandas for easier manipulation
        moa_df = moa_table.to_pandas()
        
        return moa_df

    def load_interaction(self, path):
        """Load gene-gene interaction data."""
        print(f"Loading interaction data from {path}")
        
        parquet_files = glob.glob(f"{path}/*.parquet")
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {path}")
        
        interaction_dataset = ds.dataset(parquet_files, format="parquet")
        interaction_table = interaction_dataset.to_table()
        
        return interaction_table
