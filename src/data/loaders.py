"""
Data loading module for OpenTargets datasets.
Handles loading raw Parquet files from OpenTargets platform.
"""

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pandas as pd
from pathlib import Path


class OpenTargetsLoader:
    """Loads raw OpenTargets parquet files.
    
    Single Responsibility: Load and return raw data tables without transformation.
    """
    
    def __init__(self, config=None):
        """Initialize loader with optional configuration."""
        self.config = config
    
    def load_indication_data(self, path):
        """Load and preprocess indication data."""
        print(f"Loading indication data from {path}")
        indication_table = ds.dataset(path, format='parquet').to_table()
        
        # Filter to only approved indications
        indication_table = indication_table.filter(
            pc.field('maxPhaseForIndication') == 4
        )
        
        return indication_table
    
    def load_molecule_data(self, path):
        """Load and preprocess molecule data."""
        print(f"Loading molecule data from {path}")
        molecule_table = ds.dataset(path, format='parquet').to_table()
        
        # Convert to pandas for easier manipulation
        molecule_df = molecule_table.to_pandas()
        
        # Filter to small molecules only
        molecule_df = molecule_df[molecule_df['drugType'] == 'Small molecule'].copy()
        
        # Select relevant columns
        relevant_columns = [
            'id', 'name', 'drugType', 'maximumClinicalTrialPhase',
            'hasBeenWithdrawn', 'blackBoxWarning', 'yearOfFirstApproval'
        ]
        
        # Add linkedDiseases if it exists
        if 'linkedDiseases' in molecule_df.columns:
            relevant_columns.append('linkedDiseases')
        
        # Filter to only existing columns
        existing_columns = [col for col in relevant_columns if col in molecule_df.columns]
        molecule_df = molecule_df[existing_columns].copy()
        
        return molecule_df
    
    def load_disease_data(self, path):
        """Load and preprocess disease data."""
        print(f"Loading disease data from {path}")
        disease_table = ds.dataset(path, format='parquet').to_table()
        
        # Convert to pandas
        disease_df = disease_table.to_pandas()
        
        # Select relevant columns
        relevant_columns = ['id', 'name', 'therapeuticAreas', 'parents']
        existing_columns = [col for col in relevant_columns if col in disease_df.columns]
        disease_df = disease_df[existing_columns].copy()
        
        # Filter out specific therapeutic areas if configured
        if hasattr(self, 'config') and self.config:
            excluded_areas = [
                'MONDO_0024458',  # Injury or poisoning
                'OTAR_0000018',   # Genetic, familial or congenital disease
                'EFO_0000651',    # Phenotype
                'EFO_0001444',    # Measurement
            ]
            
            if 'therapeuticAreas' in disease_df.columns:
                def has_excluded_area(areas):
                    if pd.isna(areas) or not areas:
                        return False
                    if isinstance(areas, str):
                        import ast
                        try:
                            areas = ast.literal_eval(areas)
                        except:
                            return False
                    return any(area in excluded_areas for area in areas)
                
                disease_df = disease_df[~disease_df['therapeuticAreas'].apply(has_excluded_area)].copy()
        
        return disease_df
    
    def load_gene_data(self, path, version):
        """Load and preprocess gene/target data."""
        print(f"Loading gene data from {path}")
        gene_table = ds.dataset(path, format='parquet').to_table()
        
        # Version-specific column selection
        if version >= 22.04:
            relevant_columns = ['id', 'approvedSymbol', 'biotype', 'pathways']
        else:
            relevant_columns = ['id', 'approvedSymbol', 'bioType', 'pathways']
        
        # Select only existing columns
        existing_columns = [col for col in relevant_columns if col in gene_table.column_names]
        gene_table = gene_table.select(existing_columns)
        
        return gene_table
    
    def load_associations_data(self, path, version):
        """Load and preprocess associations data."""
        print(f"Loading associations data from {path}")
        associations_table = ds.dataset(path, format='parquet').to_table()
        
        # Determine score column based on version
        if version >= 22.04:
            score_column = 'score'
        else:
            score_column = 'score'  # Same for now, but kept for clarity
        
        # Select relevant columns
        relevant_columns = ['diseaseId', 'targetId', score_column]
        existing_columns = [col for col in relevant_columns if col in associations_table.column_names]
        associations_table = associations_table.select(existing_columns)
        
        return associations_table, score_column
    
    def load_mechanism_of_action(self, path):
        """
        Load mechanismOfAction dataset for drug-gene edge features.
        
        Returns:
            DataFrame with columns: [chemblIds, actionType, mechanismOfAction, targetName, targets]
        """
        print(f"Loading mechanismOfAction data from {path}")
        
        try:
            moa_table = ds.dataset(path, format='parquet').to_table()
            moa_df = moa_table.to_pandas()
            
            # Select relevant columns
            relevant_columns = ['chemblIds', 'actionType', 'mechanismOfAction', 'targetName', 'targets']
            existing_columns = [col for col in relevant_columns if col in moa_df.columns]
            moa_df = moa_df[existing_columns].copy()
            
            print(f"Loaded {len(moa_df)} mechanism of action records")
            return moa_df
            
        except Exception as e:
            print(f"Warning: Could not load mechanism of action data: {e}")
            return pd.DataFrame()
    
    def load_drug_warnings(self, path):
        """
        Load drugWarnings dataset for enhanced drug node features.
        
        Returns:
            DataFrame with drug warnings beyond black box warnings
        """
        print(f"Loading drugWarnings data from {path}")
        
        try:
            warnings_table = ds.dataset(path, format='parquet').to_table()
            warnings_df = warnings_table.to_pandas()
            print(f"Loaded {len(warnings_df)} drug warning records")
            return warnings_df
        except Exception as e:
            print(f"Warning: Could not load drug warnings data: {e}")
            return pd.DataFrame()
    
    def load_interaction(self, path):
        """
        Load interaction dataset for drug-drug edges.
        
        Returns:
            DataFrame with drug-drug interaction information
        """
        print(f"Loading interaction data from {path}")
        
        try:
            interaction_table = ds.dataset(path, format='parquet').to_table()
            interaction_df = interaction_table.to_pandas()
            print(f"Loaded {len(interaction_df)} drug interaction records")
            return interaction_df
        except Exception as e:
            print(f"Warning: Could not load interaction data: {e}")
            return pd.DataFrame()
    
    def load_known_drugs_aggregated(self, path):
        """
        Load knownDrugsAggregated dataset for clinical trial phase information.
        
        Returns:
            DataFrame with clinical trial phases and approval status
        """
        print(f"Loading knownDrugsAggregated data from {path}")
        
        try:
            known_drugs_table = ds.dataset(path, format='parquet').to_table()
            known_drugs_df = known_drugs_table.to_pandas()
            print(f"Loaded {len(known_drugs_df)} known drug records with clinical phase info")
            return known_drugs_df
        except Exception as e:
            print(f"Warning: Could not load known drugs data: {e}")
            return pd.DataFrame()
