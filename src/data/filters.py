"""
Data filtering module.
Handles filtering of molecules and associations based on validation criteria.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc


class MoleculeFilter:
    """Filters molecules based on validation criteria.
    
    Single Responsibility: Filter molecules to only include those with validated connections.
    """
    
    def __init__(self):
        """Initialize molecule filter."""
        pass
    
    def filter_linked_molecules(self, molecule_df, indication_df, known_drugs_df=None):
        """Filter molecules to only include those with validated connections."""
        print("Filtering molecules with validated connections...")
        # First, remove molecules with parent IDs (keep only parent molecules)
        if "parentId" in molecule_df.columns:
            molecule_df = molecule_df[pd.isna(molecule_df["parentId"])].copy()

        
        # Get molecules from indications
        indication_molecules = set()
        if 'id' in indication_df.columns:
            indication_molecules = set(indication_df['id'].unique())
        
        # Get molecules from known drugs (phase 3+)
        known_drug_molecules = set()
        if known_drugs_df is not None and not known_drugs_df.empty:
            if 'drugId' in known_drugs_df.columns and 'phase' in known_drugs_df.columns:
                known_drug_molecules = set(
                    known_drugs_df[known_drugs_df['phase'] >= 3]['drugId'].unique()
                )
        
        # Get molecules with metadata links
        metadata_molecules = set()
        if 'linkedDiseases' in molecule_df.columns:
            def has_linked_diseases(row):
                linked = row.get('linkedDiseases')
                if pd.isna(linked) or not linked:
                    return False
                if isinstance(linked, str):
                    import ast
                    try:
                        linked = ast.literal_eval(linked)
                    except:
                        return False
                return len(linked) > 0 if isinstance(linked, (list, dict)) else False
            
            metadata_molecules = set(
                molecule_df[molecule_df.apply(has_linked_diseases, axis=1)]['id'].unique()
            )
        
        # Combine all valid molecules
        valid_molecules = indication_molecules | known_drug_molecules | metadata_molecules
        
        print(f"  Molecules in Indications: {len(indication_molecules)}")
        print(f"  Molecules in Known Drugs (Ph 3+): {len(known_drug_molecules)}")
        print(f"  Molecules with Metadata Links: {len(metadata_molecules)}")
        print(f"  Total Unique Valid Molecules: {len(valid_molecules)}")
        
        # Filter molecule dataframe
        filtered_df = molecule_df[molecule_df['id'].isin(valid_molecules)].copy()
        print(f"Filtered to {len(filtered_df)} molecules with validated connections")
        
        return filtered_df


class AssociationFilter:
    """Filters associations by score and node membership.
    
    Single Responsibility: Filter disease-gene associations based on quality criteria.
    """
    
    def __init__(self):
        """Initialize association filter."""
        pass
    
    def filter_by_score(self, associations_table, score_column, threshold=0.1):
        """Filter associations by minimum score threshold."""
        filtered = associations_table.filter(
            pc.field(score_column) >= threshold
        )
        return filtered
    
    def filter_by_genes_and_diseases(self, associations_table, gene_ids, disease_ids, 
                                     score_column, threshold=0.1):
        """Filter associations by linked genes and diseases with score threshold."""
        print(f"Filtering associations by genes and diseases (threshold >= {threshold})...")
        
        # Convert to sets for faster lookup
        gene_set = set(gene_ids)
        disease_set = set(disease_ids)
        
        # Filter by score first
        filtered = self.filter_by_score(associations_table, score_column, threshold)
        
        # Convert to pandas for easier filtering
        df = filtered.to_pandas()
        
        # Filter by gene and disease membership
        df = df[
            df['targetId'].isin(gene_set) & 
            df['diseaseId'].isin(disease_set)
        ].copy()
        
        print(f"Filtered to {len(df)} high-quality associations (threshold >= {threshold})")
        
        # Convert back to PyArrow table
        return pa.Table.from_pandas(df)
