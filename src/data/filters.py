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
        
        # Remove molecules with parent IDs (keep only parent molecules)
        # EXACT COPY from original line 404
        molecule_df = molecule_df[pd.isna(molecule_df['parentId'])]
        
        # 1. Molecules from approved indications
        # EXACT COPY from original line 407
        unique_indication_ids = set(indication_df['id'].unique())
        
        # 2. Molecules from known drugs (Phase 3 or 4)
        # EXACT COPY from original lines 410-414
        unique_known_ids = set()
        if known_drugs_df is not None:
            # Filter for Phase 3 and 4
            valid_known = known_drugs_df[known_drugs_df['phase'] >= 3]
            unique_known_ids = set(valid_known['drugId'].unique())
        
        # 3. Molecules with pre-linked diseases in metadata
        # EXACT COPY from original lines 417-422
        def has_linked_diseases(row):
            if 'linkedDiseases' in row and isinstance(row['linkedDiseases'], dict):
                return row['linkedDiseases'].get('count', 0) > 0
            return False
            
        unique_metadata_ids = set(molecule_df[molecule_df.apply(has_linked_diseases, axis=1)]['id'].unique())
        
        # Combine all valid IDs
        # EXACT COPY from original line 425
        all_valid_ids = unique_indication_ids | unique_known_ids | unique_metadata_ids
        
        print(f"  Molecules in Indications: {len(unique_indication_ids)}")
        print(f"  Molecules in Known Drugs (Ph 3+): {len(unique_known_ids)}")
        print(f"  Molecules with Metadata Links: {len(unique_metadata_ids)}")
        print(f"  Total Unique Valid Molecules: {len(all_valid_ids)}")
        
        # Filter molecule_df
        # EXACT COPY from original line 433
        filtered_molecule_df = molecule_df[molecule_df['id'].isin(all_valid_ids)]
        
        print(f"Filtered to {len(filtered_molecule_df)} molecules with validated connections")
        return filtered_molecule_df


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
