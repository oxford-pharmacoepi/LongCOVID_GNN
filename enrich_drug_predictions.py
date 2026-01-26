#!/usr/bin/env python3
"""
Enrich Long COVID Drug Predictions with ChEMBL Data

This script takes the drug predictions CSV file and enriches it with actual drug names
by querying the ChEMBL database API.
"""

import pandas as pd
import requests
import time
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import json


def find_latest_predictions(results_dir='results/long_covid'):
    """Find the most recent predictions CSV file."""
    prediction_files = list(Path(results_dir).glob('long_covid_drug_predictions_*.csv'))
    
    if not prediction_files:
        raise FileNotFoundError(f"No prediction files found in {results_dir}")
    
    # Sort by modification time
    latest_file = max(prediction_files, key=lambda p: p.stat().st_mtime)
    print(f"Found latest predictions file: {latest_file}")
    return str(latest_file)


def query_chembl_molecule(chembl_id, max_retries=3):
    """
    Query ChEMBL API for molecule information.
    
    Args:
        chembl_id: ChEMBL ID (e.g., 'CHEMBL1234')
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dictionary with drug information or None if not found
    """
    base_url = "https://www.ebi.ac.uk/chembl/api/data/molecule"
    
    for attempt in range(max_retries):
        try:
            # Query the ChEMBL API
            response = requests.get(
                f"{base_url}/{chembl_id}.json",
                headers={'Accept': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                molecule = data.get('molecule_structures', {})
                properties = data.get('molecule_properties', {})
                
                # Extract relevant information
                info = {
                    'pref_name': data.get('pref_name', None),
                    'molecule_type': data.get('molecule_type', None),
                    'first_approval': data.get('first_approval', None),
                    'oral': data.get('oral', None),
                    'parenteral': data.get('parenteral', None),
                    'topical': data.get('topical', None),
                    'black_box_warning': data.get('black_box_warning', None),
                    'natural_product': data.get('natural_product', None),
                    'prodrug': data.get('prodrug', None),
                    'therapeutic_flag': data.get('therapeutic_flag', None),
                    'max_phase': data.get('max_phase', None),
                    'withdrawn_flag': data.get('withdrawn_flag', None),
                    'molecular_formula': properties.get('full_molformula', None),
                    'canonical_smiles': molecule.get('canonical_smiles', None),
                }
                
                # Try to get synonyms
                synonyms_url = f"{base_url}/{chembl_id}/molecule_synonyms.json"
                try:
                    syn_response = requests.get(synonyms_url, timeout=5)
                    if syn_response.status_code == 200:
                        syn_data = syn_response.json()
                        synonyms = syn_data.get('molecule_synonyms', [])
                        if synonyms:
                            # Prioritize trade names and INN
                            trade_names = [s['molecule_synonym'] for s in synonyms 
                                         if s.get('syn_type') == 'TRADE_NAME']
                            inn_names = [s['molecule_synonym'] for s in synonyms 
                                       if s.get('syn_type') == 'INN']
                            other_names = [s['molecule_synonym'] for s in synonyms 
                                         if s.get('syn_type') not in ['TRADE_NAME', 'INN']]
                            
                            info['trade_names'] = trade_names
                            info['inn_names'] = inn_names
                            info['other_synonyms'] = other_names[:5]  # Limit to first 5
                except:
                    pass
                
                return info
                
            elif response.status_code == 404:
                return None  # Molecule not found
            elif response.status_code == 429:
                # Rate limit - wait longer
                time.sleep(2 ** attempt)
                continue
            else:
                print(f"  Warning: Status {response.status_code} for {chembl_id}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"  Timeout for {chembl_id}, attempt {attempt + 1}/{max_retries}")
            time.sleep(1)
        except Exception as e:
            print(f"  Error querying {chembl_id}: {e}")
            return None
    
    return None


def get_best_drug_name(info):
    """
    Get the best human-readable drug name from ChEMBL info.
    
    Priority:
    1. Trade name (if available)
    2. INN name (International Nonproprietary Name)
    3. Preferred name
    4. ChEMBL ID (fallback)
    """
    if not info:
        return None
    
    # Check for trade names first
    if info.get('trade_names'):
        return info['trade_names'][0]
    
    # Then INN names
    if info.get('inn_names'):
        return info['inn_names'][0]
    
    # Then preferred name
    if info.get('pref_name'):
        return info['pref_name']
    
    return None


def enrich_predictions(input_file, output_file=None, top_n=None, cache_file=None):
    """
    Enrich drug predictions with ChEMBL data.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (optional)
        top_n: Only process top N predictions (optional)
        cache_file: Path to cache file for ChEMBL lookups (optional)
    """
    print(f"\nEnriching predictions from: {input_file}")
    
    # Load predictions
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} predictions")
    
    # Limit to top N if specified
    if top_n:
        df = df.head(top_n)
        print(f"Processing top {top_n} predictions")
    
    # Load cache if exists
    cache = {}
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cache from {cache_file}")
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached entries")
    
    # Query ChEMBL for each drug
    enriched_data = []
    
    print("\nQuerying ChEMBL database...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Looking up drugs"):
        chembl_id = row['drug_id']
        
        # Check cache first
        if chembl_id in cache:
            info = cache[chembl_id]
        else:
            info = query_chembl_molecule(chembl_id)
            cache[chembl_id] = info
            
            # Rate limiting - be respectful to the API
            time.sleep(0.5)
        
        # Get best name
        best_name = get_best_drug_name(info)
        
        # Create enriched row
        enriched_row = row.to_dict()
        enriched_row['actual_drug_name'] = best_name if best_name else chembl_id
        
        if info:
            enriched_row['molecule_type'] = info.get('molecule_type')
            enriched_row['max_phase'] = info.get('max_phase')
            enriched_row['first_approval'] = info.get('first_approval')
            enriched_row['therapeutic_flag'] = info.get('therapeutic_flag')
            enriched_row['withdrawn_flag'] = info.get('withdrawn_flag')
            enriched_row['oral'] = info.get('oral')
            enriched_row['parenteral'] = info.get('parenteral')
            enriched_row['topical'] = info.get('topical')
            enriched_row['black_box_warning'] = info.get('black_box_warning')
            
            # Add trade names and synonyms as comma-separated strings
            if info.get('trade_names'):
                enriched_row['trade_names'] = ', '.join(info['trade_names'][:3])
            if info.get('inn_names'):
                enriched_row['inn_names'] = ', '.join(info['inn_names'][:3])
        
        enriched_data.append(enriched_row)
    
    # Create enriched dataframe
    enriched_df = pd.DataFrame(enriched_data)
    
    # Reorder columns to put actual_drug_name after drug_id
    cols = list(enriched_df.columns)
    cols.remove('actual_drug_name')
    cols.insert(cols.index('drug_name') + 1, 'actual_drug_name')
    enriched_df = enriched_df[cols]
    
    # Save cache
    if cache_file:
        print(f"\nSaving cache to {cache_file}")
        os.makedirs(os.path.dirname(cache_file) or '.', exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
    
    # Save enriched predictions
    if output_file is None:
        # Generate output filename
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_enriched{input_path.suffix}")
    
    enriched_df.to_csv(output_file, index=False)
    print(f"\nEnriched predictions saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("ENRICHMENT SUMMARY")
    print("="*80)
    
    found_names = enriched_df['actual_drug_name'].notna().sum()
    print(f"Successfully enriched: {found_names}/{len(enriched_df)} ({found_names/len(enriched_df)*100:.1f}%)")
    
    # Show top 10 with names
    print("\nTop 10 predictions with drug names:")
    print("-"*80)
    display_cols = ['rank', 'drug_id', 'actual_drug_name', 'probability', 'max_phase', 'first_approval']
    display_cols = [c for c in display_cols if c in enriched_df.columns]
    print(enriched_df[display_cols].head(10).to_string(index=False))
    print("="*80)
    
    return enriched_df


def main():
    parser = argparse.ArgumentParser(
        description='Enrich Long COVID drug predictions with ChEMBL data'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input CSV file (auto-detects latest if not provided)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output CSV file (auto-generates if not provided)'
    )
    parser.add_argument(
        '--top-n', '-n',
        type=int,
        help='Only process top N predictions (default: all)'
    )
    parser.add_argument(
        '--cache',
        type=str,
        default='results/long_covid/chembl_cache.json',
        help='Cache file for ChEMBL lookups (default: results/long_covid/chembl_cache.json)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching'
    )
    
    args = parser.parse_args()
    
    # Find input file
    if args.input:
        input_file = args.input
    else:
        input_file = find_latest_predictions()
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    # Set cache file
    cache_file = None if args.no_cache else args.cache
    
    # Enrich predictions
    enriched_df = enrich_predictions(
        input_file=input_file,
        output_file=args.output,
        top_n=args.top_n,
        cache_file=cache_file
    )
    
    print("\n✓ Enrichment complete!")


if __name__ == "__main__":
    main()
