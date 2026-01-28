#!/usr/bin/env python3
"""
Search for specific drugs in Long COVID predictions


Usage:
    python search_drug_predictions.py CHEMBL1234 CHEMBL5678
    python search_drug_predictions.py --list drugs.txt
"""

import sys
import pandas as pd
from pathlib import Path
import argparse


def search_drugs(drug_ids, predictions_file=None):
    """Search for specific drugs in predictions"""
    
    # Find latest predictions file if not specified
    if predictions_file is None:
        results_dir = Path("results/long_covid")
        # Try both naming patterns
        csv_files = list(results_dir.glob("long_covid_drug_predictions_*.csv"))
        if not csv_files:
            csv_files = list(results_dir.glob("long_covid_*_drugs_*.csv"))
        if not csv_files:
            print("   No prediction files found in results/long_covid/")
            print("   Run the drug repurposing script first.")
            return
        # Filter out enriched files to get the base predictions
        base_files = [f for f in csv_files if '_enriched' not in f.name]
        if base_files:
            predictions_file = max(base_files, key=lambda p: p.stat().st_mtime)
        else:
            predictions_file = max(csv_files, key=lambda p: p.stat().st_mtime)
        print(f"Using: {predictions_file.name}\n")
    
    # Load predictions
    df = pd.read_csv(predictions_file)
    
    # Search for each drug
    found_count = 0
    not_found = []
    
    for drug_id in drug_ids:
        drug_id = drug_id.strip().upper()
        if not drug_id.startswith('CHEMBL'):
            drug_id = f'CHEMBL{drug_id}'
        
        matches = df[df['drug_id'] == drug_id]
        
        if len(matches) > 0:
            row = matches.iloc[0]
            found_count += 1
            print(f"  {drug_id}")
            print(f"   Rank:        {row['rank']}/{len(df)}")
            print(f"   Probability: {row['probability']:.4f}")
            print(f"   Score:       {row['score']:.4f}")
            if 'drug_name' in row and row['drug_name'] != drug_id:
                print(f"   Name:        {row['drug_name']}")
            if 'approval_status' in row:
                print(f"   Status:      {row['approval_status']}")
            
            # Percentile
            percentile = (1 - row['rank'] / len(df)) * 100
            print(f"   Percentile:  {percentile:.1f}th")
            print()
        else:
            not_found.append(drug_id)
    
    # Summary
    print("=" * 60)
    print(f"Found: {found_count}/{len(drug_ids)}")
    
    if not_found:
        print(f"\n  Not found in database ({len(not_found)}):")
        for drug_id in not_found:
            print(f"   - {drug_id}")
        print("\nThese drugs were not in the training data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Search for drugs in predictions')
    parser.add_argument('drugs', nargs='*', help='ChEMBL IDs to search for')
    parser.add_argument('--list', type=str, help='File with list of ChEMBL IDs (one per line)')
    parser.add_argument('--file', type=str, help='Specific predictions CSV file to search')
    
    args = parser.parse_args()
    
    # Collect drug IDs
    drug_ids = []
    
    if args.list:
        # Read the file and handle two-column format
        with open(args.list, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Skip header line (if it contains "drug_id" or "drug_name")
                if i == 0 and ('drug_id' in line.lower() or 'drug_name' in line.lower()):
                    continue
                
                # Split by whitespace and take first column (drug_id)
                parts = line.split()
                if parts:
                    drug_id = parts[0].strip()
                    # Only add if it looks like a ChEMBL ID
                    if drug_id.upper().startswith('CHEMBL'):
                        drug_ids.append(drug_id)
    
    if args.drugs:
        drug_ids.extend(args.drugs)
    
    if not drug_ids:
        print("Usage: python search_drug_predictions.py CHEMBL1234 CHEMBL5678")
        print("   or: python search_drug_predictions.py --list drugs.txt")
        sys.exit(1)
    
    search_drugs(drug_ids, args.file)
