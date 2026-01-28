#!/usr/bin/env python3
"""
Create raw_data directory structure and download index.html files from OpenTargets.
This script sets up the complete folder structure needed for downloading all datasets.

Run this BEFORE download_parquet_files.py to prepare the directory structure.

Usage:
    uv run create_raw_data_folders.py
"""

import os
import sys
import requests
from pathlib import Path
from datetime import datetime

def download_index_file(url, local_path):
    """
    Download index.html file from OpenTargets FTP.
    
    Args:
        url: URL to download from
        local_path: Local path to save the file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Create parent directory if it doesn't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the index.html file
        with open(local_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Failed to download {url}")
        print(f"    Error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False


def create_dataset_structure(version, dataset_name, base_path):
    """
    Create directory structure and download index.html for a dataset.
    
    Args:
        version: OpenTargets version (e.g., '21.06')
        dataset_name: Name of the dataset
        base_path: Base path for raw_data
        
    Returns:
        bool: True if successful
    """
    # Construct URL and local path
    url = f"https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/{version}/output/etl/parquet/{dataset_name}/"
    local_dir = base_path / version / dataset_name
    local_index = local_dir / 'index.html'
    
    print(f"  {dataset_name:30s} ", end='', flush=True)
    
    # Create directory
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Download index.html
    if download_index_file(url, local_index):
        print("✓")
        return True
    else:
        return False


def main():
    """Main function to create raw_data folder structure."""
    print("="*80)
    print("OPENTARGETS RAW DATA FOLDER STRUCTURE CREATOR")
    print("="*80)
    print()
    
    # Base path for raw data
    base_path = Path('raw_data')
    
    # Define all datasets
    # Core datasets (used across all versions or specific versions)
    CORE_DATASETS = {
        '21.06': [
            'indication',
            'molecule',
            'diseases',
            'targets',
            'associationByOverallDirect',
        ],
        '23.06': [
            'indication',
        ],
        '24.06': [
            'indication',
        ],
    }
    
    # New datasets for richer node and edge features
    EDGE_FEATURE_DATASETS = [
        'mechanismOfAction',      
        'interaction',           
        'drugWarnings',          
        'knownDrugsAggregated',
    ]
    
    # Summary
    total_datasets = sum(len(datasets) for datasets in CORE_DATASETS.values())
    total_datasets += len(EDGE_FEATURE_DATASETS)
    
    print(f"This script will create directory structure for:")
    print(f"  • 3 OpenTargets versions (21.06, 23.06, 24.06)")
    print(f"  • {total_datasets} total datasets")
    print()
    print(f"Target directory: {base_path.absolute()}")
    print()
    
    # Confirm with user
    response = input("Continue? [Y/n]: ").strip().lower()
    if response and response not in ['y', 'yes']:
        print("Cancelled.")
        return
    
    print()
    print("="*80)
    print("CREATING DIRECTORY STRUCTURE")
    print("="*80)
    print()
    
    # Create base directory
    base_path.mkdir(exist_ok=True)
    
    # Track statistics
    success_count = 0
    total_count = 0
    failed_datasets = []
    
    # Process each version
    for version in ['21.06', '23.06', '24.06']:
        print(f"VERSION: {version}")
        print("-"*80)
        
        version_path = base_path / version
        version_path.mkdir(exist_ok=True)
        
        # Process core datasets for this version
        if version in CORE_DATASETS:
            print(f"\nCore datasets ({len(CORE_DATASETS[version])}):")
            for dataset in CORE_DATASETS[version]:
                total_count += 1
                if create_dataset_structure(version, dataset, base_path):
                    success_count += 1
                else:
                    failed_datasets.append(f"{version}/{dataset}")
        
        # Process edge feature datasets for this version
        if version == '21.06':
            print(f"\nEdge feature datasets ({len(EDGE_FEATURE_DATASETS)}):")
            for dataset in EDGE_FEATURE_DATASETS:
                total_count += 1
                if create_dataset_structure(version, dataset, base_path):
                    success_count += 1
                else:
                    failed_datasets.append(f"{version}/{dataset}")
        
        print()
    
    # Final summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print(f"Total datasets processed:  {total_count}")
    print(f"Successfully created:      {success_count}")
    print(f"Failed:                    {total_count - success_count}")
    print()
    
    if failed_datasets:
        print("Failed datasets:")
        for dataset in failed_datasets:
            print(f"  ✗ {dataset}")
        print()
        print("⚠️  Some datasets could not be downloaded.")
        print("   They may not exist in that OpenTargets version.")
        print("   This is normal - not all datasets exist in all versions.")
    else:
        print("✅ All datasets successfully prepared!")
    
    print()
    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print()
    print("Directory structure is now ready!")
    print()
    print("To download the actual parquet files, run:")
    print("  python download_parquet_files.py")
    print()
    print("Or run both steps together:")
    print("  python create_raw_data_folders.py && python download_parquet_files.py")
    print()
    
    # Create a marker file to indicate structure is ready
    marker_file = base_path / '.structure_created'
    with open(marker_file, 'w') as f:
        f.write(f"Structure created: {datetime.now()}\n")
        f.write(f"Total datasets: {total_count}\n")
        f.write(f"Successfully prepared: {success_count}\n")


if __name__ == "__main__":
    print()
    print("OpenTargets Raw Data Folder Structure Creator")
    print("=" * 80)
    print()
    
    # Check if requests is available
    try:
        import requests
    except ImportError:
        print("ERROR: 'requests' package is required but not installed.")
        print()
        print("Install it with:")
        print("  uv pip install requests")
        print()
        sys.exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
