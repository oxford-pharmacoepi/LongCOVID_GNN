#!/usr/bin/env python3
"""
Script to download parquet files from OpenTargets by parsing HTML index files.
This script assumes the directory structure and index files have already been created
by running create_raw_data_folders.py first.
"""

import os
import re
import requests
from pathlib import Path
from tqdm import tqdm as tqdm_progress
import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import sys

def parse_html_index(html_content, base_url):
    """
    Parse HTML index file to extract parquet file URLs.
    
    Args:
        html_content (str): Content of the HTML index file
        base_url (str): Base URL for the directory
        
    Returns:
        list: List of parquet file URLs
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    parquet_files = []
    
    # Look for links to parquet files
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.endswith('.parquet'):
            # Construct full URL
            full_url = urljoin(base_url, href)
            parquet_files.append((href, full_url))
    
    # Alternative: regex approach if BeautifulSoup doesn't work well
    if not parquet_files:
        # Extract parquet file names using regex
        parquet_pattern = r'href="([^"]*\.parquet)"'
        matches = re.findall(parquet_pattern, html_content, re.IGNORECASE)
        for filename in matches:
            full_url = urljoin(base_url, filename)
            parquet_files.append((filename, full_url))
    
    return parquet_files

def download_file(url, local_path, max_retries=3):
    """
    Download a file with retry logic and progress bar.
    
    Args:
        url (str): URL to download
        local_path (Path): Local path to save the file
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        bool: True if successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            with open(local_path, 'wb') as f:
                with tqdm_progress(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=f"Downloading {local_path.name}",
                    leave=False
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed for {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed to download {url} after {max_retries} attempts")
                return False
        except Exception as e:
            print(f"Unexpected error downloading {url}: {e}")
            return False

def process_directory(parquet_dir):
    """
    Process a single parquet directory, parse index.html and download files.
    
    Args:
        parquet_dir (Path): Path to the parquet subdirectory
        
    Returns:
        tuple: (success_count, total_count)
    """
    index_file = parquet_dir / 'index.html'
    
    if not index_file.exists():
        print(f"No index.html found in {parquet_dir}")
        return 0, 0
    
    try:
        # Read the index.html file
        with open(index_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Construct the base URL
        # Extract from the directory structure
        rel_path = str(parquet_dir).split('raw_data/')[-1]
        # Fix: Add the missing /output/etl/ part to match OpenTargets URL structure
        base_url = f"https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/{rel_path.replace('/parquet/', '/output/etl/parquet/')}/"
        
        # Parse parquet files from HTML
        parquet_files = parse_html_index(html_content, base_url)
        
        # Debug: Print the first few URLs to verify they're correct
        print(f"Base URL: {base_url}")
        if parquet_files:
            print(f"Sample URLs (first 3):")
            for i, (filename, url) in enumerate(parquet_files[:3]):
                print(f"  {filename} -> {url}")
        
        if not parquet_files:
            print(f"No parquet files found in index for {parquet_dir}")
            return 0, 0
        
        print(f"\nProcessing {parquet_dir.name}: Found {len(parquet_files)} parquet files")
        
        success_count = 0
        
        # Download each parquet file
        for filename, url in tqdm_progress(parquet_files, desc=f"Files in {parquet_dir.name}"):
            local_path = parquet_dir / filename
            
            # Skip if file already exists and has reasonable size
            if local_path.exists() and local_path.stat().st_size > 1000:
                print(f"Skipping {filename} (already exists)")
                success_count += 1
                continue
            
            # Download the file
            if download_file(url, local_path):
                success_count += 1
            else:
                print(f"Failed to download {filename}")
        
        return success_count, len(parquet_files)
        
    except Exception as e:
        print(f"Error processing directory {parquet_dir}: {e}")
        return 0, 0

def main():
    """Main function to download parquet files from all OpenTargets versions."""
    print("🧬 OpenTargets Parquet File Downloader")
    print("=" * 50)
    
    # Check if raw_data structure exists
    if not Path("raw_data").exists():
        print("❌ raw_data directory not found!")
        print("Please run create_raw_data_folders.py first to create the directory structure.")
        sys.exit(1)
    
    # Find all parquet directories with index files across all versions
    all_parquet_dirs = []
    
    # Look for parquet directories in all version folders
    for version in ['21.06', '23.06', '24.06']:
        version_dir = Path(f"raw_data/{version}")
        if version_dir.exists():
            # Look for directories with index.html files
            for item in version_dir.rglob("index.html"):
                parquet_dir = item.parent
                # Only include directories that look like parquet directories
                if any(dataset in str(parquet_dir) for dataset in 
                      ['indication', 'molecule', 'diseases', 'targets', 'associationByOverallDirect']):
                    if parquet_dir not in all_parquet_dirs:
                        all_parquet_dirs.append(parquet_dir)
    
    if not all_parquet_dirs:
        print("❌ No parquet directories with index files found!")
        print("Please run create_raw_data_folders.py first to download index files.")
        sys.exit(1)
    
    print(f"Found {len(all_parquet_dirs)} directories to process:")
    for parquet_dir in sorted(all_parquet_dirs):
        print(f"  - {parquet_dir}")
    
    # Process each directory
    total_success = 0
    total_files = 0
    
    for parquet_dir in tqdm_progress(all_parquet_dirs, desc="Processing directories"):
        success, total = process_directory(parquet_dir)
        total_success += success
        total_files += total
        
        # Small delay between directories to be respectful to the server
        time.sleep(1)
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"🎯 PARQUET DOWNLOAD SUMMARY:")
    print(f"Total parquet files processed: {total_files}")
    print(f"Successfully downloaded: {total_success}")
    print(f"Failed downloads: {total_files - total_success}")
    
    if total_success == total_files:
        print("🎉 ALL PARQUET FILES DOWNLOADED SUCCESSFULLY!")
        
        # Restructure data to match expected format
        print("\n🔧 Restructuring data directories...")
        try:
            import shutil
            import glob
            
            for version in ['21.06', '23.06', '24.06']:
                parquet_dir = Path(f"raw_data/{version}/parquet")
                version_dir = Path(f"raw_data/{version}")
                
                if parquet_dir.exists():
                    print(f"   Restructuring {version}...")
                    
                    # Move all contents from parquet/ to version/
                    for item in parquet_dir.iterdir():
                        if item.is_dir():
                            target = version_dir / item.name
                            if target.exists():
                                shutil.rmtree(target)
                            shutil.move(str(item), str(target))
                            print(f"     Moved {item.name}/ to {version}/")
                    
                    # Remove empty parquet directory
                    if parquet_dir.exists() and not any(parquet_dir.iterdir()):
                        parquet_dir.rmdir()
                        print(f"     Removed empty parquet/ directory")
            
            print("   ✅ Data restructuring completed!")
            
        except Exception as e:
            print(f"   ⚠️ Error during restructuring: {e}")
            print("   You may need to manually run the restructuring commands:")
            print("   mv raw_data/21.06/parquet/* raw_data/21.06/")
            print("   rmdir raw_data/21.06/parquet")
            print("   (repeat for 23.06 and 24.06)")
        
        print("\n✅ You now have the complete OpenTargets dataset for:")
        print("   - 21.06: Full training dataset (indication, molecule, diseases, targets, associations)")
        print("   - 23.06: Validation dataset (indication only)")
        print("   - 24.06: Test dataset (indication only)")
    elif total_success > 0:
        print("⚠️  Some files downloaded successfully, but some failed")
    else:
        print("❌ No parquet files were downloaded successfully")

if __name__ == "__main__":
    # Check if required packages are available
    try:
        import requests
        from tqdm import tqdm as tqdm_progress
        from bs4 import BeautifulSoup
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install with: pip install requests tqdm beautifulsoup4")
        sys.exit(1)
    
    main()