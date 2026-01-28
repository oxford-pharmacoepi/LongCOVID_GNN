"""
Data storage module.
Handles saving and loading processed data and mappings.
"""

import os
import json
import pickle
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


class DataStorage:
    """Handles saving and loading processed data.
    
    Single Responsibility: Persist and retrieve processed data and mappings.
    """
    
    def __init__(self):
        """Initialize data storage."""
        pass
    
    def save_processed_data(self, data_dict, output_dir):
        """Save processed data tables for quick loading."""
        print(f"Saving processed data to {output_dir}/")
        os.makedirs(output_dir, exist_ok=True)
        
        for name, data in data_dict.items():
            filepath = os.path.join(output_dir, f"{name}.parquet")
            
            if isinstance(data, pd.DataFrame):
                # Save pandas DataFrame as parquet
                data.to_parquet(filepath, index=False)
            elif isinstance(data, pa.Table):
                # Save PyArrow table as parquet
                pq.write_table(data, filepath)
            else:
                print(f"Warning: Unknown data type for {name}, skipping")
        
        print("Processed data saved successfully")
    
    def load_processed_data(self, data_dir):
        """Load processed data from directory."""
        print(f"Loading processed data from {data_dir}/")
        data_dict = {}
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.parquet'):
                name = filename.replace('.parquet', '')
                filepath = os.path.join(data_dir, filename)
                
                # Load as pandas DataFrame
                data_dict[name] = pd.read_parquet(filepath)
        
        print(f"Loaded {len(data_dict)} processed data files")
        return data_dict
    
    def save_mappings(self, mappings, output_dir):
        """Save mappings to files for later use."""
        print(f"Saving mappings to {output_dir}/")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each mapping type
        for key, value in mappings.items():
            if isinstance(value, dict):
                # Save dictionaries as JSON
                filepath = os.path.join(output_dir, f"{key}.json")
                with open(filepath, 'w') as f:
                    json.dump(value, f, indent=2)
            elif isinstance(value, list):
                # Save lists as JSON
                filepath = os.path.join(output_dir, f"{key}.json")
                with open(filepath, 'w') as f:
                    json.dump(value, f, indent=2)
            else:
                # Save other types as pickle
                filepath = os.path.join(output_dir, f"{key}.pkl")
                with open(filepath, 'wb') as f:
                    pickle.dump(value, f)
        
        print("Mappings saved successfully")
    
    def load_mappings(self, mappings_path):
        """Load mappings from files."""
        print(f"Loading mappings from {mappings_path}/")
        mappings = {}
        
        if not os.path.exists(mappings_path):
            raise FileNotFoundError(f"Mappings directory not found: {mappings_path}")
        
        for filename in os.listdir(mappings_path):
            filepath = os.path.join(mappings_path, filename)
            name = os.path.splitext(filename)[0]
            
            if filename.endswith('.json'):
                with open(filepath, 'r') as f:
                    mappings[name] = json.load(f)
            elif filename.endswith('.pkl'):
                with open(filepath, 'rb') as f:
                    mappings[name] = pickle.load(f)
        
        print(f"Loaded {len(mappings)} mapping files")
        return mappings
