import pandas as pd
import networkx as nx
from torch_geometric.utils import to_networkx
import json
from collections import defaultdict
import itertools
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
import numpy as np
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Global variables (will be set by load_essential_components)
idx_to_name = {}
idx_to_type = {}
drug_key_mapping = {}
drug_type_key_mapping = {}
gene_key_mapping = {}
reactome_key_mapping = {}
disease_key_mapping = {}
therapeutic_area_key_mapping = {}
approved_drugs_list = []
approved_drugs_list_name = []
disease_list = []
disease_list_name = []
gene_list = []
reactome_list = []
therapeutic_area_list = []
molecule_drugType_edge_list = []
disease_therapeutic_edge_list = []

# Set these at the top of your script
results_path = r"C:\\OpenTargets_datasets\\FP_results\\"
datetime = r"20250806003500"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the TransformerConv model class
class TransformerModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5):
        super(TransformerModel, self).__init__()
        self.num_layers = num_layers

        # Initial TransformerConv layer with concat=False
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=4, concat=False)

        # Additional TransformerConv layers
        self.conv_list = torch.nn.ModuleList(
            [TransformerConv(hidden_channels, hidden_channels, heads=4, concat=False) for _ in range(num_layers - 1)]
        )

        # Layer normalization and dropout
        self.ln = torch.nn.LayerNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Final output layer
        self.final_layer = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # First TransformerConv layer
        x = self.conv1(x, edge_index)
        x = self.ln(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Additional TransformerConv layers
        for k in range(self.num_layers - 1):
            x = self.conv_list[k](x, edge_index)
            x = self.ln(x)
            if k < self.num_layers - 2:  # Apply activation and dropout except on the last hidden layer
                x = F.relu(x)
                x = self.dropout(x)

        # Final layer to produce output
        x = self.final_layer(x)
        return x
def load_therapeutic_area_mapping(results_path):
    """Load therapeutic area ID to name mapping from therapeutic_area_table.xlsx"""
    print("📊 Loading therapeutic area mapping...")

    therapeutic_area_mapping = {}

    try:
        ta_table_path = os.path.join(results_path, "therapeutic_area_table.xlsx")
        df = pd.read_excel(ta_table_path, engine='openpyxl')

        print(f"   ✓ Loaded therapeutic area table with {len(df)} entries")

        for _, row in df.iterrows():
            ta_id = str(row['ID']).strip()
            ta_name = str(row['Therapeutic Area Name']).strip()

            if pd.notna(ta_id) and pd.notna(ta_name):
                therapeutic_area_mapping[ta_id] = ta_name

        print(f"   ✓ Created mapping for {len(therapeutic_area_mapping)} therapeutic area entries")
        sample_items = list(therapeutic_area_mapping.items())[:5]
        print(f"   ✓ Sample mappings: {dict(sample_items)}")

    except FileNotFoundError:
        print(f"   ⚠️ Warning: Therapeutic area file not found at {ta_table_path}")
        print("   → Therapeutic area IDs will be displayed instead of names")
    except Exception as e:
        print(f"   ⚠️ Warning: Error loading therapeutic area mapping: {e}")
        print("   → Therapeutic area IDs will be displayed instead of names")

    return therapeutic_area_mapping


def load_gene_symbols_mapping(results_path):
    """Load gene ID to symbol mapping from the filtered_gene_table.csv"""
    
    print("📊 Loading gene symbols mapping...")
    
    gene_symbols_mapping = {}
    therapeutic_symbols_mapping = {}
    
    try:
        # Load the gene table CSV
        gene_table_path = os.path.join(results_path, "filtered_gene_table.csv")
        df = pd.read_csv(gene_table_path)
        
        print(f"   ✓ Loaded gene table with {len(df)} genes")
        
        # Create mapping from Ensembl ID to approved symbol
        for _, row in df.iterrows():
            gene_id = row['id']  # Ensembl ID like 'ENSG00000004948'
            gene_symbol = row['approvedSymbol']  # Gene symbol like 'CALCR'
            
            if pd.notna(gene_symbol) and pd.notna(gene_id):
                # Clean the gene ID (remove any whitespace/newlines)
                clean_gene_id = str(gene_id).strip()
                clean_gene_symbol = str(gene_symbol).strip()
                
                gene_symbols_mapping[clean_gene_id] = clean_gene_symbol
                
                # Also create variations in case network uses different formats
                # Some networks might have version numbers like ENSG00000004948.1
                if '.' in clean_gene_id:
                    base_id = clean_gene_id.split('.')[0]
                    gene_symbols_mapping[base_id] = clean_gene_symbol
                else:
                    # Also add with version .1 in case network has versions
                    versioned_id = clean_gene_id + ".1"
                    gene_symbols_mapping[versioned_id] = clean_gene_symbol
        
        print(f"   ✓ Created mapping for {len(gene_symbols_mapping)} gene symbol entries")
        
        # Show sample mappings for debugging
        sample_items = list(gene_symbols_mapping.items())[:5]
        print(f"   ✓ Sample mappings: {dict(sample_items)}")
        
    except FileNotFoundError:
        print(f"   ⚠️ Warning: Gene table file not found at {gene_table_path}")
        print("   → Gene IDs will be displayed instead of symbols")
    except Exception as e:
        print(f"   ⚠️ Warning: Error loading gene symbols: {e}")
        print("   → Gene IDs will be displayed instead of symbols")
    
    return gene_symbols_mapping


def create_enhanced_idx_mappings(gene_symbols_mapping, therapeutic_symbols_mapping):
    """Create enhanced idx_to_name mappings with gene symbols and therapeutic area names"""
    global idx_to_name, idx_to_type
    
    idx_to_name = {}
    idx_to_type = {}
    
    print("🔨 Creating enhanced idx_to_name mappings with gene symbols and therapeutic area names...")
    
    # Map drugs (approved drugs)
    for i, drug_name in enumerate(approved_drugs_list_name):
        idx_to_name[i] = drug_name
        idx_to_type[i] = "Drug"
    
    # Map diseases
    for disease_id, idx in disease_key_mapping.items():
        # Find the disease name
        if disease_id in disease_list:
            disease_pos = disease_list.index(disease_id)
            disease_name = disease_list_name[disease_pos]
        else:
            disease_name = f"Disease_{disease_id}"
        
        idx_to_name[idx] = disease_name
        idx_to_type[idx] = "Disease"
    
    # Map genes with enhanced symbols (THIS IS THE PATTERN TO FOLLOW)
    genes_with_symbols = 0
    genes_without_symbols = 0
    
    # Debug: Show some example gene IDs from the network
    print("🔍 DEBUG: Sample gene IDs from network:")
    sample_gene_ids = list(gene_key_mapping.keys())[:10]
    for gene_id in sample_gene_ids:
        symbol = gene_symbols_mapping.get(gene_id, "NOT_FOUND")
        print(f"   Network gene ID: '{gene_id}' → Symbol: '{symbol}'")
    
    # Debug: Show some example gene IDs from CSV
    print("🔍 DEBUG: Sample gene symbols available:")
    sample_csv_ids = list(gene_symbols_mapping.keys())[:5]
    for gene_id in sample_csv_ids:
        symbol = gene_symbols_mapping[gene_id]
        print(f"   CSV gene ID: '{gene_id}' → Symbol: '{symbol}'")
    
    for gene_id, idx in gene_key_mapping.items():
        # Try to get the approved symbol first
        if gene_id in gene_symbols_mapping:
            gene_name = gene_symbols_mapping[gene_id]
            genes_with_symbols += 1
        else:
            # Fallback to gene ID if symbol not found
            gene_name = gene_id
            genes_without_symbols += 1
        
        idx_to_name[idx] = gene_name
        idx_to_type[idx] = "Gene"
    
    # Map reactome pathways
    for reactome_id, idx in reactome_key_mapping.items():
        reactome_name = reactome_id  # Use pathway ID as name
        idx_to_name[idx] = reactome_name
        idx_to_type[idx] = "Pathway"
    
    # Map drug types
    for drug_type, idx in drug_type_key_mapping.items():
        idx_to_name[idx] = drug_type
        idx_to_type[idx] = "DrugType"
    
    # FIXED: Map therapeutic areas with names (FOLLOWING THE GENE PATTERN)
    therapeutic_area_with_names = 0
    therapeutic_area_without_names = 0
    
    # Debug: Show some example therapeutic area IDs from the network
    print("🔍 DEBUG: Sample therapeutic area IDs from network:")
    sample_ta_ids = list(therapeutic_area_key_mapping.keys())[:10]
    for ta_id in sample_ta_ids:
        ta_name = therapeutic_symbols_mapping.get(ta_id, "NOT_FOUND")
        print(f"   Network TA ID: '{ta_id}' → Name: '{ta_name}'")
    
    # Debug: Show some example therapeutic area mappings from Excel
    print("🔍 DEBUG: Sample therapeutic area names available:")
    sample_excel_ids = list(therapeutic_symbols_mapping.keys())[:5]
    for ta_id in sample_excel_ids:
        ta_name = therapeutic_symbols_mapping[ta_id]
        print(f"   Excel TA ID: '{ta_id}' → Name: '{ta_name}'")
    
    for ta_id, idx in therapeutic_area_key_mapping.items():
        # Try to get the therapeutic area name first (SAME PATTERN AS GENES)
        if ta_id in therapeutic_symbols_mapping:
            ta_name = therapeutic_symbols_mapping[ta_id]
            therapeutic_area_with_names += 1
        else:
            # Fallback to ID if name not found
            ta_name = ta_id
            therapeutic_area_without_names += 1
        
        idx_to_name[idx] = ta_name
        idx_to_type[idx] = "TherapeuticArea"
    
    print(f"✅ Created enhanced mappings for {len(idx_to_name)} nodes")
    print(f"   • {sum(1 for t in idx_to_type.values() if t == 'Drug')} Drugs")
    print(f"   • {sum(1 for t in idx_to_type.values() if t == 'Disease')} Diseases")
    print(f"   • {genes_with_symbols + genes_without_symbols} Genes ({genes_with_symbols} with symbols, {genes_without_symbols} with IDs)")
    print(f"   • {sum(1 for t in idx_to_type.values() if t == 'Pathway')} Pathways")
    print(f"   • {sum(1 for t in idx_to_type.values() if t == 'DrugType')} Drug Types")
    print(f"   • {therapeutic_area_with_names + therapeutic_area_without_names} Therapeutic Areas ({therapeutic_area_with_names} with names, {therapeutic_area_without_names} with IDs)")
    
    # Final debug: Show some examples of final mappings
    print("🔍 DEBUG: Final gene name mappings (first 5):")
    gene_indices = [idx for idx, node_type in idx_to_type.items() if node_type == "Gene"][:5]
    for idx in gene_indices:
        print(f"   Node index {idx} → '{idx_to_name[idx]}' (type: {idx_to_type[idx]})")
    
    # Final debug: Show some examples of final therapeutic area mappings
    print("🔍 DEBUG: Final therapeutic area mappings (first 5):")
    ta_indices = [idx for idx, node_type in idx_to_type.items() if node_type == "TherapeuticArea"][:5]
    for idx in ta_indices:
        print(f"   Node index {idx} → '{idx_to_name[idx]}' (type: {idx_to_type[idx]})")



def debug_therapeutic_area_mapping(results_path):
    """Debug therapeutic area mapping to find ID format mismatches"""
    
    print("🔍 DEBUGGING THERAPEUTIC AREA MAPPING")
    print("=" * 50)
    
    # 1. Load the Excel file and see what IDs look like
    try:
        ta_table_path = os.path.join(results_path, "therapeutic_area_table.xlsx")
        df = pd.read_excel(ta_table_path, engine='openpyxl')
        
        print(f"📊 Excel file loaded: {len(df)} rows")
        print(f"📊 Column names: {list(df.columns)}")
        
        # Show first few rows
        print("\n📋 First 10 rows from Excel:")
        for i, row in df.head(10).iterrows():
            ta_id = row['ID']
            ta_name = row['Therapeutic Area Name']
            print(f"   Row {i}: ID='{ta_id}' (type: {type(ta_id)}), Name='{ta_name}'")
        
        # Create the mapping
        excel_mapping = {}
        for _, row in df.iterrows():
            ta_id = str(row['ID']).strip()
            ta_name = str(row['Therapeutic Area Name']).strip()
            if pd.notna(ta_id) and pd.notna(ta_name):
                excel_mapping[ta_id] = ta_name
        
        print(f"\n✅ Created Excel mapping with {len(excel_mapping)} entries")
        
        # Show some sample mappings from Excel
        print("\n📋 Sample Excel mappings:")
        for i, (ta_id, ta_name) in enumerate(list(excel_mapping.items())[:5]):
            print(f"   Excel: '{ta_id}' → '{ta_name}'")
        
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        return None
    
    # 2. Load the network mappings and see what IDs look like there
    try:
        print(f"\n🔍 Network therapeutic area IDs from therapeutic_area_key_mapping:")
        if 'therapeutic_area_key_mapping' in globals():
            network_ids = list(therapeutic_area_key_mapping.keys())
            print(f"   Total network IDs: {len(network_ids)}")
            
            # Show first 10 network IDs
            print(f"\n📋 First 10 network IDs:")
            for i, ta_id in enumerate(network_ids[:10]):
                print(f"   Network: '{ta_id}' (type: {type(ta_id)})")
            
            # Check for matches
            print(f"\n🔍 MATCHING CHECK:")
            matches = 0
            no_matches = 0
            
            print(f"Checking first 10 network IDs against Excel mapping...")
            for ta_id in network_ids[:10]:
                if ta_id in excel_mapping:
                    print(f"   ✅ MATCH: '{ta_id}' → '{excel_mapping[ta_id]}'")
                    matches += 1
                else:
                    print(f"   ❌ NO MATCH: '{ta_id}' not found in Excel")
                    no_matches += 1
                    
                    # Try some variations
                    variations_to_try = [
                        str(ta_id).strip(),
                        ta_id.upper() if isinstance(ta_id, str) else str(ta_id).upper(),
                        ta_id.lower() if isinstance(ta_id, str) else str(ta_id).lower(),
                        f"EFO:{ta_id.split('_')[1]}" if isinstance(ta_id, str) and '_' in ta_id else None,
                        ta_id.replace('_', ':') if isinstance(ta_id, str) else None,
                        ta_id.replace(':', '_') if isinstance(ta_id, str) else None,
                    ]
                    
                    found_match = False
                    for variation in variations_to_try:
                        if variation and variation in excel_mapping:
                            print(f"      🔧 FOUND VARIATION: '{ta_id}' → '{variation}' → '{excel_mapping[variation]}'")
                            found_match = True
                            break
                    
                    if not found_match:
                        print(f"      🔍 Tried variations but no match found")
            
            print(f"\n📊 Summary: {matches} matches, {no_matches} no matches out of first 10")
            
            # Try to find the specific EFO_0000651 you mentioned
            print(f"\n🎯 Looking for 'EFO_0000651' specifically:")
            if 'EFO_0000651' in network_ids:
                print(f"   ✅ Found 'EFO_0000651' in network IDs")
                if 'EFO_0000651' in excel_mapping:
                    print(f"   ✅ Found 'EFO_0000651' in Excel mapping: '{excel_mapping['EFO_0000651']}'")
                else:
                    print(f"   ❌ 'EFO_0000651' NOT found in Excel mapping")
                    # Try variations
                    variations = ['EFO:0000651', 'efo_0000651', 'EFO_0000651']
                    for var in variations:
                        if var in excel_mapping:
                            print(f"   🔧 Found variation '{var}': '{excel_mapping[var]}'")
            else:
                print(f"   ❌ 'EFO_0000651' not found in network IDs")
        
        else:
            print("❌ therapeutic_area_key_mapping not found in globals")
            
    except Exception as e:
        print(f"❌ Error checking network mappings: {e}")
    
    # 3. Return the excel mapping for use
    return excel_mapping


def create_fixed_therapeutic_area_mapping(results_path):
    """Create a fixed therapeutic area mapping that handles ID format differences"""
    
    print("🔧 Creating FIXED therapeutic area mapping...")
    
    therapeutic_area_mapping = {}
    
    try:
        # Load Excel file
        ta_table_path = os.path.join(results_path, "therapeutic_area_table.xlsx")
        df = pd.read_excel(ta_table_path, engine='openpyxl')
        
        # Create multiple format mappings
        for _, row in df.iterrows():
            ta_id = str(row['ID']).strip()
            ta_name = str(row['Therapeutic Area Name']).strip()
            
            if pd.notna(ta_id) and pd.notna(ta_name):
                # Store the original ID format
                therapeutic_area_mapping[ta_id] = ta_name
                
                # ALSO store common variations
                if ':' in ta_id:
                    # EFO:0000651 → EFO_0000651
                    underscore_version = ta_id.replace(':', '_')
                    therapeutic_area_mapping[underscore_version] = ta_name
                    
                if '_' in ta_id:
                    # EFO_0000651 → EFO:0000651
                    colon_version = ta_id.replace('_', ':')
                    therapeutic_area_mapping[colon_version] = ta_name
                
                # Also try uppercase/lowercase variations
                therapeutic_area_mapping[ta_id.upper()] = ta_name
                therapeutic_area_mapping[ta_id.lower()] = ta_name
        
        print(f"   ✅ Created mapping with {len(therapeutic_area_mapping)} entries (including variations)")
        
        # Test with the specific ID you mentioned
        test_ids = ['EFO_0000651', 'EFO:0000651', 'efo_0000651']
        print(f"\n🧪 Testing specific IDs:")
        for test_id in test_ids:
            if test_id in therapeutic_area_mapping:
                print(f"   ✅ '{test_id}' → '{therapeutic_area_mapping[test_id]}'")
            else:
                print(f"   ❌ '{test_id}' not found")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return therapeutic_area_mapping


def test_mapping_on_network():
    """Test the mapping on actual network data"""
    
    print("\n🧪 TESTING MAPPING ON ACTUAL NETWORK DATA")
    print("=" * 50)
    
    # Get the fixed mapping
    fixed_mapping = create_fixed_therapeutic_area_mapping(results_path)
    
    if 'therapeutic_area_key_mapping' in globals() and fixed_mapping:
        print(f"🔍 Testing on {len(therapeutic_area_key_mapping)} network therapeutic areas...")
        
        found_count = 0
        not_found_count = 0
        
        for ta_id, idx in list(therapeutic_area_key_mapping.items())[:20]:  # Test first 20
            if ta_id in fixed_mapping:
                mapped_name = fixed_mapping[ta_id]
                print(f"   ✅ '{ta_id}' → '{mapped_name}'")
                found_count += 1
            else:
                print(f"   ❌ '{ta_id}' → NO MAPPING FOUND")
                not_found_count += 1
        
        print(f"\n📊 Results: {found_count} found, {not_found_count} not found")
        
        # Test specifically for EFO_0000651
        if 'EFO_0000651' in therapeutic_area_key_mapping:
            if 'EFO_0000651' in fixed_mapping:
                print(f"🎯 EFO_0000651 → '{fixed_mapping['EFO_0000651']}'")
            else:
                print(f"🎯 EFO_0000651 → STILL NOT MAPPED!")
    
    return fixed_mapping

# UPDATED load_essential_components function with the fix

def load_essential_components(results_path, datetime):
    """Load essential components with FIXED therapeutic area mapping"""
    global drug_key_mapping, drug_type_key_mapping, gene_key_mapping
    global reactome_key_mapping, disease_key_mapping, therapeutic_area_key_mapping
    global approved_drugs_list, approved_drugs_list_name, disease_list, disease_list_name
    global gene_list, reactome_list, therapeutic_area_list
    global molecule_drugType_edge_list, disease_therapeutic_edge_list
    
    print("🔄 Loading essential components with FIXED therapeutic area mapping...")
    
    # 1. Load mappings first
    mappings_path = f"{results_path}all_mappings_{datetime}.pkl"
    try:
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        
        # Extract mappings to global scope
        drug_key_mapping = mappings['drug_key_mapping']
        drug_type_key_mapping = mappings['drug_type_key_mapping']
        gene_key_mapping = mappings['gene_key_mapping']
        reactome_key_mapping = mappings['reactome_key_mapping']
        disease_key_mapping = mappings['disease_key_mapping']
        therapeutic_area_key_mapping = mappings['therapeutic_area_key_mapping']
        approved_drugs_list = mappings['approved_drugs_list']
        approved_drugs_list_name = mappings['approved_drugs_list_name']
        disease_list = mappings['disease_list']
        disease_list_name = mappings['disease_list_name']
        gene_list = mappings['gene_list']
        reactome_list = mappings['reactome_list']
        therapeutic_area_list = mappings['therapeutic_area_list']
        
        try:
            molecule_drugType_edge_list = mappings.get('molecule_drugType_edge_list', [])
            disease_therapeutic_edge_list = mappings.get('disease_therapeutic_edge_list', [])
        except:
            print("⚠️ Warning: Edge lists not found in mappings")
            molecule_drugType_edge_list = []
            disease_therapeutic_edge_list = []
        
        # Load gene symbols mapping (this works)
        gene_symbols_mapping = load_gene_symbols_mapping(results_path)
        
        # 🔧 USE THE FIXED therapeutic area mapping
        print("\n🔧 Using FIXED therapeutic area mapping...")
        therapeutic_area_mapping = load_therapeutic_area_mapping(results_path)
        
        # Debug the mapping before using it
        debug_therapeutic_area_mapping(results_path)
        
        # Test the mapping on network data
        test_mapping_on_network()
        
        # Create enhanced mappings with the fixed therapeutic area mapping
        create_enhanced_idx_mappings(gene_symbols_mapping, therapeutic_area_mapping)
        print("✅ Mappings loaded successfully with FIXED therapeutic areas")
        
    except FileNotFoundError:
        print(f"❌ Error: Mappings file not found at {mappings_path}")
        return None
    
    # 2. Load graph
    graph_path = f"{results_path}complete_graph_{datetime}.pt"
    try:
        graph = torch.load(graph_path, map_location=device)
        print(f"✅ Graph loaded: {graph.num_nodes} nodes, {graph.num_edges} edges")
    except FileNotFoundError:
        print(f"❌ Error: Graph file not found at {graph_path}")
        return None
    
    # 3. Load FP predictions
    fp_filename = f"{results_path}TransformerModel_Test_FP_links_{datetime}.pt"
    try:
        transformer_fps = torch.load(fp_filename, map_location=device)
        print(f"✅ FP predictions loaded: {len(transformer_fps)} predictions")
    except FileNotFoundError:
        # Try CSV version
        csv_filename = f"{results_path}TransformerModel_Test_FP_links_{datetime}.csv"
        try:
            df = pd.read_csv(csv_filename)
            transformer_fps = [[row['Drug'], row['Disease'], row['Probability']] 
                              for _, row in df.iterrows()]
            print(f"✅ FP predictions loaded from CSV: {len(transformer_fps)} predictions")
        except FileNotFoundError:
            print(f"❌ Error: FP predictions not found at {fp_filename} or {csv_filename}")
            return None
    
    print(f"✅ All essential components loaded successfully!")
    
    return {
        'graph': graph,
        'transformer_fps': transformer_fps,
        'all_mappings': mappings
    }

def create_web_fp_visualizer(graph, transformer_fps, results_path, datetime, 
                           sample_size, top_diseases, top_drugs, max_path_length, max_paths_per_pair):
    """
    Create an interactive web-based visualizer for FP predictions
    
    Args:
        graph: PyTorch Geometric graph
        transformer_fps: List of FP predictions
        results_path: Path to save the HTML file
        datetime: Timestamp for filename
        sample_size: Number of FP predictions to analyze
        top_diseases: Number of top diseases to include
        top_drugs: Number of top drugs to include
        max_path_length: Maximum path length to find
        max_paths_per_pair: Maximum number of paths to find per FP pair
    """
    
    print("🌐 CREATING INTERACTIVE WEB VISUALIZER")
    print("="*50)
    print(f"⚙️ Settings: {top_drugs} drugs × {top_diseases} diseases")
    print(f"📊 Sample size: {sample_size} FP predictions")
    print(f"🛤️ Max path length: {max_path_length}")
    print(f"🔍 Max paths per pair: {max_paths_per_pair}")
    
    # Convert graph and get data
    G = to_networkx(graph, to_undirected=True)
    
    # Get optimized FP pairs
    top_fp_pairs = get_web_optimized_fp_pairs(transformer_fps, sample_size, top_diseases, top_drugs)
    
    # Find paths (optimized for web with filtering)
    all_paths_data = find_web_optimized_paths_filtered(G, top_fp_pairs, max_path_length, max_paths_per_pair)
    
    # Create network data for web
    network_data = create_web_network_data(G, all_paths_data, top_fp_pairs)
    
    # Generate the HTML file
    html_file = create_interactive_html(network_data, all_paths_data, results_path, datetime)
    
    print(f"\n🎉 SUCCESS! Interactive visualizer created:")
    print(f"📁 File: {html_file}")
    print(f"🌐 To view: Double-click the HTML file to open in your browser")
    print(f"📊 Network: {len(network_data['nodes'])} nodes, {len(network_data['edges'])} edges")
    print(f"🛤️ Paths: {sum(data['paths_found'] for data in all_paths_data.values())} total connections")
    print(f"🎯 Enhanced with Focus Mode and up to {max_paths_per_pair} paths per FP pair")
    print(f"🔴 Selected FP pairs will appear in RED with blinking effect")
    print(f"🚫 Direct drug-to-drug-type paths filtered out")
    
    return html_file, network_data

def get_web_optimized_fp_pairs(transformer_fps, sample_size, top_diseases, top_drugs):
    """Get FP pairs optimized for web visualization"""
    
    print("🔍 Selecting FP pairs for web visualization...")
    
    # Limit sample for performance
    limited_fps = transformer_fps[:sample_size]
    
    # Group by confidence and get diverse pairs
    drug_scores = defaultdict(float)
    disease_scores = defaultdict(float)
    
    # Calculate max scores for each entity
    for drug_name, disease_name, confidence in limited_fps:
        drug_scores[drug_name] = max(drug_scores[drug_name], confidence)
        disease_scores[disease_name] = max(disease_scores[disease_name], confidence)
    
    # Get top entities
    top_drug_names = sorted(drug_scores.keys(), key=lambda x: drug_scores[x], reverse=True)[:top_drugs]
    top_disease_names = sorted(disease_scores.keys(), key=lambda x: disease_scores[x], reverse=True)[:top_diseases]
    
    # Select pairs involving top entities
    selected_pairs = []
    for drug_name, disease_name, confidence in limited_fps:
        if drug_name in top_drug_names and disease_name in top_disease_names:
            if drug_name in approved_drugs_list_name and disease_name in disease_list_name:
                drug_idx = approved_drugs_list_name.index(drug_name)
                disease_list_pos = disease_list_name.index(disease_name)
                disease_id = disease_list[disease_list_pos]
                disease_idx = disease_key_mapping[disease_id]
                
                selected_pairs.append({
                    'drug_name': drug_name,
                    'disease_name': disease_name,
                    'drug_idx': drug_idx,
                    'disease_idx': disease_idx,
                    'confidence': confidence
                })
    
    print(f"   ✓ Selected {len(selected_pairs)} drug-disease pairs")
    print(f"   ✓ Drugs: {', '.join(top_drug_names[:3])}{'...' if len(top_drug_names) > 3 else ''}")
    print(f"   ✓ Diseases: {', '.join(top_disease_names[:3])}{'...' if len(top_disease_names) > 3 else ''}")
    
    return selected_pairs

def find_web_optimized_paths_filtered(G, top_fp_pairs, max_path_length, max_paths_per_pair=50):
    """
    Find paths optimized for web visualization with filtering of ONLY direct drug-to-drug-type connections from start
    """
    
    print("🛤️ Finding connection paths with minimal drug-type filtering...")
    print(f"   🔍 Settings: Up to {max_paths_per_pair} paths per pair, max {max_path_length} steps")
    print(f"   🚫 Filtering out ONLY direct starting drug-to-drug-type connections")
    
    all_paths_data = {}
    
    for i, pair in enumerate(top_fp_pairs):
        drug_idx = pair['drug_idx']
        disease_idx = pair['disease_idx']
        pair_key = f"{pair['drug_name']} -> {pair['disease_name']}"
        
        print(f"   ({i+1}/{len(top_fp_pairs)}) {pair_key}")
        
        try:
            # Find paths with much larger initial search to ensure we get enough valid paths
            path_generator = nx.all_simple_paths(G, drug_idx, disease_idx, cutoff=max_path_length)
            paths_found = []
            paths_examined = 0
            
            # Dynamic examination limit based on path length
            if max_path_length <= 3:
                max_examination_limit = max_paths_per_pair * 10
            elif max_path_length == 4:
                max_examination_limit = max_paths_per_pair * 20
            else:  # max_path_length >= 5
                max_examination_limit = max_paths_per_pair * 50  # Much more for longer paths
            
            for path in path_generator:
                paths_examined += 1
                
                # Only filter out direct drug-to-drug-type from the starting drug
                if is_valid_path_no_direct_drug_type(path, drug_idx):
                    paths_found.append(path)
                    
                    # Stop when we have enough valid paths
                    if len(paths_found) >= max_paths_per_pair:
                        break
                
                # Prevent infinite search - examine reasonable number of paths
                if paths_examined >= max_examination_limit:
                    break
            
            if paths_found:
                paths_found.sort(key=len)  # Shortest first
                
                path_details = []
                for path in paths_found:
                    path_info = []
                    for node_idx in path:
                        node_name = idx_to_name.get(node_idx, f"Node_{node_idx}")
                        node_type = idx_to_type.get(node_idx, "Unknown")
                        path_info.append({
                            'idx': node_idx,
                            'name': node_name,
                            'type': node_type
                        })
                    
                    path_details.append({
                        'length': len(path) - 1,
                        'nodes': path_info,
                        'node_indices': path
                    })
                
                all_paths_data[pair_key] = {
                    'pair': pair,
                    'paths_found': len(paths_found),
                    'paths': path_details,
                    'shortest_length': min(p['length'] for p in path_details)
                }
                
                print(f"      ✓ Found {len(paths_found)} valid paths (examined {paths_examined}, shortest: {min(p['length'] for p in path_details)} steps)")
            else:
                all_paths_data[pair_key] = {
                    'pair': pair,
                    'paths_found': 0,
                    'paths': [],
                    'shortest_length': 0
                }
                print(f"      ✗ No valid paths found after examining {paths_examined} paths")
        
        except Exception as e:
            print(f"      ✗ Error: {e}")
            all_paths_data[pair_key] = {
                'pair': pair,
                'paths_found': 0,
                'paths': [],
                'error': str(e)
            }
    
    return all_paths_data

def is_valid_path_no_direct_drug_type(path, drug_idx):
    """
    Check if a path is valid (doesn't contain DIRECT drug-to-drug-type connections)
    Only filters out the immediate direct connection from the starting drug to drug type
    
    Args:
        path: List of node indices in the path
        drug_idx: Index of the starting drug node
    
    Returns:
        bool: True if path is valid (no DIRECT drug-to-drug-type connection from start)
    """
    if len(path) < 2:
        return True
    
    # ONLY check if the path starts with our specific drug -> drug_type (first two nodes only)
    if len(path) >= 2:
        first_node = path[0]
        second_node = path[1]
        
        # If first node is our specific drug and second node is a drug type, filter it out
        if first_node == drug_idx:
            second_node_type = idx_to_type.get(second_node, "Unknown")
            if second_node_type == "DrugType":
                return False  # Filter out this direct connection
    
    # Allow all other paths (including drug-type connections elsewhere in the path)
    return True  # Path is valid

def create_web_network_data(G, all_paths_data, top_fp_pairs):
    """Create network data optimized for web visualization"""
    
    print("🔗 Preparing network data for web...")
    
    # Collect all nodes from paths
    all_nodes = set()
    all_edges = set()
    
    for data in all_paths_data.values():
        for path in data['paths']:
            nodes = path['node_indices']
            all_nodes.update(nodes)
            
            # Add edges from path
            for i in range(len(nodes) - 1):
                edge = tuple(sorted([nodes[i], nodes[i+1]]))
                all_edges.add(edge)
    
    # Create nodes data
    nodes_data = []
    node_type_colors = {
        'Drug': '#1f77b4',      # Blue
        'Disease': '#ff7f0e',   # Orange  
        'Gene': '#2ca02c',      # Green
        'Pathway': '#d62728',   # Red
        'DrugType': '#9467bd',  # Purple
        'TherapeuticArea': '#8c564b'  # Brown
    }
    
    fp_drug_indices = {pair['drug_idx'] for pair in top_fp_pairs}
    fp_disease_indices = {pair['disease_idx'] for pair in top_fp_pairs}
    
    for node_idx in all_nodes:
        node_name = idx_to_name.get(node_idx, f"Node_{node_idx}")
        node_type = idx_to_type.get(node_idx, "Unknown")
        
        is_fp_drug = node_idx in fp_drug_indices
        is_fp_disease = node_idx in fp_disease_indices
        is_fp_target = is_fp_drug or is_fp_disease
        
        # Get degree for sizing
        degree = G.degree(node_idx)
        
        nodes_data.append({
            'id': node_idx,
            'name': node_name,
            'type': node_type,
            'color': node_type_colors.get(node_type, '#999999'),
            'size': 15,  # Standardized size for all nodes
            'degree': degree,
            'is_fp_target': is_fp_target,
            'is_fp_drug': is_fp_drug,
            'is_fp_disease': is_fp_disease
        })
    
    # Create edges data
    edges_data = []
    for edge in all_edges:
        source, target = edge
        edges_data.append({
            'source': source,
            'target': target
        })
    
    print(f"   ✓ Prepared {len(nodes_data)} nodes and {len(edges_data)} edges")
    
    return {
        'nodes': nodes_data,
        'edges': edges_data
    }

def create_interactive_html(network_data, all_paths_data, results_path, datetime):
    """Create the interactive HTML file with enhanced styling for selected pairs"""
    
    print("🎨 Generating interactive HTML with red selection and blinking effects...")
    
    # Prepare FP pairs data for the interface
    fp_pairs_info = []
    for pair_key, data in all_paths_data.items():
        fp_pairs_info.append({
            'pair_name': pair_key,
            'drug_name': data['pair']['drug_name'],
            'disease_name': data['pair']['disease_name'],
            'confidence': float(data['pair']['confidence']),
            'paths_found': int(data['paths_found']),
            'shortest_length': int(data.get('shortest_length', 0)),
            'drug_idx': int(data['pair']['drug_idx']),
            'disease_idx': int(data['pair']['disease_idx'])
        })
    
    # Sort by confidence
    fp_pairs_info.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Get all unique diseases from FP pairs for the dropdown
    unique_diseases = list(set([pair['disease_name'] for pair in fp_pairs_info]))
    unique_diseases.sort()

    unique_drugs = list(set([pair['drug_name'] for pair in fp_pairs_info]))
    unique_drugs.sort()
    
    # Convert network data to ensure JSON serialization
    json_safe_network_data = {
        'nodes': [
            {
                'id': int(node['id']),
                'name': str(node['name']),
                'type': str(node['type']),
                'color': str(node['color']),
                'size': float(node['size']),
                'degree': int(node['degree']),
                'is_fp_target': bool(node['is_fp_target']),
                'is_fp_drug': bool(node['is_fp_drug']),
                'is_fp_disease': bool(node['is_fp_disease'])
            }
            for node in network_data['nodes']
        ],
        'edges': [
            {
                'source': int(edge['source']),
                'target': int(edge['target'])
            }
            for edge in network_data['edges']
        ]
    }
    
    # Convert paths data to ensure JSON serialization
    json_safe_paths_data = {}
    for pair_key, data in all_paths_data.items():
        json_safe_paths_data[pair_key] = {
            'pair': {
                'drug_name': str(data['pair']['drug_name']),
                'disease_name': str(data['pair']['disease_name']),
                'confidence': float(data['pair']['confidence']),
                'drug_idx': int(data['pair']['drug_idx']),
                'disease_idx': int(data['pair']['disease_idx'])
            },
            'paths_found': int(data['paths_found']),
            'shortest_length': int(data.get('shortest_length', 0)),
            'paths': [
                {
                    'length': int(path['length']),
                    'node_indices': [int(idx) for idx in path['node_indices']]
                }
                for path in data['paths']
            ]
        }
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FP Drug-Disease Network Explorer - Enhanced with Red Selection</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #fff;
            height: 100vh;
            overflow: hidden;
        }}

        .container {{
            display: flex;
            height: 100vh;
        }}

        .sidebar {{
            width: 380px;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.2);
            overflow-y: auto;
            box-shadow: 2px 0 20px rgba(0, 0, 0, 0.1);
        }}

        .main-view {{
            flex: 1;
            position: relative;
        }}

        .header {{
            padding: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }}

        .header h1 {{
            margin: 0;
            color: #333;
            font-size: 24px;
            font-weight: 600;
        }}

        .header p {{
            margin: 5px 0 0 0;
            color: #666;
            font-size: 14px;
        }}

        .controls {{
            padding: 20px;
        }}

        .control-group {{
            margin-bottom: 20px;
        }}

        .control-group h3 {{
            margin: 0 0 10px 0;
            color: #333;
            font-size: 16px;
        }}

        .search-box {{
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }}

        .search-box:focus {{
            outline: none;
            border-color: #667eea;
        }}

        .disease-dropdown {{
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
            background: white;
        }}

        .disease-dropdown:focus {{
            outline: none;
            border-color: #667eea;
        }}

        .filter-buttons {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }}

        .filter-btn {{
            padding: 6px 12px;
            border: 1px solid #ddd;
            background: white;
            border-radius: 20px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.3s;
        }}

        .filter-btn:hover {{
            background: #f0f0f0;
        }}

        .filter-btn.active {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}

        .reset-btn {{
            padding: 10px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
            font-size: 14px;
            transition: background 0.3s;
            margin-bottom: 8px;
        }}

        .reset-btn:hover {{
            background: #5a6fd8;
        }}

        .focus-btn {{
            padding: 10px;
            background: #6c757d;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
            font-size: 14px;
            transition: background 0.3s;
            margin-bottom: 8px;
        }}

        .focus-btn:hover {{
            background: #5a6268;
        }}

        .focus-btn.active {{
            background: #000000;
            border: 2px solid #333;
        }}

        .focus-btn.active:hover {{
            background: #333;
        }}

        .filter-status {{
            background: rgba(103, 126, 234, 0.1);
            border: 1px solid #667eea;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 15px;
            font-size: 14px;
            color: #333;
        }}

        .filter-status.active {{
            background: rgba(103, 126, 234, 0.2);
            border-color: #5a6fd8;
        }}

        .clear-filter-btn {{
            background: #dc3545;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 12px;
            cursor: pointer;
            margin-left: 8px;
        }}

        .clear-filter-btn:hover {{
            background: #c82333;
        }}

        .fp-pairs {{
            max-height: 400px;
            overflow-y: auto;
        }}

        .fp-pair {{
            padding: 12px;
            border: 1px solid #eee;
            border-radius: 8px;
            margin-bottom: 8px;
            cursor: pointer;
            transition: all 0.3s;
        }}

        .fp-pair:hover {{
            background: #f8f9ff;
            border-color: #667eea;
        }}

        .fp-pair.selected {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}

        .fp-pair.focused {{
            background: #000000;
            color: white;
            border-color: #000000;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
        }}

        /* NEW: Red selection styling for FP pairs */
        .fp-pair.red-selected {{
            background: #dc3545 !important;
            color: white !important;
            border-color: #dc3545 !important;
            box-shadow: 0 0 15px rgba(220, 53, 69, 0.6);
            animation: redPulse 2s infinite;
        }}

        /* Blinking animation for selected FP pair */
        @keyframes redPulse {{
            0% {{ 
                box-shadow: 0 0 15px rgba(220, 53, 69, 0.6);
                transform: scale(1);
            }}
            50% {{ 
                box-shadow: 0 0 25px rgba(220, 53, 69, 0.9);
                transform: scale(1.02);
            }}
            100% {{ 
                box-shadow: 0 0 15px rgba(220, 53, 69, 0.6);
                transform: scale(1);
            }}
        }}

        .fp-pair-title {{
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 4px;
        }}

        .fp-pair-info {{
            font-size: 12px;
            opacity: 0.8;
        }}

        .network-container {{
            width: 100%;
            height: 100%;
        }}

        .network-svg {{
            width: 100%;
            height: 100%;
        }}

        .node {{
            cursor: pointer;
            stroke: white;
            stroke-width: 2px;
            transition: opacity 0.3s ease;
        }}

        .node.fp-target {{
            stroke: #000000;
            stroke-width: 4px;
        }}

        /* NEW: Red styling for selected FP target nodes */
        .node.red-selected {{
            fill: #dc3545 !important;
            stroke: #ffffff !important;
            stroke-width: 6px !important;
            filter: drop-shadow(0 0 10px rgba(220, 53, 69, 0.8));
            animation: nodeRedPulse 2s infinite;
        }}

        /* Blinking animation for selected FP target nodes */
        @keyframes nodeRedPulse {{
            0% {{ 
                filter: drop-shadow(0 0 10px rgba(220, 53, 69, 0.8));
                transform: scale(1);
            }}
            50% {{ 
                filter: drop-shadow(0 0 20px rgba(220, 53, 69, 1.0));
                transform: scale(1.1);
            }}
            100% {{ 
                filter: drop-shadow(0 0 10px rgba(220, 53, 69, 0.8));
                transform: scale(1);
            }}
        }}

        .node.dimmed {{
            opacity: 0.05;
        }}

        .node.focused {{
            opacity: 1;
            stroke-width: 4px;
        }}

        .node.clickable-disease {{
            cursor: pointer;
            stroke: #ff7f0e;
            stroke-width: 3px;
        }}

        .node.clickable-disease:hover {{
            stroke: #ff4500;
            stroke-width: 5px;
        }}

        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
            stroke-width: 1.5px;
            transition: opacity 0.3s ease;
        }}

        .link.dimmed {{
            opacity: 0.02;
        }}

        .link.highlighted {{
            stroke: #000000;
            stroke-width: 4px;
            stroke-opacity: 1;
            opacity: 1;
        }}

        /* NEW: Red styling for selected FP pair paths */
        .link.red-highlighted {{
            stroke: #dc3545 !important;
            stroke-width: 6px !important;
            stroke-opacity: 1 !important;
            opacity: 1 !important;
            filter: drop-shadow(0 0 5px rgba(220, 53, 69, 0.6));
            animation: linkRedPulse 2s infinite;
        }}

        /* Blinking animation for selected FP pair paths */
        @keyframes linkRedPulse {{
            0% {{ 
                stroke-width: 6px;
                filter: drop-shadow(0 0 5px rgba(220, 53, 69, 0.6));
            }}
            50% {{ 
                stroke-width: 8px;
                filter: drop-shadow(0 0 10px rgba(220, 53, 69, 0.9));
            }}
            100% {{ 
                stroke-width: 6px;
                filter: drop-shadow(0 0 5px rgba(220, 53, 69, 0.6));
            }}
        }}

        .node-label {{
            font-family: 'Segoe UI', sans-serif;
            font-size: 12px;
            pointer-events: none;
            text-anchor: middle;
            fill: #333;
            transition: opacity 0.3s ease;
        }}

        .node-label.dimmed {{
            opacity: 0.02;
        }}

        .node-label.focused {{
            opacity: 1;
            font-weight: bold;
        }}

        /* NEW: Red styling for selected FP target labels */
        .node-label.red-selected {{
            fill: #dc3545 !important;
            font-weight: bold !important;
            font-size: 14px !important;
            filter: drop-shadow(0 0 3px rgba(220, 53, 69, 0.8));
        }}

        .info-panel {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            max-width: 300px;
            display: none;
        }}

        .info-panel h3 {{
            margin: 0 0 10px 0;
            color: #333;
        }}

        .info-panel p {{
            margin: 5px 0;
            color: #666;
            font-size: 14px;
        }}

        .legend {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}

        .legend h4 {{
            margin: 0 0 10px 0;
            color: #333;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 5px;
            font-size: 12px;
        }}

        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            margin-right: 8px;
        }}

        .stats {{
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            margin: 20px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}

        .stats h4 {{
            margin: 0 0 10px 0;
            color: #333;
        }}

        .stat-item {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-size: 14px;
            color: #666;
        }}

        .focus-mode-indicator {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 10px 15px;
            border-radius: 8px;
            font-weight: bold;
            display: none;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }}

        .smart-filter-indicator {{
            position: absolute;
            top: 70px;
            left: 20px;
            background: rgba(255, 127, 14, 0.9);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-weight: bold;
            display: none;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }}

        /* NEW: Red selection indicator */
        .red-selection-indicator {{
            position: absolute;
            top: 120px;
            left: 20px;
            background: rgba(220, 53, 69, 0.9);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-weight: bold;
            display: none;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            animation: redPulse 2s infinite;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Sidebar -->
        <div class="sidebar">
            <div class="header">
                <h1>🧠 FP Network Explorer</h1>
                <p>Interactive Drug-Disease Connection Analysis with Red Selection & Filtered Paths</p>
            </div>

            <div class="controls">
                <div class="control-group">
                    <h3>🎯Option 1: Smart Disease Filter</h3>
                    <select class="disease-dropdown" id="diseaseDropdown">
                        <option value="">Select a disease...</option>
                        {chr(10).join([f'<option value="{disease}">{disease}</option>' for disease in unique_diseases])}
                    </select>
                    <p style="font-size: 12px; color: #666; margin-top: 8px;">
                        💡 <strong>Tip:</strong> Click on disease nodes to filter FP pairs!
                    </p>
                </div>

                <div class="control-group">
                    <h3>🎯Option 2: Smart Drug Filter</h3>
                    <select class="disease-dropdown" id="drugDropdown">
                        <option value="">Select a drug...</option>
                        {chr(10).join([f'<option value="{drug}">{drug}</option>' for drug in unique_drugs])}
                    </select>
                    <p style="font-size: 12px; color: #666; margin-top: 8px;">
                        💡 <strong>Tip:</strong> Click on drug nodes to filter FP pairs!
                    </p>
                </div>

                <!-- Filter Status Display -->
                <div class="filter-status" id="filterStatus" style="display: none;">
                    <div id="filterStatusText">No filter applied</div>
                </div>

                <div class="control-group">
                    <button class="focus-btn" id="focusToggle" onclick="toggleFocusMode()">🎯 Focus Mode: OFF</button>
                </div>

                <div class="control-group">
                    <h3>🎯 FP Drug-Disease Pairs <span id="pairCount">({len(fp_pairs_info)})</span></h3>
                    <p style="font-size: 12px; color: #333; margin-bottom: 10px;">
                        <strong>Normal Mode:</strong> Selected drug--disease pair + paths in RED with blinking<br>
                        <strong>Focus Mode:</strong> Selected drug--disease pair in RED with blinking, paths in BLACK<br>
                        🚫 <strong> Direct drug-to-drug-type links filtered out</strong>
                    </p>
                    <div class="fp-pairs" id="fpPairsList">
                        <!-- FP pairs will be loaded here -->
                    </div>
                </div>
            </div>

            <div class="stats">
                <h4>📊 Network Statistics</h4>
                <div class="stat-item">
                    <span>Nodes:</span>
                    <span>{len(json_safe_network_data['nodes'])}</span>
                </div>
                <div class="stat-item">
                    <span>Edges:</span>
                    <span>{len(json_safe_network_data['edges'])}</span>
                </div>
                <div class="stat-item">
                    <span>FP Pairs:</span>
                    <span>{len(fp_pairs_info)}</span>
                </div>
                <div class="stat-item">
                    <span>Total Paths:</span>
                    <span>{sum(int(data['paths_found']) for data in all_paths_data.values())}</span>
                </div>
                <div class="stat-item">
                    <span>🧬 Gene Display:</span>
                    <span>✅ Using Symbols</span>
                </div>
                <div class="stat-item">
                    <span>🚫 Path Filtering:</span>
                    <span>✅ Minimal (Drug-Type)</span>
                </div>
            </div>
        </div>

        <!-- Main network view -->
        <div class="main-view">
            <div class="focus-mode-indicator" id="focusModeIndicator">
                🎯 FOCUS MODE ACTIVE - Only selected FP pair visible
            </div>

            <div class="smart-filter-indicator" id="smartFilterIndicator">
                🎯 SMART FILTER: Disease selected
            </div>

            <!-- NEW: Red selection indicator -->
            <div class="red-selection-indicator" id="redSelectionIndicator">
                🔴 RED SELECTION ACTIVE
            </div>

            <div class="network-container">
                <svg class="network-svg" id="networkSvg"></svg>
            </div>

            <!-- Info panel (appears on node click) -->
            <div class="info-panel" id="infoPanel">
                <h3 id="infoPanelTitle">Node Information</h3>
                <p id="infoPanelContent">Click a node to see details</p>
            </div>

            <!-- Legend -->
            <div class="legend">
                <h4>🎨 Node Types</h4>
                <div class="legend-item">
                    <div class="legend-color" style="background: #1f77b4;"></div>
                    <span>Drugs</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ff7f0e;"></div>
                    <span>Diseases</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #2ca02c;"></div>
                    <span>Genes</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #d62728;"></div>
                    <span>Pathways</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #9467bd;"></div>
                    <span>Drug Types</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #8c564b;"></div>
                    <span>Therapeutic Areas</span>  
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #dc3545; border: 2px solid #ffffff;"></div>
                    <span>FP Targets</span>
                </div>
          
            </div>
        </div>
    </div>

    <script>
        // Network data
        const networkData = {json.dumps(json_safe_network_data, indent=8)};
        const fpPairsData = {json.dumps(fp_pairs_info, indent=8)};
        const pathsData = {json.dumps(json_safe_paths_data, indent=8)};
        const allFpPairsData = [...fpPairsData]; // Keep original for reset

        // Focus mode and filter state
        let focusModeEnabled = false;
        let currentFocusedPair = null;
        let currentDiseaseFilter = null;
        let filteredFpPairs = [...fpPairsData];
        
        // NEW: Red selection state
        let currentRedSelectedPair = null;

        /* D3.js visualization with boundary constraints */
        const svg = d3.select("#networkSvg");
        let width = window.innerWidth - 380;
        let height = window.innerHeight;

        svg.attr("width", width).attr("height", height);

        /* Create containers for different elements */
        const linkContainer = svg.append("g").attr("class", "links");
        const nodeContainer = svg.append("g").attr("class", "nodes");
        const labelContainer = svg.append("g").attr("class", "labels");

        /* Define margins to keep nodes away from edges */
        const margin = 50;

        // Create force simulation with boundary constraints
        const simulation = d3.forceSimulation(networkData.nodes)
            .force("link", d3.forceLink(networkData.edges).id(d => d.id).distance(80))
            .force("charge", d3.forceManyBody().strength(-200))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(d => d.size + 5))
            .force("boundary", function() {{
                for (let node of networkData.nodes) {{
                    node.x = Math.max(margin + node.size, Math.min(width - margin - node.size, node.x || width/2));
                    node.y = Math.max(margin + node.size, Math.min(height - margin - node.size, node.y || height/2));
                }}
            }});

        /* Create links */
        const links = linkContainer.selectAll("line")
            .data(networkData.edges)
            .enter().append("line")
            .attr("class", "link");

        /* Create nodes with smart disease filtering */
        const nodes = nodeContainer.selectAll("circle")
            .data(networkData.nodes)
            .enter().append("circle")
            .attr("class", d => {{
                let classes = "node";
                if (d.is_fp_target) classes += " fp-target";
                if (d.type === "Disease" && d.is_fp_disease) classes += " clickable-disease";
                return classes;
            }})
            .attr("r", d => d.size)
            .attr("fill", d => d.color)
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended))
            .on("click", handleNodeClick);

        /* Create labels */
        const labels = labelContainer.selectAll("text")
            .data(networkData.nodes)
            .enter().append("text")
            .attr("class", "node-label")
            .text(d => d.name.length > 15 ? d.name.substring(0, 15) + "..." : d.name);

        // Enhanced tick handler with boundary constraints
        simulation.on("tick", () => {{
            networkData.nodes.forEach(d => {{
                d.x = Math.max(margin + d.size, Math.min(width - margin - d.size, d.x));
                d.y = Math.max(margin + d.size, Math.min(height - margin - d.size, d.y));
            }});

            links
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            nodes
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);

            labels
                .attr("x", d => d.x)
                .attr("y", d => d.y + 5);
        }});

        /* Initialize node positions */
        networkData.nodes.forEach(d => {{
            if (!d.x) d.x = Math.random() * (width + 60) - 30;
            if (!d.y) d.y = Math.random() * (height + 60) - 30;
        }});

        /* Enhanced node click handler with smart disease filtering */
        function handleNodeClick(event, d) {{
            // Prevent event bubbling
            event.stopPropagation();
            
            // If it's a clickable disease node, apply disease filter
            if (d.type === "Disease" && d.is_fp_disease) {{
                applyDiseaseFilter(d.name);
                showSmartFilterIndicator(d.name);
            }}
            // If it's a drug node, apply drug filter
            if (d.type === "Drug" && d.is_fp_drug) {{
                applyDrugFilter(d.name);
                showSmartFilterIndicator(d.name);
            }}
            
            // Always show node info
            showNodeInfo(event, d);
        }}

        /* Apply disease filter to FP pairs */
        function applyDiseaseFilter(diseaseName) {{
            currentDiseaseFilter = diseaseName;
            
            // Filter FP pairs to only those containing this disease
            filteredFpPairs = allFpPairsData.filter(pair => 
                pair.disease_name === diseaseName
            );
            
            // Update dropdown
            const dropdown = d3.select("#diseaseDropdown");
            dropdown.property("value", diseaseName);
            
            // Update FP pairs display
            updateFPPairsList();
            
            // Update filter status
            updateFilterStatus(`Filtered by disease: ${{diseaseName}}`, true);
            
            console.log(`🎯 Smart filter applied: ${{diseaseName}} (${{filteredFpPairs.length}} pairs)`);
        }}

        /* Apply drug filter to FP pairs */
        function applyDrugFilter(drugName) {{
            const drugIdx = networkData.nodes.findIndex(n => n.name === drugName && n.type === "Drug");
            if (drugIdx === -1) return; // Drug not found
            
            // Filter FP pairs to only those containing this drug
            filteredFpPairs = allFpPairsData.filter(pair => 
                pair.drug_name === drugName
            );
            
            // Update dropdown
            const dropdown = d3.select("#drugDropdown");
            dropdown.property("value", drugName);
            
            // Update FP pairs display
            updateFPPairsList();
            
            // Update filter status
            updateFilterStatus(`Filtered by drug: ${{drugName}}`, true);
            
            console.log(`🎯 Smart filter applied: ${{drugName}} (${{filteredFpPairs.length}} pairs)`);
        }}

        /* Clear disease filter */
        function clearDiseaseFilter() {{
            currentDiseaseFilter = null;
            filteredFpPairs = [...allFpPairsData];
            
            // Reset dropdown
            const dropdown = d3.select("#diseaseDropdown");
            dropdown.property("value", "");
            
            // Update displays
            updateFPPairsList();
            updateFilterStatus("No filter applied", false);
            hideSmartFilterIndicator();
            
            console.log("✅ Disease filter cleared");
        }}

        /* Clear drug filter */
        function clearDrugFilter() {{
            currentDiseaseFilter = null;
            filteredFpPairs = [...allFpPairsData];  
            // Reset dropdown
            const dropdown = d3.select("#drugDropdown");
            dropdown.property("value", "");
            // Update displays
            updateFPPairsList();
            hideSmartFilterIndicator();
            updateFilterStatus("No filter applied", false);
            console.log("✅ Drug filter cleared");
        }}

        /* Show smart filter indicator */
        function showSmartFilterIndicator(diseaseName) {{
            const indicator = d3.select("#smartFilterIndicator");
            indicator.text(`🎯 SMART FILTER: ${{diseaseName}}`)
                    .style("display", "block");
        }}

        /* Hide smart filter indicator */
        function hideSmartFilterIndicator() {{
            d3.select("#smartFilterIndicator").style("display", "none");
        }}

        /* NEW: Show red selection indicator */
        function showRedSelectionIndicator(pairName) {{
            const indicator = d3.select("#redSelectionIndicator");
            indicator.text(`🔴 SELECTED: ${{pairName}}`)
                    .style("display", "block");
        }}

        /* NEW: Hide red selection indicator */
        function hideRedSelectionIndicator() {{
            d3.select("#redSelectionIndicator").style("display", "none");
        }}

        /* Update filter status display */
        function updateFilterStatus(text, isActive) {{
            const status = d3.select("#filterStatus");
            const statusText = d3.select("#filterStatusText");
            
            statusText.html(text + (isActive ? ' <button class="clear-filter-btn" onclick="clearDiseaseFilter()">Clear</button>' : ''));
            status.classed("active", isActive)
                  .style("display", isActive ? "block" : "none");
        }}

        /* Update FP pairs list with current filter */
        function updateFPPairsList() {{
            const container = d3.select("#fpPairsList");
            const pairCount = d3.select("#pairCount");
            
            // Update count
            pairCount.text(`(${{filteredFpPairs.length}})`);
            
            // Clear existing pairs
            container.selectAll(".fp-pair").remove();
            
            // Add filtered pairs
            const pairs = container.selectAll(".fp-pair")
                .data(filteredFpPairs)
                .enter().append("div")
                .attr("class", "fp-pair")
                .on("click", highlightFPPair);

            pairs.append("div")
                .attr("class", "fp-pair-title")
                .text(d => d.pair_name);

            pairs.append("div")
                .attr("class", "fp-pair-info")
                .text(d => `Confidence: ${{d.confidence.toFixed(4)}} | ${{d.paths_found}} connection paths | Shortest: ${{d.shortest_length}} steps`);
        }}

        /* Handle dropdown selection */
        d3.select("#diseaseDropdown").on("change", function() {{
            const selectedDisease = this.value;
            
            if (selectedDisease) {{
                applyDiseaseFilter(selectedDisease);
                showSmartFilterIndicator(selectedDisease);
            }} else {{
                clearDiseaseFilter();
            }}
        }});
        
        d3.select("#drugDropdown").on("change", function() {{
            const selectedDrug = this.value;
            if (selectedDrug) {{
                applyDrugFilter(selectedDrug);
                showSmartFilterIndicator(selectedDrug);
            }} else {{
                clearDrugFilter();
            }}
        }});

        /* Toggle Focus Mode */
        function toggleFocusMode() {{
            focusModeEnabled = !focusModeEnabled;
            const toggleBtn = d3.select("#focusToggle");
            const indicator = d3.select("#focusModeIndicator");
            
            if (focusModeEnabled) {{
                toggleBtn.text("🎯 Focus Mode: ON")
                    .classed("active", true);
                indicator.style("display", "block");
                
                // If there's a current selection, apply focus immediately
                if (currentFocusedPair) {{
                    applyFocusMode(currentFocusedPair);
                }}
            }} else {{
                toggleBtn.text("🎯 Focus Mode: OFF")
                    .classed("active", false);
                indicator.style("display", "none");
                
                // Clear focus mode
                clearFocusMode();
            }}
            
            console.log(`Focus mode: ${{focusModeEnabled ? 'ON' : 'OFF'}}`);
        }}

        /* Apply Focus Mode with RED nodes but BLACK edges */
        function applyFocusMode(pairData) {{
            if (!focusModeEnabled) return;
            
            const pairKey = pairData.pair_name;
            const pairPaths = pathsData[pairKey];
            
            if (!pairPaths || pairPaths.paths.length === 0) {{
                console.log("No paths found for this pair");
                return;
            }}
            
            // Get all nodes that are part of ANY path for this pair
            const pathNodes = new Set();
            const pathEdges = new Set();
            
            pairPaths.paths.forEach(path => {{
                path.node_indices.forEach(nodeId => pathNodes.add(nodeId));
                
                // Add edges from this path
                for (let i = 0; i < path.node_indices.length - 1; i++) {{
                    const edge = [path.node_indices[i], path.node_indices[i + 1]].sort().join('-');
                    pathEdges.add(edge);
                }}
            }});
            
            console.log(`Focus mode: Highlighting ${{pathNodes.size}} nodes and ${{pathEdges.size}} edges - RED nodes, BLACK edges`);
            
            // Apply focus styling to nodes
            nodes.classed("dimmed", d => !pathNodes.has(d.id))
                 .classed("focused", d => pathNodes.has(d.id))
                 // NEW: Make the specific drug and disease RED in focus mode
                 .classed("red-selected", d => 
                    d.id === pairData.drug_idx || d.id === pairData.disease_idx
                 );
            
            // Apply focus styling to labels
            labels.classed("dimmed", d => !pathNodes.has(d.id))
                  .classed("focused", d => pathNodes.has(d.id))
                  // NEW: Make the specific drug and disease labels RED in focus mode
                  .classed("red-selected", d => 
                    d.id === pairData.drug_idx || d.id === pairData.disease_idx
                  );
            
            // Apply BLACK highlighting to focus mode links (NO red edges)
            links.classed("dimmed", function(d) {{
                    const edgeKey = [d.source.id, d.target.id].sort().join('-');
                    return !pathEdges.has(edgeKey);
                }})
                 .classed("highlighted", function(d) {{
                    const edgeKey = [d.source.id, d.target.id].sort().join('-');
                    return pathEdges.has(edgeKey);
                }})
                 // Make sure focus mode edges are BLACK, not red
                 .classed("red-highlighted", false);
        }}

        /* Clear Focus Mode */
        function clearFocusMode() {{
            // Remove all focus-related classes including red selection
            nodes.classed("dimmed", false)
                 .classed("focused", false)
                 .classed("red-selected", false);  // Clear red nodes from focus mode
            
            labels.classed("dimmed", false)
                  .classed("focused", false)
                  .classed("red-selected", false);  // Clear red labels from focus mode
            
            links.classed("dimmed", false)
                 .classed("highlighted", false);
        }}

        /* NEW: Apply Red Selection Mode */
        function applyRedSelection(pairData) {{
            const pairKey = pairData.pair_name;
            const pairPaths = pathsData[pairKey];
            
            if (!pairPaths || pairPaths.paths.length === 0) {{
                console.log("No paths found for this pair");
                return;
            }}
            
            // Clear any previous red selection
            clearRedSelection();
            
            // Set current red selected pair
            currentRedSelectedPair = pairData;
            
            // Get all nodes that are part of ANY path for this pair
            const pathNodes = new Set();
            const pathEdges = new Set();
            
            pairPaths.paths.forEach(path => {{
                path.node_indices.forEach(nodeId => pathNodes.add(nodeId));
                
                // Add edges from this path
                for (let i = 0; i < path.node_indices.length - 1; i++) {{
                    const edge = [path.node_indices[i], path.node_indices[i + 1]].sort().join('-');
                    pathEdges.add(edge);
                }}
            }});
            
            console.log(`🔴 Red selection: Highlighting ${{pathNodes.size}} nodes and ${{pathEdges.size}} edges in RED`);
            
            // Apply red styling to the specific drug and disease nodes
            nodes.classed("red-selected", d => 
                d.id === pairData.drug_idx || d.id === pairData.disease_idx
            );
            
            // Apply red styling to labels for the specific drug and disease
            labels.classed("red-selected", d => 
                d.id === pairData.drug_idx || d.id === pairData.disease_idx
            );
            
            // Apply red styling to all path links
            links.classed("red-highlighted", function(d) {{
                const edgeKey = [d.source.id, d.target.id].sort().join('-');
                return pathEdges.has(edgeKey);
            }});
            
            // Show red selection indicator
            showRedSelectionIndicator(pairKey);
        }}

        /* NEW: Clear Red Selection Mode */
        function clearRedSelection() {{
            // Remove all red selection classes
            nodes.classed("red-selected", false);
            labels.classed("red-selected", false);
            links.classed("red-highlighted", false);
            
            // Clear current red selected pair
            currentRedSelectedPair = null;
            
            // Hide red selection indicator
            hideRedSelectionIndicator();
        }}

        /* Show node information */
        function showNodeInfo(event, d) {{
            const panel = d3.select("#infoPanel");
            const title = d3.select("#infoPanelTitle");
            const content = d3.select("#infoPanelContent");

            title.text(d.name);
            
            let infoHtml = `
                <p><strong>Type:</strong> ${{d.type}}</p>
                <p><strong>Degree:</strong> ${{d.degree}} connections</p>
                <p><strong>FP Target:</strong> ${{d.is_fp_target ? "Yes" : "No"}}</p>
                ${{d.is_fp_drug ? "<p><strong>Role:</strong> FP Drug</p>" : ""}}
                ${{d.is_fp_disease ? "<p><strong>Role:</strong> FP Disease</p>" : ""}}
            `;
            
            // Add smart filter tip for disease nodes
            if (d.type === "Disease" && d.is_fp_disease) {{
                infoHtml += `<p><strong>💡 Tip:</strong> Click to filter FP pairs by this disease!</p>`;
            }}
            
            content.html(infoHtml);
            panel.style("display", "block");
        }}

        // Enhanced FP pair highlighting with separate focus mode (BLACK) and red selection
        function highlightFPPair(event, d) {{
            // Remove previous selections
            d3.selectAll(".fp-pair").classed("selected", false)
                                   .classed("focused", false)
                                   .classed("red-selected", false);
            
            // Set current selection
            currentFocusedPair = d;
            
            if (focusModeEnabled) {{
                // Focus mode: Use BLACK highlighting and focused styling
                d3.select(this).classed("focused", true);
                
                // Clear any red selection first
                clearRedSelection();
                
                // Apply focus mode with BLACK edges
                applyFocusMode(d);
            }} else {{
                // Normal mode: Use RED selection
                d3.select(this).classed("red-selected", true);
                
                // Clear any focus mode styling first
                clearFocusMode();
                
                // Apply red selection
                applyRedSelection(d);
            }}
        }}

        /* Standard path highlighting (non-focus mode) */
        function highlightPath(nodeIndices) {{
            const pathEdges = [];
            for (let i = 0; i < nodeIndices.length - 1; i++) {{
                pathEdges.push({{
                    source: nodeIndices[i],
                    target: nodeIndices[i + 1]
                }});
            }}

            links.classed("highlighted", function(d) {{
                return pathEdges.some(edge => 
                    (d.source.id === edge.source && d.target.id === edge.target) ||
                    (d.source.id === edge.target && d.target.id === edge.source)
                );
            }});
        }}

        /* Clear highlighting and properly reset state */
        function clearHighlighting() {{
            // Clear visual selections
            d3.selectAll(".fp-pair").classed("selected", false)
                                   .classed("focused", false)
                                   .classed("red-selected", false);
            
            // Reset current focused pair
            currentFocusedPair = null;
            
            // Clear red selection
            clearRedSelection();
            
            if (focusModeEnabled) {{
                // In focus mode, clear the focus but keep mode enabled
                clearFocusMode();
                console.log("✅ Cleared focus selection (focus mode still active)");
            }} else {{
                // In normal mode, just clear standard highlighting
                links.classed("highlighted", false);
                console.log("✅ Cleared standard highlighting and red selection");
            }}
        }}

        /* Enhanced reset that properly resets everything including smart filters */
        function resetView() {{
            // Turn off focus mode first
            if (focusModeEnabled) {{
                focusModeEnabled = false;
                d3.select("#focusToggle").text("🎯 Focus Mode: OFF").classed("active", false);
                d3.select("#focusModeIndicator").style("display", "none");
            }}
            
            // Clear disease filter
            clearDiseaseFilter();
            
            // Clear all selections and reset state
            d3.selectAll(".fp-pair").classed("selected", false)
                                   .classed("focused", false)
                                   .classed("red-selected", false);
            currentFocusedPair = null;
            
            // Clear red selection
            clearRedSelection();
            
            // Clear all visual effects
            clearFocusMode();
            links.classed("highlighted", false);
            
            // Reset zoom and position
            svg.transition().duration(750).call(
                zoom.transform,
                d3.zoomIdentity
            );
            
            // Reset all styling to normal
            nodes.style("opacity", 1.0)
                 .classed("dimmed", false)
                 .classed("focused", false)
                 .classed("red-selected", false);
            labels.style("opacity", 1.0)
                  .classed("dimmed", false)
                  .classed("focused", false)
                  .classed("red-selected", false);
            links.style("opacity", 1.0)
                 .style("stroke", "#999")
                 .style("stroke-width", "1.5px")
                 .classed("highlighted", false)
                 .classed("red-highlighted", false)
                 .classed("dimmed", false);
            
            // Reset filters to "All"
            d3.selectAll(".filter-btn").classed("active", false);
            d3.select(".filter-btn[data-type='all']").classed("active", true);
            
            // Clear search box
            d3.select("#searchBox").property("value", "");
            
            console.log("✅ Complete reset: All filters cleared, FP pairs restored, red selection cleared");
        }}

        /* Enhanced filter functionality that works with smart disease filtering */
        d3.selectAll(".filter-btn").on("click", function() {{
            if (focusModeEnabled) {{
                console.log("Type filters disabled in focus mode");
                return;
            }}
            
            clearHighlighting();
            
            d3.selectAll(".filter-btn").classed("active", false);
            d3.select(this).classed("active", true);

            const filterType = d3.select(this).attr("data-type");
            
            if (filterType === "all") {{
                nodes.style("opacity", 1.0);
                labels.style("opacity", 1.0);
            }} else if (filterType === "Disease") {{
                // Show only disease nodes
                nodes.style("opacity", d => d.type === "Disease" ? 1.0 : 0.1);
                labels.style("opacity", d => d.type === "Disease" ? 1.0 : 0.1);
                
                // Highlight clickable diseases
                nodes.classed("clickable-disease", d => d.type === "Disease" && d.is_fp_disease);
            }} else {{
                nodes.style("opacity", d => d.type === filterType ? 1.0 : 0.1);
                labels.style("opacity", d => d.type === filterType ? 1.0 : 0.1);
            }}
        }});

        /* Enhanced search functionality that works with smart filtering */
        d3.select("#searchBox").on("input", function() {{
            if (focusModeEnabled) {{
                console.log("Search disabled in focus mode");
                this.value = "";
                return;
            }}
            
            clearHighlighting();
            
            const searchTerm = this.value.toLowerCase();
            
            if (searchTerm === "") {{
                nodes.style("opacity", 1.0);
                labels.style("opacity", 1.0);
            }} else {{
                nodes.style("opacity", d => 
                    d.name.toLowerCase().includes(searchTerm) ? 1.0 : 0.1
                );
                labels.style("opacity", d => 
                    d.name.toLowerCase().includes(searchTerm) ? 1.0 : 0.1
                );
            }}
        }});

        /* Enhanced drag functions */
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}

        function dragged(event, d) {{
            const flexMargin = 50;
            d.fx = Math.max(-flexMargin, Math.min(width + flexMargin, event.x));
            d.fy = Math.max(-flexMargin, Math.min(height + flexMargin, event.y));
        }}

        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}

        /* Add zoom and pan functionality */
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", function(event) {{
                const {{ transform }} = event;
                linkContainer.attr("transform", transform);
                nodeContainer.attr("transform", transform);
                labelContainer.attr("transform", transform);
            }});

        svg.call(zoom);

        /* Initialize the visualization */
        updateFPPairsList(); // Use the new function instead of populateFPPairsList

        // Enhanced window resize handler
        window.addEventListener('resize', function() {{
            const newWidth = window.innerWidth - 380;
            const newHeight = window.innerHeight;
            
            svg.attr("width", newWidth).attr("height", newHeight);
            
            width = newWidth;
            height = newHeight;
            
            simulation.force("center", d3.forceCenter(newWidth / 2, newHeight / 2));
            
            networkData.nodes.forEach(d => {{
                if (d.x) d.x = Math.max(margin + d.size, Math.min(newWidth - margin - d.size, d.x));
                if (d.y) d.y = Math.max(margin + d.size, Math.min(newHeight - margin - d.size, d.y));
            }});
            
            simulation.restart();
        }});

        /* Hide info panel when clicking elsewhere */
        svg.on("click", function(event) {{
            if (event.target.tagName === 'svg') {{
                d3.select("#infoPanel").style("display", "none");
            }}
        }});

        console.log("🎉 Enhanced FP Network Visualizer with SELECTIVE FILTERING and DUAL MODE loaded!");
        console.log(`📊 Network: ${{networkData.nodes.length}} nodes, ${{networkData.edges.length}} edges`);
        console.log(`🎯 FP Pairs: ${{fpPairsData.length}}`);
        console.log(`🔴 Normal Mode: Selected FP pairs + paths appear in RED with blinking animation`);
        console.log(`🔴⚫ Focus Mode: Selected drug/disease in RED, connection paths in BLACK`);
        console.log(`🧬 Gene Display: Using approved gene symbols instead of Ensembl IDs`);
        console.log(`🚫 Minimal Filtering: Only direct starting drug-to-drug-type connections filtered out`);
        console.log(`🎯 Smart Filtering: Click on disease/drug nodes or use dropdown to filter FP pairs`);
        console.log(`💡 More Paths: Should now find many more valid connection paths!`);
    </script>
</body>
</html>"""

    # Save the HTML file
    filename = f"FP_Interactive_Network_GeneSymbols_RedSelection_{datetime}.html"
    filepath = os.path.join(results_path, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"   ✓ Enhanced HTML file with RED SELECTION and FILTERED PATHS created: {filename}")
    
    return filepath


# MAIN FUNCTION TO CREATE WEB VISUALIZER WITH RED SELECTION AND FILTERED PATHS
def run_web_fp_visualizer(sample_size=5000, top_diseases=100, top_drugs=50, max_path_length=4, max_paths_per_pair=50):
    """
    Create the interactive web-based FP visualizer with red selection and filtered paths
    
    Args:
        sample_size: Number of FP predictions to analyze (5000 for maximum FP pairs)
        top_diseases: Number of top diseases to include (100 for comprehensive coverage)
        top_drugs: Number of top drugs to include (50 for comprehensive coverage)
        max_path_length: Maximum path length (4 recommended for performance)
        max_paths_per_pair: Maximum number of paths to find per FP pair
    """
    
    print("🌐 CREATING INTERACTIVE WEB-BASED FP VISUALIZER WITH RED SELECTION")
    print("="*70)
    print(f"🧬 NEW: Gene symbols (like CALCR, ABCC8) displayed instead of Ensembl IDs")
    print(f"🔴 NEW: Selected FP pairs - Normal: RED nodes+paths, Focus: RED nodes+BLACK paths")
    print(f"🚫 NEW: Only direct starting drug-to-drug-type connections filtered out")
    print(f"🛤️ Path Discovery Settings: Up to {max_paths_per_pair} paths per pair, max {max_path_length} steps")
    
    # Load essential components
    components = load_essential_components(results_path, datetime)
    if not components:
        print("❌ Failed to load essential components")
        return None
    
    graph = components['graph']
    transformer_fps = components['transformer_fps']
    
    # Create the web visualizer
    html_file, network_data = create_web_fp_visualizer(
        graph=graph,
        transformer_fps=transformer_fps,
        results_path=results_path,
        datetime=datetime,
        sample_size=sample_size,
        top_diseases=top_diseases,
        top_drugs=top_drugs,
        max_path_length=max_path_length,
        max_paths_per_pair=max_paths_per_pair
    )
    
    print(f"\n🎊 WEB VISUALIZER WITH RED SELECTION COMPLETE!")
    print(f"📁 File location: {html_file}")
    print(f"🚀 To use: Double-click the HTML file to open in your browser")
    print(f"💡 Features: Interactive network, search, filters, path highlighting, focus mode")
    print(f"🧬 NEW: Gene symbols displayed instead of Ensembl IDs (e.g., CALCR vs ENSG00000004948)")
    print(f"🔴 NEW: Normal mode - RED nodes+paths, Focus mode - RED nodes+BLACK paths")
    print(f"🚫 NEW: Only direct starting drug-to-drug-type connections filtered out")
    print(f"🛤️ Path Discovery: Up to {max_paths_per_pair} connection paths per FP pair")
    
    return html_file, network_data

# PERFORMANCE OPTIONS WITH DIFFERENT PATH LIMITS
def run_web_fp_visualizer_light_paths():
    """Conservative - fewer paths, faster processing"""
    return run_web_fp_visualizer(
        sample_size=300,
        top_diseases=8,
        top_drugs=5,
        max_path_length=3,
        max_paths_per_pair=25  # Moderate path discovery
    )

def run_web_fp_visualizer_moderate_paths():
    """Balanced - good performance with decent path coverage"""
    return run_web_fp_visualizer(
        sample_size=1000,
        top_diseases=15,
        top_drugs=12,
        max_path_length=4,
        max_paths_per_pair=50  # Current default
    )

def run_web_fp_visualizer_extensive_paths():
    """Comprehensive - maximum path discovery (slower but thorough)"""
    return run_web_fp_visualizer(
        sample_size=2000,
        top_diseases=25,
        top_drugs=20,
        max_path_length=5,
        max_paths_per_pair=100  # 🔥 EXTENSIVE path discovery!
    )

def run_web_fp_visualizer_maximum_paths():
    """MAXIMUM - Find as many paths as possible (very slow but complete)"""
    return run_web_fp_visualizer(
        sample_size=3000,
        top_diseases=30,
        top_drugs=25,
        max_path_length=6,
        max_paths_per_pair=200  # 🚀 MAXIMUM path discovery!
    )

# # USAGE EXAMPLES AND INSTRUCTIONS
# if __name__ == "__main__":
#     # IMPORTANT: Update these paths to match your setup
#     # results_path = r"C:\\OpenTargets_datasets\\FP_results\\"
#     # datetime = r"20250806003500"
    
#     print("🌐 ENHANCED WEB-BASED FP VISUALIZER WITH RED SELECTION & FILTERED PATHS")
#     print("="*80)
#     print("🔴 NEW FEATURES:")
#     print("  • Gene Symbols: Display CALCR, ABCC8, GABRA1 instead of ENSG00000004948, etc.")
#     print("  • Normal Mode: Selected FP pairs + paths appear in RED with blinking")
#     print("  • Focus Mode: Selected drug/disease in RED, connection paths in BLACK")
#     print("  • Minimal Filtering: Only direct starting drug-to-drug-type connections filtered out")
#     print("  • More Paths: Should now find many more valid biological connection paths")
#     print()
#     print("Choose your option:")
#     print("1. Light Paths (25 paths per pair) - Fast ⚡")
#     print("2. Moderate Paths (50 paths per pair) - Balanced ⚖️ - RECOMMENDED")
#     print("3. Extensive Paths (100 paths per pair) - Comprehensive 🔍")
#     print("4. Maximum Paths (200 paths per pair) - Complete 🚀")
#     print()
    
#     # RECOMMENDED: Moderate paths version
#     # html_file, network_data = run_web_fp_visualizer_moderate_paths()
    
#     # Alternative options:
#     html_file, network_data = run_web_fp_visualizer_light_paths()        # Faster
#     # html_file, network_data = run_web_fp_visualizer_extensive_paths()    # More detailed
#     # html_file, network_data = run_web_fp_visualizer_maximum_paths()      # Most comprehensive

def analyze_topological_patterns(high_confidence_fps, network_data, paths_data):
    """Analyze topological patterns in high-confidence predictions"""
    
    topology_stats = {
        'path_length_distribution': defaultdict(int),
        'common_motifs': defaultdict(int),
        'hub_usage': defaultdict(int),
        'node_type_sequences': defaultdict(int)
    }
    
    for fp_pair in high_confidence_fps:
        pair_key = f"{fp_pair['drug_name']} -> {fp_pair['disease_name']}"
        
        if pair_key in paths_data:
            for path in paths_data[pair_key]['paths']:
                # Path length analysis
                topology_stats['path_length_distribution'][path['length']] += 1
                
                # Node type sequence analysis
                type_sequence = ' -> '.join([node['type'] for node in path['nodes']])
                topology_stats['node_type_sequences'][type_sequence] += 1
                
                # Hub analysis (nodes with high degree)
                for node in path['nodes']:
                    if node['degree'] > 50:  # Define hub threshold
                        topology_stats['hub_usage'][node['name']] += 1
    
    return topology_stats

def find_topological_motifs(paths_data, min_frequency=5):
    """Find recurring topological motifs across high-confidence predictions"""
    
    motifs = defaultdict(int)
    
    for pair_key, data in paths_data.items():
        for path in data['paths']:
            # Extract 3-node motifs
            for i in range(len(path['nodes']) - 2):
                motif = tuple([path['nodes'][i]['type'], 
                              path['nodes'][i+1]['type'], 
                              path['nodes'][i+2]['type']])
                motifs[motif] += 1
    
    # Return only frequent motifs
    return {motif: count for motif, count in motifs.items() if count >= min_frequency}

# Add these functions to your existing script (FP_short_08_understand.py)
# Insert after your existing functions but before the main execution block

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def analyze_topological_patterns(high_confidence_fps, network_data, paths_data, confidence_threshold=0.8):
    """Analyze topological patterns in high-confidence predictions"""
    
    print(f"🔍 ANALYZING TOPOLOGICAL PATTERNS (confidence ≥ {confidence_threshold})")
    print("="*60)
    
    topology_stats = {
        'path_length_distribution': defaultdict(int),
        'common_motifs': defaultdict(int),
        'hub_usage': defaultdict(int),
        'node_type_sequences': defaultdict(int),
        'degree_distribution': [],
        'path_diversity': defaultdict(int)
    }
    
    # Filter for high confidence predictions
    high_conf_pairs = [fp for fp in high_confidence_fps if fp['confidence'] >= confidence_threshold]
    
    print(f"📊 Analyzing {len(high_conf_pairs)} high-confidence FP pairs...")
    
    total_paths = 0
    pairs_with_paths = 0
    
    for fp_pair in high_conf_pairs:
        pair_key = f"{fp_pair['drug_name']} -> {fp_pair['disease_name']}"
        
        if pair_key in paths_data and paths_data[pair_key]['paths_found'] > 0:
            pairs_with_paths += 1
            pair_paths = paths_data[pair_key]['paths']
            
            # Path diversity (number of different paths per pair)
            topology_stats['path_diversity'][len(pair_paths)] += 1
            
            for path in pair_paths:
                total_paths += 1
                
                # Path length analysis
                topology_stats['path_length_distribution'][path['length']] += 1
                
                # Node type sequence analysis
                type_sequence = ' -> '.join([node['type'] for node in path['nodes']])
                topology_stats['node_type_sequences'][type_sequence] += 1
                
                # Collect node degrees for analysis
                for node in path['nodes']:
                    node_data = next((n for n in network_data['nodes'] if n['id'] == node['idx']), None)
                    if node_data:
                        topology_stats['degree_distribution'].append(node_data['degree'])
                        
                        # Hub analysis (nodes with high degree)
                        if node_data['degree'] > 50:  # Define hub threshold
                            topology_stats['hub_usage'][node['name']] += 1
    
    print(f"✅ Analysis complete:")
    print(f"   • Total high-confidence pairs: {len(high_conf_pairs)}")
    print(f"   • Pairs with paths: {pairs_with_paths}")
    print(f"   • Total paths analyzed: {total_paths}")
    
    return topology_stats

def find_topological_motifs(paths_data, min_frequency=5):
    """Find recurring topological motifs across high-confidence predictions"""
    
    print(f"🧩 FINDING TOPOLOGICAL MOTIFS (min frequency: {min_frequency})")
    print("="*50)
    
    motifs_2node = defaultdict(int)
    motifs_3node = defaultdict(int)
    motifs_4node = defaultdict(int)
    
    total_motifs = 0
    
    for pair_key, data in paths_data.items():
        if data['paths_found'] > 0:
            for path in data['paths']:
                node_types = [node['type'] for node in path['nodes']]
                
                # Extract 2-node motifs
                for i in range(len(node_types) - 1):
                    motif = tuple(node_types[i:i+2])
                    motifs_2node[motif] += 1
                    total_motifs += 1
                
                # Extract 3-node motifs
                for i in range(len(node_types) - 2):
                    motif = tuple(node_types[i:i+3])
                    motifs_3node[motif] += 1
                
                # Extract 4-node motifs
                for i in range(len(node_types) - 3):
                    motif = tuple(node_types[i:i+4])
                    motifs_4node[motif] += 1
    
    # Filter by frequency
    frequent_motifs = {
        '2-node': {motif: count for motif, count in motifs_2node.items() if count >= min_frequency},
        '3-node': {motif: count for motif, count in motifs_3node.items() if count >= min_frequency},
        '4-node': {motif: count for motif, count in motifs_4node.items() if count >= min_frequency}
    }
    
    print(f"✅ Found motifs:")
    for motif_type, motifs in frequent_motifs.items():
        print(f"   • {motif_type}: {len(motifs)} frequent patterns")
    
    return frequent_motifs

def analyze_hub_dependency(topology_stats, network_data, top_n=20):
    """Analyze dependency on network hubs"""
    
    print(f"🌟 ANALYZING HUB DEPENDENCY (top {top_n} hubs)")
    print("="*40)
    
    # Get top hubs by usage
    top_hubs = sorted(topology_stats['hub_usage'].items(), 
                     key=lambda x: x[1], reverse=True)[:top_n]
    
    # Get top hubs by degree
    hub_nodes = sorted(network_data['nodes'], 
                      key=lambda x: x['degree'], reverse=True)[:top_n]
    
    print("🔝 Most frequently used hubs in high-confidence paths:")
    for i, (hub_name, usage_count) in enumerate(top_hubs, 1):
        hub_data = next((n for n in network_data['nodes'] if n['name'] == hub_name), None)
        degree = hub_data['degree'] if hub_data else 'Unknown'
        node_type = hub_data['type'] if hub_data else 'Unknown'
        print(f"   {i:2d}. {hub_name} ({node_type}) - Used {usage_count} times, Degree: {degree}")
    
    print(f"\n🌐 Highest degree nodes in network:")
    for i, node in enumerate(hub_nodes, 1):
        usage = topology_stats['hub_usage'].get(node['name'], 0)
        print(f"   {i:2d}. {node['name']} ({node['type']}) - Degree: {node['degree']}, Used: {usage} times")
    
    return top_hubs, hub_nodes

def create_topology_report(topology_stats, motifs, results_path, datetime):
    """Create a comprehensive topology analysis report"""
    
    print("📝 CREATING TOPOLOGY ANALYSIS REPORT")
    print("="*40)
    
    report_filename = f"Topology_Analysis_Report_{datetime}.txt"
    report_path = os.path.join(results_path, report_filename)
    
    with open(report_path, 'w') as f:
        f.write("TOPOLOGICAL ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        # Path length distribution
        f.write("PATH LENGTH DISTRIBUTION:\n")
        f.write("-"*30 + "\n")
        for length, count in sorted(topology_stats['path_length_distribution'].items()):
            f.write(f"Length {length}: {count} paths\n")
        f.write("\n")
        
        # Most common node type sequences
        f.write("TOP NODE TYPE SEQUENCES:\n")
        f.write("-"*30 + "\n")
        top_sequences = sorted(topology_stats['node_type_sequences'].items(), 
                              key=lambda x: x[1], reverse=True)[:20]
        for sequence, count in top_sequences:
            f.write(f"{sequence}: {count} times\n")
        f.write("\n")
        
        # Path diversity
        f.write("PATH DIVERSITY (paths per FP pair):\n")
        f.write("-"*30 + "\n")
        for num_paths, count in sorted(topology_stats['path_diversity'].items()):
            f.write(f"{num_paths} paths: {count} FP pairs\n")
        f.write("\n")
        
        # Most frequent motifs
        f.write("MOST FREQUENT MOTIFS:\n")
        f.write("-"*30 + "\n")
        for motif_type, motif_dict in motifs.items():
            f.write(f"\n{motif_type.upper()} MOTIFS:\n")
            top_motifs = sorted(motif_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            for motif, count in top_motifs:
                motif_str = " -> ".join(motif)
                f.write(f"  {motif_str}: {count} times\n")
        f.write("\n")
        
        # Hub usage
        f.write("TOP NETWORK HUBS USED:\n")
        f.write("-"*30 + "\n")
        top_hubs = sorted(topology_stats['hub_usage'].items(), 
                         key=lambda x: x[1], reverse=True)[:15]
        for hub_name, usage_count in top_hubs:
            f.write(f"{hub_name}: {usage_count} times\n")
    
    print(f"✅ Report saved: {report_filename}")
    return report_path

def visualize_topology_patterns(topology_stats, motifs, results_path, datetime):
    """Create visualization plots for topology analysis"""
    
    print("📊 CREATING TOPOLOGY VISUALIZATIONS")
    print("="*40)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Topological Analysis of High-Confidence FP Predictions', fontsize=16, fontweight='bold')
    
    # 1. Path Length Distribution
    ax1 = axes[0, 0]
    lengths = list(topology_stats['path_length_distribution'].keys())
    counts = list(topology_stats['path_length_distribution'].values())
    ax1.bar(lengths, counts, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Path Length')
    ax1.set_ylabel('Number of Paths')
    ax1.set_title('Path Length Distribution')
    ax1.grid(True, alpha=0.3)
    
    # 2. Path Diversity
    ax2 = axes[0, 1]
    diversity_lengths = list(topology_stats['path_diversity'].keys())
    diversity_counts = list(topology_stats['path_diversity'].values())
    ax2.bar(diversity_lengths, diversity_counts, color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Number of Paths per FP Pair')
    ax2.set_ylabel('Number of FP Pairs')
    ax2.set_title('Path Diversity per FP Pair')
    ax2.grid(True, alpha=0.3)
    
    # 3. Top 2-node Motifs
    ax3 = axes[1, 0]
    if motifs['2-node']:
        top_2node = sorted(motifs['2-node'].items(), key=lambda x: x[1], reverse=True)[:8]
        motif_labels = [' -> '.join(motif) for motif, _ in top_2node]
        motif_counts = [count for _, count in top_2node]
        ax3.barh(range(len(motif_labels)), motif_counts, color='lightgreen', alpha=0.7)
        ax3.set_yticks(range(len(motif_labels)))
        ax3.set_yticklabels(motif_labels)
        ax3.set_xlabel('Frequency')
        ax3.set_title('Top 2-Node Motifs')
    
    # 4. Node Degree Distribution in Paths
    ax4 = axes[1, 1]
    if topology_stats['degree_distribution']:
        ax4.hist(topology_stats['degree_distribution'], bins=30, color='gold', alpha=0.7)
        ax4.set_xlabel('Node Degree')
        ax4.set_ylabel('Frequency in Paths')
        ax4.set_title('Degree Distribution of Nodes in Paths')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"Topology_Analysis_Plots_{datetime}.png"
    plot_path = os.path.join(results_path, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Plots saved: {plot_filename}")
    return plot_path

# MAIN TOPOLOGY ANALYSIS RUNNER FUNCTION
def run_comprehensive_topology_analysis(confidence_threshold=0.8, min_motif_frequency=5):
    """
    Run comprehensive topology analysis on your FP predictions
    
    Args:
        confidence_threshold: Minimum confidence for high-confidence analysis (0.8 recommended)
        min_motif_frequency: Minimum frequency for motif detection (5 recommended)
    """
    
    print("🧠 COMPREHENSIVE TOPOLOGY ANALYSIS")
    print("="*60)
    print(f"🎯 Settings: confidence ≥ {confidence_threshold}, motif frequency ≥ {min_motif_frequency}")
    
    # Load your data (using your existing functions)
    print("📂 Loading data...")
    components = load_essential_components(results_path, datetime)
    if not components:
        print("❌ Failed to load essential components")
        return None
    
    # Get network data for topology analysis
    print("🔗 Preparing network data...")
    graph = components['graph']
    transformer_fps = components['transformer_fps']
    
    # Convert to format needed for analysis
    fp_pairs_data = []
    for i, (drug_name, disease_name, confidence) in enumerate(transformer_fps[:1000]):  # Limit for analysis
        if drug_name in approved_drugs_list_name and disease_name in disease_list_name:
            drug_idx = approved_drugs_list_name.index(drug_name)
            disease_list_pos = disease_list_name.index(disease_name)
            disease_id = disease_list[disease_list_pos]
            disease_idx = disease_key_mapping[disease_id]
            
            fp_pairs_data.append({
                'drug_name': drug_name,
                'disease_name': disease_name,
                'drug_idx': drug_idx,
                'disease_idx': disease_idx,
                'confidence': confidence,
                'pair_name': f"{drug_name} -> {disease_name}"
            })
    
    # Get network data
    G = to_networkx(graph, to_undirected=True)
    network_data = {
        'nodes': [
            {
                'id': node_idx,
                'name': idx_to_name.get(node_idx, f"Node_{node_idx}"),
                'type': idx_to_type.get(node_idx, "Unknown"),
                'degree': G.degree(node_idx)
            }
            for node_idx in G.nodes()
        ]
    }
    
    # Find paths for analysis
    print("🛤️ Finding paths (this may take a moment)...")
    paths_data = find_web_optimized_paths_filtered(G, fp_pairs_data[:100], max_path_length=4, max_paths_per_pair=50)
    
    # Run topology analysis
    print("\n🔍 Running topology analysis...")
    topology_stats = analyze_topological_patterns(fp_pairs_data, network_data, paths_data, confidence_threshold)
    
    print("\n🧩 Finding topological motifs...")
    motifs = find_topological_motifs(paths_data, min_motif_frequency)
    
    print("\n🌟 Analyzing hub dependency...")
    top_hubs, hub_nodes = analyze_hub_dependency(topology_stats, network_data)
    
    # Create reports and visualizations
    print("\n📝 Creating reports...")
    report_path = create_topology_report(topology_stats, motifs, results_path, datetime)
    
    print("\n📊 Creating visualizations...")
    plot_path = visualize_topology_patterns(topology_stats, motifs, results_path, datetime)
    
    print("\n🎉 TOPOLOGY ANALYSIS COMPLETE!")
    print(f"📁 Report: {report_path}")
    print(f"📊 Plots: {plot_path}")
    
    return {
        'topology_stats': topology_stats,
        'motifs': motifs,
        'top_hubs': top_hubs,
        'hub_nodes': hub_nodes,
        'report_path': report_path,
        'plot_path': plot_path
    }

# QUICK RUNNER FUNCTIONS
def run_topology_analysis_quick():
    """Quick topology analysis with standard settings"""
    return run_comprehensive_topology_analysis(confidence_threshold=0.8, min_motif_frequency=3)

def run_topology_analysis_detailed():
    """Detailed topology analysis with higher thresholds"""
    return run_comprehensive_topology_analysis(confidence_threshold=0.9, min_motif_frequency=10)

# Add this to the end of your FP_short_08_understand.py file

# if __name__ == "__main__":
#     # Your existing code...
    
#     # NEW: Run topology analysis
#     print("\n" + "="*80)
#     print("🧠 STARTING TOPOLOGY ANALYSIS")
#     print("="*80)
    
#     # Choose one of these options:
    
#     # OPTION 1: Quick analysis (recommended to start)
#     results = run_topology_analysis_quick()
    
#     # OPTION 2: Detailed analysis (for high-confidence focus)
#     # results = run_topology_analysis_detailed()
    
#     # OPTION 3: Custom analysis
#     # results = run_comprehensive_topology_analysis(confidence_threshold=0.85, min_motif_frequency=7)
    
#     if results:
#         print("\n🎊 TOPOLOGY ANALYSIS RESULTS:")
#         print(f"📊 Path Length Distribution: {dict(results['topology_stats']['path_length_distribution'])}")
#         print(f"🧩 Found {len(results['motifs']['2-node'])} frequent 2-node motifs")
#         print(f"🌟 Top hub: {results['top_hubs'][0] if results['top_hubs'] else 'None'}")
#         print(f"📁 Full report saved at: {results['report_path']}")
#         print(f"📊 Visualizations saved at: {results['plot_path']}")


def analyze_hub_selectivity(topology_stats, network_data, paths_data, top_n=50):
    """
    Analyze why certain hubs are used while others aren't
    This will help interpret your fascinating selectivity pattern
    """
    
    print("🎯 ANALYZING HUB SELECTIVITY PATTERNS")
    print("="*50)
    
    # Get all nodes sorted by degree
    all_nodes_by_degree = sorted(network_data['nodes'], key=lambda x: x['degree'], reverse=True)
    
    # Categorize nodes by usage vs degree
    used_hubs = []
    unused_hubs = []
    
    for node in all_nodes_by_degree[:top_n]:
        usage_count = topology_stats['hub_usage'].get(node['name'], 0)
        
        node_info = {
            'name': node['name'],
            'type': node['type'],
            'degree': node['degree'],
            'usage': usage_count,
            'usage_rate': usage_count / node['degree'] if node['degree'] > 0 else 0
        }
        
        if usage_count > 0:
            used_hubs.append(node_info)
        else:
            unused_hubs.append(node_info)
    
    print(f"📊 HUB USAGE ANALYSIS (Top {top_n} by degree):")
    print(f"   • Used hubs: {len(used_hubs)}")
    print(f"   • Unused hubs: {len(unused_hubs)}")
    
    print(f"\n🔥 MOST EFFICIENTLY USED HUBS (usage per degree):")
    efficient_hubs = sorted(used_hubs, key=lambda x: x['usage_rate'], reverse=True)[:10]
    for i, hub in enumerate(efficient_hubs, 1):
        print(f"   {i:2d}. {hub['name'][:30]:30s} ({hub['type']:12s}) - "
              f"Rate: {hub['usage_rate']:.3f} ({hub['usage']} uses / {hub['degree']} degree)")
    
    print(f"\n❌ HIGH-DEGREE BUT UNUSED HUBS:")
    unused_sorted = sorted(unused_hubs, key=lambda x: x['degree'], reverse=True)[:10]
    for i, hub in enumerate(unused_sorted, 1):
        print(f"   {i:2d}. {hub['name'][:30]:30s} ({hub['type']:12s}) - "
              f"Degree: {hub['degree']} (UNUSED)")
    
    # Analyze by node type
    print(f"\n📈 USAGE BY NODE TYPE:")
    type_stats = {}
    for hub in used_hubs + unused_hubs:
        node_type = hub['type']
        if node_type not in type_stats:
            type_stats[node_type] = {'used': 0, 'unused': 0, 'total_usage': 0}
        
        if hub['usage'] > 0:
            type_stats[node_type]['used'] += 1
            type_stats[node_type]['total_usage'] += hub['usage']
        else:
            type_stats[node_type]['unused'] += 1
    
    for node_type, stats in sorted(type_stats.items(), key=lambda x: x[1]['total_usage'], reverse=True):
        total = stats['used'] + stats['unused']
        usage_pct = (stats['used'] / total * 100) if total > 0 else 0
        print(f"   {node_type:15s}: {stats['used']:3d}/{total:3d} used ({usage_pct:5.1f}%), "
              f"Total usage: {stats['total_usage']:4d}")
    
    return used_hubs, unused_hubs, type_stats

def analyze_biological_pathway_patterns(paths_data, min_paths=3):
    """
    Analyze common biological pathway patterns in high-confidence predictions
    """
    
    print(f"\n🧬 ANALYZING BIOLOGICAL PATHWAY PATTERNS")
    print("="*50)
    
    # Track pathway patterns
    pathway_patterns = defaultdict(int)
    drug_disease_patterns = defaultdict(list)
    
    for pair_key, data in paths_data.items():
        if data['paths_found'] >= min_paths:
            
            # Extract drug and disease from pair_key
            parts = pair_key.split(' -> ')
            if len(parts) == 2:
                drug_name, disease_name = parts
                
                for path in data['paths']:
                    # Create pathway signature
                    path_signature = []
                    for node in path['nodes']:
                        if node['type'] in ['Gene', 'Pathway']:
                            path_signature.append(f"{node['type']}:{node['name']}")
                    
                    if len(path_signature) >= 2:  # At least 2 biological nodes
                        signature_str = ' -> '.join(path_signature)
                        pathway_patterns[signature_str] += 1
                        drug_disease_patterns[signature_str].append((drug_name, disease_name))
    
    print(f"🔍 MOST COMMON BIOLOGICAL PATHWAYS:")
    top_pathways = sorted(pathway_patterns.items(), key=lambda x: x[1], reverse=True)[:15]
    
    for i, (pathway, count) in enumerate(top_pathways, 1):
        print(f"   {i:2d}. Used {count:3d} times: {pathway}")
        
        # Show example drug-disease pairs using this pathway
        examples = drug_disease_patterns[pathway][:3]
        for drug, disease in examples:
            print(f"       → {drug} → {disease}")
        if len(examples) < len(drug_disease_patterns[pathway]):
            remaining = len(drug_disease_patterns[pathway]) - len(examples)
            print(f"       ... and {remaining} more pairs")
        print()
    
    return pathway_patterns, drug_disease_patterns

def compare_high_vs_low_confidence_topology(transformer_fps, paths_data, network_data, 
                                          high_threshold=0.9, low_threshold=0.3):
    """
    Compare topological patterns between high and low confidence predictions
    """
    
    print(f"\n⚖️ COMPARING HIGH vs LOW CONFIDENCE TOPOLOGY")
    print("="*50)
    
    # Separate high and low confidence predictions
    high_conf_pairs = []
    low_conf_pairs = []
    
    for drug_name, disease_name, confidence in transformer_fps[:2000]:  # Analyze more pairs
        pair_key = f"{drug_name} -> {disease_name}"
        
        if confidence >= high_threshold and pair_key in paths_data:
            high_conf_pairs.append((pair_key, confidence, paths_data[pair_key]))
        elif confidence <= low_threshold and pair_key in paths_data:
            low_conf_pairs.append((pair_key, confidence, paths_data[pair_key]))
    
    print(f"📊 COMPARISON GROUPS:")
    print(f"   • High confidence (≥{high_threshold}): {len(high_conf_pairs)} pairs")
    print(f"   • Low confidence (≤{low_threshold}): {len(low_conf_pairs)} pairs")
    
    # Compare path characteristics
    def analyze_group(pairs, group_name):
        total_paths = 0
        avg_path_length = 0
        hub_usage = defaultdict(int)
        path_count = 0
        
        for pair_key, conf, path_data in pairs:
            if path_data['paths_found'] > 0:
                for path in path_data['paths']:
                    total_paths += 1
                    avg_path_length += path['length']
                    
                    # Count hub usage
                    for node in path['nodes']:
                        node_data = next((n for n in network_data['nodes'] if n['id'] == node['idx']), None)
                        if node_data and node_data['degree'] > 100:  # Define hub threshold
                            hub_usage[node['name']] += 1
        
        avg_path_length = avg_path_length / total_paths if total_paths > 0 else 0
        
        print(f"\n{group_name} GROUP CHARACTERISTICS:")
        print(f"   • Total paths found: {total_paths}")
        print(f"   • Average path length: {avg_path_length:.2f}")
        print(f"   • Unique hubs used: {len(hub_usage)}")
        
        # Top hubs for this group
        top_hubs = sorted(hub_usage.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"   • Top hubs:")
        for hub_name, usage in top_hubs:
            print(f"     - {hub_name}: {usage} times")
        
        return {
            'total_paths': total_paths,
            'avg_path_length': avg_path_length,
            'hub_usage': dict(hub_usage)
        }
    
    high_stats = analyze_group(high_conf_pairs, "HIGH CONFIDENCE")
    low_stats = analyze_group(low_conf_pairs, "LOW CONFIDENCE")
    
    # Compare the groups
    print(f"\n🎯 KEY DIFFERENCES:")
    if high_stats['avg_path_length'] > 0 and low_stats['avg_path_length'] > 0:
        length_diff = high_stats['avg_path_length'] - low_stats['avg_path_length']
        print(f"   • Path length difference: {length_diff:+.2f} steps")
        
        if abs(length_diff) > 0.5:
            if length_diff > 0:
                print("     → High confidence predictions use LONGER paths")
                print("     → Suggests more complex biological mechanisms")
            else:
                print("     → High confidence predictions use SHORTER paths")
                print("     → Suggests more direct biological connections")
    
    # Compare hub usage overlap
    high_hubs = set(high_stats['hub_usage'].keys())
    low_hubs = set(low_stats['hub_usage'].keys())
    common_hubs = high_hubs.intersection(low_hubs)
    
    print(f"   • Hub usage overlap: {len(common_hubs)}/{len(high_hubs.union(low_hubs))} "
          f"({len(common_hubs)/len(high_hubs.union(low_hubs))*100:.1f}%)")
    
    if len(common_hubs) < len(high_hubs) * 0.5:
        print("     → HIGH and LOW confidence predictions use DIFFERENT hubs!")
        print("     → Strong evidence for topology-driven selectivity")
    
    return high_stats, low_stats


# Enhanced Topology Analysis - Add these functions to your script

def analyze_hub_selectivity(topology_stats, network_data, paths_data, top_n=50):
    """
    Analyze why certain hubs are used while others aren't
    This will help interpret your fascinating selectivity pattern
    """
    
    print("🎯 ANALYZING HUB SELECTIVITY PATTERNS")
    print("="*50)
    
    # Get all nodes sorted by degree
    all_nodes_by_degree = sorted(network_data['nodes'], key=lambda x: x['degree'], reverse=True)
    
    # Categorize nodes by usage vs degree
    used_hubs = []
    unused_hubs = []
    
    for node in all_nodes_by_degree[:top_n]:
        usage_count = topology_stats['hub_usage'].get(node['name'], 0)
        
        node_info = {
            'name': node['name'],
            'type': node['type'],
            'degree': node['degree'],
            'usage': usage_count,
            'usage_rate': usage_count / node['degree'] if node['degree'] > 0 else 0
        }
        
        if usage_count > 0:
            used_hubs.append(node_info)
        else:
            unused_hubs.append(node_info)
    
    print(f"📊 HUB USAGE ANALYSIS (Top {top_n} by degree):")
    print(f"   • Used hubs: {len(used_hubs)}")
    print(f"   • Unused hubs: {len(unused_hubs)}")
    
    print(f"\n🔥 MOST EFFICIENTLY USED HUBS (usage per degree):")
    efficient_hubs = sorted(used_hubs, key=lambda x: x['usage_rate'], reverse=True)[:10]
    for i, hub in enumerate(efficient_hubs, 1):
        print(f"   {i:2d}. {hub['name'][:30]:30s} ({hub['type']:12s}) - "
              f"Rate: {hub['usage_rate']:.3f} ({hub['usage']} uses / {hub['degree']} degree)")
    
    print(f"\n❌ HIGH-DEGREE BUT UNUSED HUBS:")
    unused_sorted = sorted(unused_hubs, key=lambda x: x['degree'], reverse=True)[:10]
    for i, hub in enumerate(unused_sorted, 1):
        print(f"   {i:2d}. {hub['name'][:30]:30s} ({hub['type']:12s}) - "
              f"Degree: {hub['degree']} (UNUSED)")
    
    # Analyze by node type
    print(f"\n📈 USAGE BY NODE TYPE:")
    type_stats = {}
    for hub in used_hubs + unused_hubs:
        node_type = hub['type']
        if node_type not in type_stats:
            type_stats[node_type] = {'used': 0, 'unused': 0, 'total_usage': 0}
        
        if hub['usage'] > 0:
            type_stats[node_type]['used'] += 1
            type_stats[node_type]['total_usage'] += hub['usage']
        else:
            type_stats[node_type]['unused'] += 1
    
    for node_type, stats in sorted(type_stats.items(), key=lambda x: x[1]['total_usage'], reverse=True):
        total = stats['used'] + stats['unused']
        usage_pct = (stats['used'] / total * 100) if total > 0 else 0
        print(f"   {node_type:15s}: {stats['used']:3d}/{total:3d} used ({usage_pct:5.1f}%), "
              f"Total usage: {stats['total_usage']:4d}")
    
    return used_hubs, unused_hubs, type_stats

def analyze_biological_pathway_patterns(paths_data, min_paths=3):
    """
    Analyze common biological pathway patterns in high-confidence predictions
    """
    
    print(f"\n🧬 ANALYZING BIOLOGICAL PATHWAY PATTERNS")
    print("="*50)
    
    # Track pathway patterns
    pathway_patterns = defaultdict(int)
    drug_disease_patterns = defaultdict(list)
    
    for pair_key, data in paths_data.items():
        if data['paths_found'] >= min_paths:
            
            # Extract drug and disease from pair_key
            parts = pair_key.split(' -> ')
            if len(parts) == 2:
                drug_name, disease_name = parts
                
                for path in data['paths']:
                    # Create pathway signature
                    path_signature = []
                    for node in path['nodes']:
                        if node['type'] in ['Gene', 'Pathway']:
                            path_signature.append(f"{node['type']}:{node['name']}")
                    
                    if len(path_signature) >= 2:  # At least 2 biological nodes
                        signature_str = ' -> '.join(path_signature)
                        pathway_patterns[signature_str] += 1
                        drug_disease_patterns[signature_str].append((drug_name, disease_name))
    
    print(f"🔍 MOST COMMON BIOLOGICAL PATHWAYS:")
    top_pathways = sorted(pathway_patterns.items(), key=lambda x: x[1], reverse=True)[:15]
    
    for i, (pathway, count) in enumerate(top_pathways, 1):
        print(f"   {i:2d}. Used {count:3d} times: {pathway}")
        
        # Show example drug-disease pairs using this pathway
        examples = drug_disease_patterns[pathway][:3]
        for drug, disease in examples:
            print(f"       → {drug} → {disease}")
        if len(examples) < len(drug_disease_patterns[pathway]):
            remaining = len(drug_disease_patterns[pathway]) - len(examples)
            print(f"       ... and {remaining} more pairs")
        print()
    
    return pathway_patterns, drug_disease_patterns

def compare_high_vs_low_confidence_topology(transformer_fps, paths_data, network_data, 
                                          high_threshold=0.9, low_threshold=0.3):
    """
    Compare topological patterns between high and low confidence predictions
    """
    
    print(f"\n⚖️ COMPARING HIGH vs LOW CONFIDENCE TOPOLOGY")
    print("="*50)
    
    # Separate high and low confidence predictions
    high_conf_pairs = []
    low_conf_pairs = []
    
    for drug_name, disease_name, confidence in transformer_fps[:2000]:  # Analyze more pairs
        pair_key = f"{drug_name} -> {disease_name}"
        
        if confidence >= high_threshold and pair_key in paths_data:
            high_conf_pairs.append((pair_key, confidence, paths_data[pair_key]))
        elif confidence <= low_threshold and pair_key in paths_data:
            low_conf_pairs.append((pair_key, confidence, paths_data[pair_key]))
    
    print(f"📊 COMPARISON GROUPS:")
    print(f"   • High confidence (≥{high_threshold}): {len(high_conf_pairs)} pairs")
    print(f"   • Low confidence (≤{low_threshold}): {len(low_conf_pairs)} pairs")
    
    # Compare path characteristics
    def analyze_group(pairs, group_name):
        total_paths = 0
        avg_path_length = 0
        hub_usage = defaultdict(int)
        path_count = 0
        
        for pair_key, conf, path_data in pairs:
            if path_data['paths_found'] > 0:
                for path in path_data['paths']:
                    total_paths += 1
                    avg_path_length += path['length']
                    
                    # Count hub usage
                    for node in path['nodes']:
                        node_data = next((n for n in network_data['nodes'] if n['id'] == node['idx']), None)
                        if node_data and node_data['degree'] > 100:  # Define hub threshold
                            hub_usage[node['name']] += 1
        
        avg_path_length = avg_path_length / total_paths if total_paths > 0 else 0
        
        print(f"\n{group_name} GROUP CHARACTERISTICS:")
        print(f"   • Total paths found: {total_paths}")
        print(f"   • Average path length: {avg_path_length:.2f}")
        print(f"   • Unique hubs used: {len(hub_usage)}")
        
        # Top hubs for this group
        top_hubs = sorted(hub_usage.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"   • Top hubs:")
        for hub_name, usage in top_hubs:
            print(f"     - {hub_name}: {usage} times")
        
        return {
            'total_paths': total_paths,
            'avg_path_length': avg_path_length,
            'hub_usage': dict(hub_usage)
        }
    
    high_stats = analyze_group(high_conf_pairs, "HIGH CONFIDENCE")
    low_stats = analyze_group(low_conf_pairs, "LOW CONFIDENCE")
    
    # Compare the groups
    print(f"\n🎯 KEY DIFFERENCES:")
    if high_stats['avg_path_length'] > 0 and low_stats['avg_path_length'] > 0:
        length_diff = high_stats['avg_path_length'] - low_stats['avg_path_length']
        print(f"   • Path length difference: {length_diff:+.2f} steps")
        
        if abs(length_diff) > 0.5:
            if length_diff > 0:
                print("     → High confidence predictions use LONGER paths")
                print("     → Suggests more complex biological mechanisms")
            else:
                print("     → High confidence predictions use SHORTER paths")
                print("     → Suggests more direct biological connections")
    
    # Compare hub usage overlap
    high_hubs = set(high_stats['hub_usage'].keys())
    low_hubs = set(low_stats['hub_usage'].keys())
    common_hubs = high_hubs.intersection(low_hubs)
    
    print(f"   • Hub usage overlap: {len(common_hubs)}/{len(high_hubs.union(low_hubs))} "
          f"({len(common_hubs)/len(high_hubs.union(low_hubs))*100:.1f}%)")
    
    if len(common_hubs) < len(high_hubs) * 0.5:
        print("     → HIGH and LOW confidence predictions use DIFFERENT hubs!")
        print("     → Strong evidence for topology-driven selectivity")
    
    return high_stats, low_stats



def analyze_eye_infection_dominance(transformer_fps, paths_data, network_data):
    """
    Analyze why eye infection appears so frequently in high-confidence predictions
    """
    
    print("👁️ ANALYZING EYE INFECTION DOMINANCE")
    print("="*60)
    
    # Count eye infection occurrences
    eye_infection_pairs = []
    total_pairs = 0
    eye_infection_confidences = []
    
    for drug_name, disease_name, confidence in transformer_fps[:5000]:  # Check more pairs
        total_pairs += 1
        if "eye infection" in disease_name.lower():
            eye_infection_pairs.append((drug_name, disease_name, confidence))
            eye_infection_confidences.append(confidence)
    
    print(f"📊 EYE INFECTION STATISTICS:")
    print(f"   • Eye infection pairs: {len(eye_infection_pairs)}/{total_pairs} ({len(eye_infection_pairs)/total_pairs*100:.1f}%)")
    print(f"   • Average confidence: {sum(eye_infection_confidences)/len(eye_infection_confidences):.4f}")
    print(f"   • Max confidence: {max(eye_infection_confidences):.4f}")
    print(f"   • Min confidence: {min(eye_infection_confidences):.4f}")
    
    # Find eye infection in your disease mapping
    eye_infection_disease_id = None
    eye_infection_idx = None
    
    for i, disease_name in enumerate(disease_list_name):
        if "eye infection" in disease_name.lower():
            eye_infection_disease_id = disease_list[i]
            eye_infection_idx = disease_key_mapping[eye_infection_disease_id]
            print(f"   • Found in network: '{disease_name}' (ID: {eye_infection_disease_id}, idx: {eye_infection_idx})")
            break
    
    # Analyze network properties of eye infection
    if eye_infection_idx:
        eye_node = next((n for n in network_data['nodes'] if n['id'] == eye_infection_idx), None)
        if eye_node:
            print(f"\n🔍 EYE INFECTION NETWORK PROPERTIES:")
            print(f"   • Degree: {eye_node['degree']}")
            print(f"   • Type: {eye_node['type']}")
            print(f"   • Name: {eye_node['name']}")
    
    # Top drugs for eye infection
    print(f"\n💊 TOP DRUGS FOR EYE INFECTION (by confidence):")
    top_eye_drugs = sorted(eye_infection_pairs, key=lambda x: x[2], reverse=True)[:15]
    for i, (drug, disease, conf) in enumerate(top_eye_drugs, 1):
        print(f"   {i:2d}. {drug[:30]:30s} → {conf:.4f}")
    
    return eye_infection_pairs, eye_infection_idx

def analyze_eye_infection_pathways(eye_infection_idx, paths_data, network_data, top_n=10):
    """
    Analyze the specific pathways leading to eye infection predictions
    """
    
    print(f"\n🛤️ ANALYZING EYE INFECTION PATHWAYS")
    print("="*50)
    
    eye_infection_paths = []
    
    # Find all paths involving eye infection
    for pair_key, data in paths_data.items():
        if "eye infection" in pair_key.lower() and data['paths_found'] > 0:
            for path in data['paths']:
                eye_infection_paths.append({
                    'pair': pair_key,
                    'path': path,
                    'confidence': data['pair']['confidence']
                })
    
    print(f"📊 Found {len(eye_infection_paths)} paths involving eye infection")
    
    # Analyze common genes in eye infection paths
    gene_usage = defaultdict(int)
    pathway_patterns = defaultdict(int)
    
    for path_info in eye_infection_paths:
        path = path_info['path']
        
        # Count gene usage
        for node in path['nodes']:
            if node['type'] == 'Gene':
                gene_usage[node['name']] += 1
        
        # Analyze pathway patterns
        gene_sequence = []
        for node in path['nodes']:
            if node['type'] == 'Gene':
                gene_sequence.append(node['name'])
        
        if len(gene_sequence) >= 2:
            pattern = ' → '.join(gene_sequence)
            pathway_patterns[pattern] += 1
    
    print(f"\n🧬 MOST USED GENES IN EYE INFECTION PATHS:")
    top_genes = sorted(gene_usage.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for i, (gene, count) in enumerate(top_genes, 1):
        print(f"   {i:2d}. {gene[:25]:25s} - Used {count:3d} times")
    
    print(f"\n🔗 MOST COMMON GENE PATHWAYS TO EYE INFECTION:")
    top_patterns = sorted(pathway_patterns.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for i, (pattern, count) in enumerate(top_patterns, 1):
        print(f"   {i:2d}. {pattern} ({count} times)")
    
    return gene_usage, pathway_patterns

def compare_eye_infection_vs_others(transformer_fps, paths_data, sample_size=1000):
    """
    Compare eye infection predictions with other diseases
    """
    
    print(f"\n⚖️ COMPARING EYE INFECTION VS OTHER DISEASES")
    print("="*55)
    
    eye_infection_stats = {'confidences': [], 'path_lengths': [], 'path_counts': []}
    other_disease_stats = {'confidences': [], 'path_lengths': [], 'path_counts': []}
    
    for drug_name, disease_name, confidence in transformer_fps[:sample_size]:
        pair_key = f"{drug_name} -> {disease_name}"
        
        if pair_key in paths_data:
            path_data = paths_data[pair_key]
            
            if "eye infection" in disease_name.lower():
                eye_infection_stats['confidences'].append(confidence)
                if path_data['paths_found'] > 0:
                    eye_infection_stats['path_counts'].append(path_data['paths_found'])
                    avg_length = sum(p['length'] for p in path_data['paths']) / len(path_data['paths'])
                    eye_infection_stats['path_lengths'].append(avg_length)
            else:
                other_disease_stats['confidences'].append(confidence)
                if path_data['paths_found'] > 0:
                    other_disease_stats['path_counts'].append(path_data['paths_found'])
                    avg_length = sum(p['length'] for p in path_data['paths']) / len(path_data['paths'])
                    other_disease_stats['path_lengths'].append(avg_length)
    
    # Calculate averages
    def safe_avg(lst):
        return sum(lst) / len(lst) if lst else 0
    
    print(f"📊 COMPARISON RESULTS:")
    print(f"   EYE INFECTION:")
    print(f"   • Sample size: {len(eye_infection_stats['confidences'])}")
    print(f"   • Avg confidence: {safe_avg(eye_infection_stats['confidences']):.4f}")
    print(f"   • Avg path count: {safe_avg(eye_infection_stats['path_counts']):.1f}")
    print(f"   • Avg path length: {safe_avg(eye_infection_stats['path_lengths']):.1f}")
    
    print(f"   OTHER DISEASES:")
    print(f"   • Sample size: {len(other_disease_stats['confidences'])}")
    print(f"   • Avg confidence: {safe_avg(other_disease_stats['confidences']):.4f}")
    print(f"   • Avg path count: {safe_avg(other_disease_stats['path_counts']):.1f}")
    print(f"   • Avg path length: {safe_avg(other_disease_stats['path_lengths']):.1f}")
    
    # Key insights
    eye_conf = safe_avg(eye_infection_stats['confidences'])
    other_conf = safe_avg(other_disease_stats['confidences'])
    
    if eye_conf > other_conf:
        diff = eye_conf - other_conf
        print(f"\n🎯 KEY FINDING: Eye infection has {diff:.4f} HIGHER average confidence!")
        print(f"   This suggests the model has learned strong patterns for eye infection.")
    
    return eye_infection_stats, other_disease_stats

def investigate_eye_infection_bias(network_data, eye_infection_idx):
    """
    Investigate if there's a structural bias toward eye infection in the network
    """
    
    print(f"\n🔍 INVESTIGATING POTENTIAL EYE INFECTION BIAS")
    print("="*50)
    
    if not eye_infection_idx:
        print("❌ Eye infection not found in network")
        return None
    
    # Find eye infection node
    eye_node = next((n for n in network_data['nodes'] if n['id'] == eye_infection_idx), None)
    
    if not eye_node:
        print("❌ Eye infection node not found")
        return None
    
    print(f"🎯 EYE INFECTION NODE ANALYSIS:")
    print(f"   • Degree: {eye_node['degree']}")
    print(f"   • Type: {eye_node['type']}")
    
    # Compare with other disease nodes
    disease_nodes = [n for n in network_data['nodes'] if n['type'] == 'Disease']
    disease_degrees = [n['degree'] for n in disease_nodes]
    
    avg_disease_degree = sum(disease_degrees) / len(disease_degrees)
    max_disease_degree = max(disease_degrees)
    
    print(f"\n📊 DISEASE NODE COMPARISON:")
    print(f"   • Eye infection degree: {eye_node['degree']}")
    print(f"   • Average disease degree: {avg_disease_degree:.1f}")
    print(f"   • Max disease degree: {max_disease_degree}")
    print(f"   • Eye infection rank: {sorted(disease_degrees, reverse=True).index(eye_node['degree']) + 1}/{len(disease_nodes)}")
    
    if eye_node['degree'] > avg_disease_degree * 2:
        print(f"   🚨 BIAS DETECTED: Eye infection is a HIGH-DEGREE disease node!")
        print(f"   This could explain why it appears in many high-confidence predictions.")
    
    return eye_node

# MAIN FUNCTION TO RUN EYE INFECTION ANALYSIS
def run_eye_infection_analysis():
    """
    Comprehensive analysis of why eye infection dominates your predictions
    """
    
    print("👁️ COMPREHENSIVE EYE INFECTION ANALYSIS")
    print("="*70)
    print("🎯 Goal: Understand why eye infection appears so frequently")
    
    # Load your data
    components = load_essential_components(results_path, datetime)
    if not components:
        print("❌ Failed to load components")
        return None
    
    graph = components['graph']
    transformer_fps = components['transformer_fps']
    
    # Get network data
    G = to_networkx(graph, to_undirected=True)
    network_data = {
        'nodes': [
            {
                'id': node_idx,
                'name': idx_to_name.get(node_idx, f"Node_{node_idx}"),
                'type': idx_to_type.get(node_idx, "Unknown"),
                'degree': G.degree(node_idx)
            }
            for node_idx in G.nodes()
        ]
    }
    
    # Get FP pairs data (larger sample to see full picture)
    fp_pairs_data = []
    for i, (drug_name, disease_name, confidence) in enumerate(transformer_fps[:2000]):  # Increased sample
        if drug_name in approved_drugs_list_name and disease_name in disease_list_name:
            drug_idx = approved_drugs_list_name.index(drug_name)
            disease_list_pos = disease_list_name.index(disease_name)
            disease_id = disease_list[disease_list_pos]
            disease_idx = disease_key_mapping[disease_id]
            
            fp_pairs_data.append({
                'drug_name': drug_name,
                'disease_name': disease_name,
                'drug_idx': drug_idx,
                'disease_idx': disease_idx,
                'confidence': confidence,
                'pair_name': f"{drug_name} -> {disease_name}"
            })
    
    # Get paths data (you might already have this from previous analysis)
    print("🛤️ Finding paths for eye infection analysis...")
    paths_data = find_web_optimized_paths_filtered(G, fp_pairs_data[:300], max_path_length=4, max_paths_per_pair=50)
    
    # Run analysis
    eye_infection_pairs, eye_infection_idx = analyze_eye_infection_dominance(transformer_fps, paths_data, network_data)
    
    gene_usage, pathway_patterns = analyze_eye_infection_pathways(eye_infection_idx, paths_data, network_data)
    
    eye_stats, other_stats = compare_eye_infection_vs_others(transformer_fps, paths_data, 1000)
    
    eye_node = investigate_eye_infection_bias(network_data, eye_infection_idx)
    
    print(f"\n🎊 EYE INFECTION ANALYSIS COMPLETE!")
    print("="*70)
    print("🎯 KEY INSIGHTS:")
    print("1. Check if eye infection is a high-degree node (structural bias)")
    print("2. Examine if certain genes create 'shortcuts' to eye infection")
    print("3. Consider if training data had many eye infection examples")
    print("4. Look for pathway patterns that consistently lead to eye infection")
    
    return {
        'eye_infection_pairs': eye_infection_pairs,
        'gene_usage': gene_usage,
        'pathway_patterns': pathway_patterns,
        'eye_stats': eye_stats,
        'other_stats': other_stats,
        'eye_node': eye_node
    }

# SPECIFIC FUNCTION TO ANALYZE WHETHER YOU NEED ALL EYE INFECTION PATHS
def should_i_use_all_eye_infection_paths():
    """
    Determine if you should include all eye infection paths or filter them
    """
    
    print("🤔 SHOULD YOU USE ALL EYE INFECTION PATHS?")
    print("="*50)
    
    print("✅ REASONS TO KEEP ALL EYE INFECTION PATHS:")
    print("   1. They reveal WHY your model is so confident")
    print("   2. They show consistent biological pathways")
    print("   3. They might indicate real biological connections")
    print("   4. High confidence = model learned something important")
    
    print("\n⚠️ REASONS TO FILTER SOME EYE INFECTION PATHS:")
    print("   1. If it's just a high-degree node creating shortcuts")
    print("   2. If pathways are repetitive and not informative")
    print("   3. If it's overwhelming other disease analysis")
    print("   4. If you want to study diverse disease patterns")
    
    print("\n💡 RECOMMENDATIONS:")
    print("   • RUN THE ANALYSIS ABOVE FIRST")
    print("   • If eye infection degree >> average disease degree → possible bias")
    print("   • If pathways are diverse and meaningful → keep them")
    print("   • If you want to study other diseases → create separate analysis")
    
    print("\n🎯 NEXT STEPS:")
    print("   1. Run: run_eye_infection_analysis()")
    print("   2. Check eye infection degree vs other diseases")
    print("   3. Examine pathway diversity")
    print("   4. Decide based on your research goals")

# Add this to your existing script
if __name__ == "__main__":
    print("👁️ EYE INFECTION INVESTIGATION")
    print("="*40)
    
    # First, understand what you're seeing
    should_i_use_all_eye_infection_paths()
    
    print("\n" + "="*40)
    
    # Then run the full analysis
    results = run_eye_infection_analysis()

# import networkx as nx
# from collections import defaultdict, Counter
# import pandas as pd

def analyze_gene_shortcuts_for_disease(G, paths_data, network_data, target_disease_name, disease_idx=None):
    """
    Analyze if certain genes create 'shortcuts' to a specific disease
    
    Args:
        G: NetworkX graph
        paths_data: Dictionary of path data
        network_data: Network node information
        target_disease_name: Name of disease to analyze (e.g., "recurrent thrombophlebitis")
        disease_idx: Optional disease index (will auto-find if not provided)
    """
    
    print(f"🔍 ANALYZING GENE SHORTCUTS TO {target_disease_name.upper()}")
    print("="*60)
    
    # Extract all paths that lead to target disease
    disease_paths = []
    disease_drugs = set()
    
    # Search for disease in paths (case-insensitive, partial match)
    target_lower = target_disease_name.lower()
    
    for pair_key, data in paths_data.items():
        pair_disease = pair_key.split(' -> ')[-1].lower()  # Get disease part
        
        # Check if target disease is in the pair name
        if target_lower in pair_disease and data['paths_found'] > 0:
            drug_name = data['pair']['drug_name']
            disease_drugs.add(drug_name)
            
            for path in data['paths']:
                disease_paths.append({
                    'drug': drug_name,
                    'path': path,
                    'path_nodes': path['node_indices'],
                    'confidence': data['pair']['confidence']
                })
    
    print(f"📊 {target_disease_name.upper()} PATH ANALYSIS:")
    print(f"   • Total {target_disease_name} paths: {len(disease_paths)}")
    print(f"   • Unique drugs leading to {target_disease_name}: {len(disease_drugs)}")
    
    if len(disease_paths) == 0:
        print(f"❌ No paths found for {target_disease_name}")
        print("💡 Try checking:")
        print("   • Disease name spelling")
        print("   • Available diseases in your data")
        return None, None
    
    # Analyze gene usage across different drugs
    gene_usage_by_drug = defaultdict(lambda: defaultdict(int))
    gene_total_usage = defaultdict(int)
    gene_drug_count = defaultdict(set)
    
    for path_info in disease_paths:
        drug = path_info['drug']
        
        for node in path_info['path']['nodes']:
            if node['type'] == 'Gene':
                gene_name = node['name']
                gene_usage_by_drug[gene_name][drug] += 1
                gene_total_usage[gene_name] += 1
                gene_drug_count[gene_name].add(drug)
    
    print(f"\n🧬 GENE SHORTCUT ANALYSIS:")
    
    # Calculate shortcut metrics
    shortcut_genes = []
    for gene, total_usage in gene_total_usage.items():
        num_drugs = len(gene_drug_count[gene])
        usage_per_drug = total_usage / num_drugs if num_drugs > 0 else 0
        
        # Custom shortcut score
        shortcut_score = num_drugs * usage_per_drug * (total_usage / len(disease_paths))
        
        shortcut_genes.append({
            'gene': gene,
            'total_usage': total_usage,
            'num_drugs': num_drugs,
            'usage_per_drug': usage_per_drug,
            'shortcut_score': shortcut_score,
            'drug_coverage': num_drugs / len(disease_drugs) if len(disease_drugs) > 0 else 0
        })
    
    # Sort by shortcut score
    shortcut_genes.sort(key=lambda x: x['shortcut_score'], reverse=True)
    
    print(f"🚨 TOP SHORTCUT GENES (genes that many drugs use to reach {target_disease_name}):")
    print(f"{'Rank':<4} {'Gene':<20} {'Usage':<6} {'Drugs':<6} {'Coverage':<8} {'Score':<8}")
    print("-" * 60)
    
    for i, gene_info in enumerate(shortcut_genes[:15], 1):
        gene_name_short = gene_info['gene'][:19]  # Truncate long names
        coverage_pct = f"{gene_info['drug_coverage']:.1%}"  # Format percentage
        
        print(f"{i:<4} {gene_name_short:<20} "
              f"{gene_info['total_usage']:<6} "
              f"{gene_info['num_drugs']:<6} "
              f"{coverage_pct:<8} "
              f"{gene_info['shortcut_score']:.2f}")
    
    return shortcut_genes, gene_usage_by_drug

# def analyze_direct_connections_to_disease(G, disease_idx, network_data, target_disease_name):
#     """
#     Analyze direct connections to target disease node
#     """
    
#     print(f"\n🎯 DIRECT CONNECTIONS TO {target_disease_name.upper()}")
#     print("="*50)
    
#     if disease_idx is None:
#         print(f"❌ {target_disease_name} node not found in network")
#         return None
    
#     # Get direct neighbors
#     direct_neighbors = list(G.neighbors(disease_idx))
    
#     print(f"📊 {target_disease_name} has {len(direct_neighbors)} direct neighbors")
    
#     # Categorize neighbors by type
#     neighbor_types = defaultdict(list)
#     for neighbor_idx in direct_neighbors:
#         neighbor_data = next((n for n in network_data['nodes'] if n['id'] == neighbor_idx), None)
#         if neighbor_data:
#             neighbor_types[neighbor_data['type']].append({
#                 'idx': neighbor_idx,
#                 'name': neighbor_data['name'],
#                 'degree': neighbor_data['degree']
#             })
    
#     print(f"\n🔗 DIRECT NEIGHBORS BY TYPE:")
#     for node_type, neighbors in neighbor_types.items():
#         print(f"   {node_type}: {len(neighbors)} nodes")
        
#         # Show top neighbors by degree for genes
#         if node_type == 'Gene' and len(neighbors) > 0:
#             top_neighbors = sorted(neighbors, key=lambda x: x['degree'], reverse=True)[:10]
#             print(f"   Top {node_type} neighbors:")
#             for neighbor in top_neighbors:
#                 print(f"     - {neighbor['name'][:25]:25s} (degree: {neighbor['degree']})")
    
#     return neighbor_types

# def find_common_path_patterns_to_disease(paths_data, target_disease_name):
#     """
#     Find common path patterns that lead to target disease
#     """
    
#     print(f"\n🛤️ COMMON PATH PATTERNS TO {target_disease_name.upper()}")
#     print("="*50)
    
#     # Extract path patterns
#     path_patterns = defaultdict(int)
#     path_examples = defaultdict(list)
#     target_lower = target_disease_name.lower()
    
#     for pair_key, data in paths_data.items():
#         pair_disease = pair_key.split(' -> ')[-1].lower()
        
#         if target_lower in pair_disease and data['paths_found'] > 0:
#             drug_name = data['pair']['drug_name']
            
#             for path in data['paths']:
#                 # Create pattern signature
#                 pattern_nodes = []
#                 for node in path['nodes']:
#                     if node['type'] in ['Gene', 'DrugType', 'TherapeuticArea']:
#                         pattern_nodes.append(f"{node['type']}:{node['name']}")
                
#                 if len(pattern_nodes) >= 2:
#                     pattern = ' -> '.join(pattern_nodes)
#                     path_patterns[pattern] += 1
#                     path_examples[pattern].append(drug_name)
    
#     print(f"🔍 MOST COMMON PATH PATTERNS:")
#     top_patterns = sorted(path_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
    
#     for i, (pattern, count) in enumerate(top_patterns, 1):
#         print(f"\n{i:2d}. Pattern used {count} times:")
#         print(f"    {pattern}")
        
#         # Show example drugs using this pattern
#         example_drugs = list(set(path_examples[pattern]))[:5]
#         print(f"    Example drugs: {', '.join(example_drugs)}")
#         if len(path_examples[pattern]) > 5:
#             print(f"    ... and {len(path_examples[pattern]) - 5} more")
    
#     return path_patterns, path_examples

# def detect_hub_genes_for_disease(shortcut_genes, G, disease_idx, target_disease_name):
#     """
#     Detect if shortcut genes are actually hub genes in the network
#     """
    
#     print(f"\n🌟 HUB GENE ANALYSIS FOR {target_disease_name.upper()} SHORTCUTS")
#     print("="*60)
    
#     if not shortcut_genes:
#         print("❌ No shortcut genes to analyze")
#         return [], []
    
#     # Get top shortcut genes
#     top_shortcuts = shortcut_genes[:10]
    
#     print(f"🔍 ANALYZING TOP {len(top_shortcuts)} SHORTCUT GENES:")
#     print(f"{'Gene':<20} {'Shortcut Score':<15} {'Network Degree':<15} {'Distance to Disease':<15}")
#     print("-" * 70)
    
#     hub_analysis = []
    
#     for gene_info in top_shortcuts:
#         gene_name = gene_info['gene']
        
#         # Find gene node in network
#         gene_node = None
#         for node_idx in G.nodes():
#             if idx_to_name.get(node_idx, '') == gene_name:
#                 gene_node = node_idx
#                 break
        
#         if gene_node is not None:
#             gene_degree = G.degree(gene_node)
            
#             # Calculate shortest path to disease
#             try:
#                 if disease_idx:
#                     distance_to_disease = nx.shortest_path_length(G, gene_node, disease_idx)
#                 else:
#                     distance_to_disease = "Unknown"
#             except:
#                 distance_to_disease = "No path"
            
#             hub_analysis.append({
#                 'gene': gene_name,
#                 'shortcut_score': gene_info['shortcut_score'],
#                 'network_degree': gene_degree,
#                 'distance_to_disease': distance_to_disease
#             })
            
#             gene_name_short = gene_name[:19]  # Truncate for display
#             print(f"{gene_name_short:<20} "
#                   f"{gene_info['shortcut_score']:<15.2f} "
#                   f"{gene_degree:<15} "
#                   f"{distance_to_disease}")
    
#     # Identify true shortcuts vs network hubs
#     print(f"\n🎯 SHORTCUT CLASSIFICATION:")
    
#     true_shortcuts = []
#     network_hubs = []
    
#     for analysis in hub_analysis:
#         if isinstance(analysis['distance_to_disease'], int) and analysis['distance_to_disease'] <= 3:
#             if analysis['network_degree'] > 500:  # High degree threshold
#                 network_hubs.append(analysis)
#                 print(f"🌟 HUB: {analysis['gene']} (degree: {analysis['network_degree']}, distance: {analysis['distance_to_disease']})")
#             else:
#                 true_shortcuts.append(analysis)
#                 print(f"🎯 SHORTCUT: {analysis['gene']} (degree: {analysis['network_degree']}, distance: {analysis['distance_to_disease']})")
    
#     print(f"\n📊 SUMMARY:")
#     print(f"   • True shortcuts (low degree, close to disease): {len(true_shortcuts)}")
#     print(f"   • Network hubs (high degree, acting as shortcuts): {len(network_hubs)}")
    
#     if len(network_hubs) > len(true_shortcuts):
#         print(f"   🚨 FINDING: {target_disease_name} confidence likely due to NETWORK HUBS!")
#         print(f"   → Model uses high-degree genes as easy paths to {target_disease_name}")
#     else:
#         print(f"   🎯 FINDING: {target_disease_name} confidence due to SPECIFIC GENE SHORTCUTS!")
#         print(f"   → Model learned biologically meaningful connections")
    
#     return true_shortcuts, network_hubs

# def find_disease_index(target_disease_name):
#     """
#     Find the disease index in your mappings
#     """
#     target_lower = target_disease_name.lower()
    
#     for i, disease_name in enumerate(disease_list_name):
#         if target_lower in disease_name.lower():
#             disease_id = disease_list[i]
#             disease_idx = disease_key_mapping[disease_id]
#             print(f"   • Found in network: '{disease_name}' (ID: {disease_id}, idx: {disease_idx})")
#             return disease_idx, disease_name
    
#     print(f"   ❌ Disease '{target_disease_name}' not found in network")
#     return None, None

# def run_complete_gene_shortcuts_analysis_for_disease(target_disease_name):
#     """
#     Run complete analysis of gene shortcuts for any disease
    
#     Args:
#         target_disease_name: Name of disease to analyze (e.g., "recurrent thrombophlebitis")
#     """
    
#     print(f"🧬 COMPREHENSIVE GENE SHORTCUTS ANALYSIS FOR {target_disease_name.upper()}")
#     print("="*70)
#     print(f"🎯 Goal: Understand if genes create shortcuts to {target_disease_name}")
    
#     # Load your data
#     components = load_essential_components(results_path, datetime)
#     if not components:
#         print("❌ Failed to load components")
#         return None
    
#     graph = components['graph']
#     transformer_fps = components['transformer_fps']
    
#     # Get network data
#     G = to_networkx(graph, to_undirected=True)
#     network_data = {
#         'nodes': [
#             {
#                 'id': node_idx,
#                 'name': idx_to_name.get(node_idx, f"Node_{node_idx}"),
#                 'type': idx_to_type.get(node_idx, "Unknown"),
#                 'degree': G.degree(node_idx)
#             }
#             for node_idx in G.nodes()
#         ]
#     }
    
#     # Find target disease node
#     disease_idx, found_disease_name = find_disease_index(target_disease_name)
    
#     # Get FP pairs data
#     fp_pairs_data = []
#     for i, (drug_name, disease_name, confidence) in enumerate(transformer_fps[:500]):
#         if drug_name in approved_drugs_list_name and disease_name in disease_list_name:
#             drug_idx = approved_drugs_list_name.index(drug_name)
#             disease_list_pos = disease_list_name.index(disease_name)
#             disease_id = disease_list[disease_list_pos]
#             disease_idx_temp = disease_key_mapping[disease_id]
            
#             fp_pairs_data.append({
#                 'drug_name': drug_name,
#                 'disease_name': disease_name,
#                 'drug_idx': drug_idx,
#                 'disease_idx': disease_idx_temp,
#                 'confidence': confidence,
#                 'pair_name': f"{drug_name} -> {disease_name}"
#             })
    
#     # Get paths
#     print("🛤️ Finding paths for shortcut analysis...")
#     paths_data = find_web_optimized_paths_filtered(G, fp_pairs_data[:200], max_path_length=4, max_paths_per_pair=100)
    
#     # Run shortcut analysis
#     shortcut_genes, gene_usage_by_drug = analyze_gene_shortcuts_for_disease(
#         G, paths_data, network_data, target_disease_name, disease_idx
#     )
    
#     if shortcut_genes is None:
#         print(f"❌ Analysis failed for {target_disease_name}")
#         return None
    
#     neighbor_types = analyze_direct_connections_to_disease(G, disease_idx, network_data, target_disease_name)
    
#     path_patterns, path_examples = find_common_path_patterns_to_disease(paths_data, target_disease_name)
    
#     true_shortcuts, network_hubs = detect_hub_genes_for_disease(shortcut_genes, G, disease_idx, target_disease_name)
    
#     print(f"\n🎊 GENE SHORTCUTS ANALYSIS COMPLETE FOR {target_disease_name.upper()}!")
#     print("="*70)
    
#     # Summary insights
#     print(f"\n🎯 KEY FINDINGS:")
#     if len(network_hubs) > len(true_shortcuts):
#         print(f"1. 🚨 NETWORK HUB BIAS: High-degree genes create shortcuts to {target_disease_name}")
#         print(f"   → Model exploits structural network properties")
#         print(f"   → High confidence may be due to network topology, not biology")
#     else:
#         print(f"1. 🎯 BIOLOGICAL SHORTCUTS: Specific genes create meaningful paths to {target_disease_name}")
#         print(f"   → Model learned genuine biological connections")
#         print(f"   → High confidence reflects real drug-disease relationships")
    
#     if shortcut_genes:
#         print(f"2. Top shortcut gene: {shortcut_genes[0]['gene']} (score: {shortcut_genes[0]['shortcut_score']:.2f})")
#         print(f"3. Number of drugs using shortcuts: {len(set().union(*[list(drug_dict.keys()) for drug_dict in gene_usage_by_drug.values()]))}")
#         print(f"4. Most common path pattern shows biological mechanism for {target_disease_name}")
    
#     return {
#         'disease_name': target_disease_name,
#         'shortcut_genes': shortcut_genes,
#         'true_shortcuts': true_shortcuts,
#         'network_hubs': network_hubs,
#         'path_patterns': path_patterns,
#         'neighbor_types': neighbor_types
#     }

# # USAGE EXAMPLES
# if __name__ == "__main__":
#     print("🧬 GENE SHORTCUTS ANALYSIS FOR ANY DISEASE")
#     print("="*60)
    
#     # Analyze recurrent thrombophlebitis
#     results_thrombophlebitis = run_complete_gene_shortcuts_analysis_for_disease("eye infection")
    
#     # You can also try other diseases:
#     # results_diabetes = run_complete_gene_shortcuts_analysis_for_disease("diabetes")
#     # results_hypertension = run_complete_gene_shortcuts_analysis_for_disease("hypertension")
#     # results_asthma = run_complete_gene_shortcuts_analysis_for_disease("asthma")

import networkx as nx
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

def analyze_network_degree_distribution(network_data):
    """
    Analyze the degree distribution of your network
    """
    degrees = [node['degree'] for node in network_data['nodes']]
    
    print("📊 NETWORK DEGREE DISTRIBUTION ANALYSIS:")
    print("="*50)
    print(f"   • Total nodes: {len(degrees):,}")
    print(f"   • Mean degree: {np.mean(degrees):.1f}")
    print(f"   • Median degree: {np.median(degrees):.1f}")
    print(f"   • Standard deviation: {np.std(degrees):.1f}")
    print(f"   • Min degree: {min(degrees)}")
    print(f"   • Max degree: {max(degrees):,}")
    
    # Calculate percentiles
    percentiles = {
        50: np.percentile(degrees, 50),
        75: np.percentile(degrees, 75),
        90: np.percentile(degrees, 90),
        95: np.percentile(degrees, 95),
        99: np.percentile(degrees, 99)
    }
    
    print(f"\n📈 DEGREE PERCENTILES:")
    for p, value in percentiles.items():
        count_above = sum(1 for d in degrees if d >= value)
        print(f"   • {p}th percentile: {value:.0f} (≥{count_above:,} nodes)")
    
    # Specific counts for your thresholds
    print(f"\n🎯 THRESHOLD ANALYSIS:")
    thresholds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for threshold in thresholds:
        count = sum(1 for d in degrees if d >= threshold)
        percentage = (count / len(degrees)) * 100
        print(f"   • Degree ≥{threshold}: {count:,} nodes ({percentage:.1f}%)")
    
    return {
        'degrees': degrees,
        'percentiles': percentiles,
        'stats': {
            'mean': np.mean(degrees),
            'median': np.median(degrees),
            'std': np.std(degrees),
            'min': min(degrees),
            'max': max(degrees)
        }
    }

def classify_gene_by_network_percentiles(gene_degree, degree_stats):
    """
    Classify genes based on network degree percentiles
    """
    percentiles = degree_stats['percentiles']
    
    if gene_degree >= percentiles[99]:
        return "🔥 TOP 1%", "red"
    elif gene_degree >= percentiles[95]:
        return "🌟 TOP 5%", "orange"
    elif gene_degree >= percentiles[90]:
        return "🔸 TOP 10%", "yellow"
    elif gene_degree >= percentiles[75]:
        return "⬆️ TOP 25%", "lightblue"
    else:
        return "📊 BELOW 75%", "gray"

def detect_hub_genes_with_network_analysis(shortcut_genes, G, disease_idx, target_disease_name, network_data, top_n=50):
    """
    Enhanced hub gene analysis with network degree distribution context
    
    Args:
        top_n: Number of top shortcut genes to analyze (default: 50)
    """
    
    print(f"\n🌟 ENHANCED HUB GENE ANALYSIS FOR {target_disease_name.upper()}")
    print("="*70)
    
    # First, analyze the network degree distribution
    degree_stats = analyze_network_degree_distribution(network_data)
    
    if not shortcut_genes:
        print("❌ No shortcut genes to analyze")
        return [], [], degree_stats
    
    # Get top shortcut genes
    top_shortcuts = shortcut_genes[:top_n]
    
    print(f"\n🔍 ANALYZING TOP {len(top_shortcuts)} SHORTCUT GENES WITH NETWORK CONTEXT:")
    print(f"{'Gene':<15} {'Score':<8} {'Degree':<8} {'Percentile':<12} {'Distance':<8} {'Classification'}")
    print("-" * 80)
    
    hub_analysis = []
    degree_classification_counts = defaultdict(int)
    
    for gene_info in top_shortcuts:
        gene_name = gene_info['gene']
        
        # Find gene node in network
        gene_node = None
        for node_idx in G.nodes():
            if idx_to_name.get(node_idx, '') == gene_name:
                gene_node = node_idx
                break
        
        if gene_node is not None:
            gene_degree = G.degree(gene_node)
            
            # Calculate shortest path to disease
            try:
                if disease_idx:
                    distance_to_disease = nx.shortest_path_length(G, gene_node, disease_idx)
                else:
                    distance_to_disease = "Unknown"
            except:
                distance_to_disease = "No path"
            
            # Classify by network percentiles
            percentile_class, color = classify_gene_by_network_percentiles(gene_degree, degree_stats)
            degree_classification_counts[percentile_class] += 1
            
            hub_analysis.append({
                'gene': gene_name,
                'shortcut_score': gene_info['shortcut_score'],
                'network_degree': gene_degree,
                'distance_to_disease': distance_to_disease,
                'percentile_class': percentile_class,
                'color': color
            })
            
            # Print formatted results
            gene_short = gene_name[:14]
            print(f"{gene_short:<15} "
                  f"{gene_info['shortcut_score']:<8.1f} "
                  f"{gene_degree:<8} "
                  f"{percentile_class:<12} "
                  f"{distance_to_disease:<8} "
                  f"Score rank #{len(hub_analysis)}")
    
    # Summary of degree classifications
    print(f"\n📊 DEGREE DISTRIBUTION SUMMARY FOR TOP {top_n} SHORTCUT GENES:")
    print("-" * 50)
    for class_name, count in degree_classification_counts.items():
        percentage = (count / len(top_shortcuts)) * 100
        print(f"   • {class_name}: {count}/{len(top_shortcuts)} genes ({percentage:.1f}%)")
    
    # Enhanced biological vs topological analysis
    print(f"\n🎯 ENHANCED SHORTCUT CLASSIFICATION (TOP {top_n}):")
    print("-" * 50)
    
    top_5_percent_genes = []
    top_10_percent_genes = []
    moderate_degree_genes = []
    
    for analysis in hub_analysis:
        gene_name = analysis['gene']
        degree = analysis['network_degree']
        score = analysis['shortcut_score']
        percentile = analysis['percentile_class']
        
        if "TOP 1%" in percentile or "TOP 5%" in percentile:
            top_5_percent_genes.append(analysis)
            print(f"🔥 HIGH-DEGREE HUB: {gene_name} ({percentile}, degree: {degree}, score: {score:.1f})")
        elif "TOP 10%" in percentile:
            top_10_percent_genes.append(analysis)
            print(f"🌟 MODERATE HUB: {gene_name} ({percentile}, degree: {degree}, score: {score:.1f})")
        else:
            moderate_degree_genes.append(analysis)
            print(f"🎯 BIOLOGICAL SHORTCUT: {gene_name} ({percentile}, degree: {degree}, score: {score:.1f})")
    
    # Final interpretation
    print(f"\n🎊 INTERPRETATION SUMMARY:")
    print("="*50)
    print(f"   • Top 5% degree genes: {len(top_5_percent_genes)}/{top_n} ({len(top_5_percent_genes)/top_n*100:.1f}%)")
    print(f"   • Top 10% degree genes: {len(top_10_percent_genes)}/{top_n} ({len(top_10_percent_genes)/top_n*100:.1f}%)")
    print(f"   • Moderate degree genes: {len(moderate_degree_genes)}/{top_n} ({len(moderate_degree_genes)/top_n*100:.1f}%)")
    
    if len(moderate_degree_genes) >= len(top_5_percent_genes):
        print(f"\n🎯 FINDING: {target_disease_name} shortcuts are primarily BIOLOGICAL targets!")
        print(f"   → {len(moderate_degree_genes)} moderate-degree vs {len(top_5_percent_genes)} high-degree genes")
        print(f"   → Model learned specific biological pathways, not network topology")
    else:
        print(f"\n🌐 FINDING: {target_disease_name} shortcuts may have topological bias!")
        print(f"   → {len(top_5_percent_genes)} high-degree vs {len(moderate_degree_genes)} moderate-degree genes")
        print(f"   → Model may be exploiting network connectivity")
    
    # Additional analysis for larger sample
    if top_n >= 20:
        print(f"\n📈 DETAILED BREAKDOWN:")
        print(f"   • TOP 1% genes: {len([g for g in hub_analysis if 'TOP 1%' in g['percentile_class']])}")
        print(f"   • TOP 5% genes: {len([g for g in hub_analysis if 'TOP 5%' in g['percentile_class']])}")
        print(f"   • TOP 10% genes: {len([g for g in hub_analysis if 'TOP 10%' in g['percentile_class']])}")
        print(f"   • TOP 25% genes: {len([g for g in hub_analysis if 'TOP 25%' in g['percentile_class']])}")
        print(f"   • Below 75% genes: {len([g for g in hub_analysis if 'BELOW 75%' in g['percentile_class']])}")
        
        # Show where the pattern changes
        score_threshold_analysis = []
        for i, analysis in enumerate(hub_analysis[:top_n], 1):
            if 'BELOW 75%' in analysis['percentile_class'] or 'TOP 25%' in analysis['percentile_class']:
                score_threshold_analysis.append((i, analysis['gene'], analysis['shortcut_score'], analysis['percentile_class']))
        
        if score_threshold_analysis:
            print(f"\n🔍 FIRST MODERATE-DEGREE GENES:")
            for rank, gene, score, percentile in score_threshold_analysis[:5]:
                print(f"   • Rank #{rank}: {gene} (score: {score:.1f}, {percentile})")
    
    return top_5_percent_genes, top_10_percent_genes, moderate_degree_genes, degree_stats

def run_enhanced_gene_shortcuts_analysis_for_disease(target_disease_name):
    """
    Run enhanced analysis with network degree distribution context
    """
    
    print(f"🧬 ENHANCED GENE SHORTCUTS ANALYSIS FOR {target_disease_name.upper()}")
    print("="*70)
    print(f"🎯 Goal: Understand shortcuts with network degree context")
    
    # Load your data
    components = load_essential_components(results_path, datetime)
    if not components:
        print("❌ Failed to load components")
        return None
    
    graph = components['graph']
    transformer_fps = components['transformer_fps']
    
    # Get network data
    G = to_networkx(graph, to_undirected=True)
    network_data = {
        'nodes': [
            {
                'id': node_idx,
                'name': idx_to_name.get(node_idx, f"Node_{node_idx}"),
                'type': idx_to_type.get(node_idx, "Unknown"),
                'degree': G.degree(node_idx)
            }
            for node_idx in G.nodes()
        ]
    }
    
    # Find target disease node
    disease_idx, found_disease_name = find_disease_index(target_disease_name)
    
    # Get FP pairs data
    fp_pairs_data = []
    for i, (drug_name, disease_name, confidence) in enumerate(transformer_fps[:100000]):
        if drug_name in approved_drugs_list_name and disease_name in disease_list_name:
            drug_idx = approved_drugs_list_name.index(drug_name)
            disease_list_pos = disease_list_name.index(disease_name)
            disease_id = disease_list[disease_list_pos]
            disease_idx_temp = disease_key_mapping[disease_id]
            
            fp_pairs_data.append({
                'drug_name': drug_name,
                'disease_name': disease_name,
                'drug_idx': drug_idx,
                'disease_idx': disease_idx_temp,
                'confidence': confidence,
                'pair_name': f"{drug_name} -> {disease_name}"
            })
    
    # Get paths
    print("🛤️ Finding paths for enhanced analysis...")
    paths_data = find_web_optimized_paths_filtered(G, fp_pairs_data[:], max_path_length=4, max_paths_per_pair=100)
    
    # Run shortcut analysis (using existing function)
    shortcut_genes, gene_usage_by_drug = analyze_gene_shortcuts_for_disease(
        G, paths_data, network_data, target_disease_name, disease_idx
    )
    
    if shortcut_genes is None:
        print(f"❌ Analysis failed for {target_disease_name}")
        return None
    
    # Enhanced hub analysis with network context
    top_5_percent_genes, top_10_percent_genes, moderate_degree_genes, degree_stats = detect_hub_genes_with_network_analysis(
        shortcut_genes, G, disease_idx, target_disease_name, network_data, top_n=50  # Analyze top 50 genes
    )
    
    print(f"\n🎊 ENHANCED ANALYSIS COMPLETE FOR {target_disease_name.upper()}!")
    print("="*70)
    
    return {
        'disease_name': target_disease_name,
        'shortcut_genes': shortcut_genes,
        'top_5_percent_genes': top_5_percent_genes,
        'top_10_percent_genes': top_10_percent_genes,
        'moderate_degree_genes': moderate_degree_genes,
        'degree_stats': degree_stats,
        'network_data': network_data
    }

def find_disease_index(target_disease_name):
    """
    Find the disease index in your mappings
    """
    target_lower = target_disease_name.lower()
    
    for i, disease_name in enumerate(disease_list_name):
        if target_lower in disease_name.lower():
            disease_id = disease_list[i]
            disease_idx = disease_key_mapping[disease_id]
            print(f"   • Found in network: '{disease_name}' (ID: {disease_id}, idx: {disease_idx})")
            return disease_idx, disease_name
    
    print(f"   ❌ Disease '{target_disease_name}' not found in network")
    return None, None

# # USAGE EXAMPLE
# if __name__ == "__main__":
#     print("🧬 ENHANCED GENE SHORTCUTS ANALYSIS WITH NETWORK DEGREE CONTEXT")
#     print("="*70)
    
#     # Run enhanced analysis for eye infection
#     eye_results = run_enhanced_gene_shortcuts_analysis_for_disease("iridocyclitis")
    
#     # You can also run for thrombophlebitis
#     # thrombophlebitis_results = run_enhanced_gene_shortcuts_analysis_for_disease("recurrent thrombophlebitis")

def discover_all_diseases_in_fp_predictions(transformer_fps, sample_size=None):
    """
    Discover all diseases that appear in your FP predictions
    
    Args:
        transformer_fps: Your FP predictions
        sample_size: How many predictions to analyze (None = all)
    """
    
    print("🔍 DISCOVERING ALL DISEASES IN FP PREDICTIONS")
    print("="*60)
    
    # Use all predictions if sample_size not specified
    if sample_size is None:
        fps_to_analyze = transformer_fps
        print(f"📊 Analyzing ALL {len(transformer_fps):,} FP predictions")
    else:
        fps_to_analyze = transformer_fps[:sample_size]
        print(f"📊 Analyzing top {sample_size:,} FP predictions")
    
    # Collect all diseases with their statistics
    disease_stats = defaultdict(lambda: {
        'count': 0,
        'max_confidence': 0,
        'avg_confidence': 0,
        'confidences': [],
        'drugs': set()
    })
    
    for drug_name, disease_name, confidence in fps_to_analyze:
        disease_stats[disease_name]['count'] += 1
        disease_stats[disease_name]['max_confidence'] = max(disease_stats[disease_name]['max_confidence'], confidence)
        disease_stats[disease_name]['confidences'].append(confidence)
        disease_stats[disease_name]['drugs'].add(drug_name)
    
    # Calculate averages
    for disease, stats in disease_stats.items():
        stats['avg_confidence'] = sum(stats['confidences']) / len(stats['confidences'])
        stats['num_drugs'] = len(stats['drugs'])
    
    # Sort by count (most frequent first)
    sorted_diseases = sorted(disease_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    print(f"\n📋 DISCOVERED {len(sorted_diseases)} UNIQUE DISEASES:")
    print(f"{'Rank':<4} {'Disease':<40} {'Count':<8} {'Drugs':<6} {'Max Conf':<8} {'Avg Conf':<8}")
    print("-" * 80)
    
    # Show top diseases
    for i, (disease, stats) in enumerate(sorted_diseases[:50], 1):
        disease_short = disease[:39] if len(disease) > 39 else disease
        print(f"{i:<4} {disease_short:<40} "
              f"{stats['count']:<8} "
              f"{stats['num_drugs']:<6} "
              f"{stats['max_confidence']:<8.3f} "
              f"{stats['avg_confidence']:<8.3f}")
    
    if len(sorted_diseases) > 50:
        print(f"... and {len(sorted_diseases) - 50} more diseases")
    
    return sorted_diseases, disease_stats

def check_diseases_in_network_mappings(sorted_diseases, disease_stats, top_n=100):
    """
    Check which diseases from FP predictions exist in your network mappings
    """
    
    # Access global variables
    global disease_list_name
    
    print(f"\n🔗 CHECKING NETWORK MAPPING COVERAGE")
    print("="*50)
    
    # Check which diseases are mappable
    mappable_diseases = []
    unmappable_diseases = []
    
    for disease_name, stats in sorted_diseases[:top_n]:
        # Check if disease exists in your network
        found_in_network = False
        mapped_disease_name = None
        
        for i, network_disease_name in enumerate(disease_list_name):
            if disease_name.lower() == network_disease_name.lower():
                found_in_network = True
                mapped_disease_name = network_disease_name
                break
        
        if found_in_network:
            mappable_diseases.append((disease_name, stats, mapped_disease_name))
        else:
            unmappable_diseases.append((disease_name, stats))
    
    print(f"📊 MAPPING RESULTS FOR TOP {top_n} DISEASES:")
    print(f"   • Mappable to network: {len(mappable_diseases)}")
    print(f"   • Not mappable: {len(unmappable_diseases)}")
    print(f"   • Coverage: {len(mappable_diseases)/top_n*100:.1f}%")
    
    print(f"\n✅ MAPPABLE DISEASES (can be analyzed):")
    print(f"{'Rank':<4} {'Disease':<35} {'Count':<8} {'Drugs':<6} {'Avg Conf':<8}")
    print("-" * 65)
    
    for i, (disease_name, stats, mapped_name) in enumerate(mappable_diseases[:30], 1):
        disease_short = disease_name[:34] if len(disease_name) > 34 else disease_name
        print(f"{i:<4} {disease_short:<35} "
              f"{stats['count']:<8} "
              f"{stats['num_drugs']:<6} "
              f"{stats['avg_confidence']:<8.3f}")
    
    if len(unmappable_diseases) > 0:
        print(f"\n❌ TOP UNMAPPABLE DISEASES:")
        for i, (disease_name, stats) in enumerate(unmappable_diseases[:10], 1):
            print(f"   {i}. {disease_name} ({stats['count']} predictions)")
    
    return mappable_diseases, unmappable_diseases

def analyze_gene_shortcuts_for_disease_enhanced(G, paths_data, network_data, target_disease_name, disease_idx=None, degree_stats=None):
    """
    Enhanced version with network degree distribution context
    """
    
    print(f"🔍 ANALYZING GENE SHORTCUTS TO {target_disease_name.upper()}")
    print("="*60)
    
    # Extract all paths that lead to target disease
    disease_paths = []
    disease_drugs = set()
    
    # Search for disease in paths (case-insensitive, partial match)
    target_lower = target_disease_name.lower()
    
    for pair_key, data in paths_data.items():
        pair_disease = pair_key.split(' -> ')[-1].lower()  # Get disease part
        
        # Check if target disease is in the pair name
        if target_lower in pair_disease and data['paths_found'] > 0:
            drug_name = data['pair']['drug_name']
            disease_drugs.add(drug_name)
            
            for path in data['paths']:
                disease_paths.append({
                    'drug': drug_name,
                    'path': path,
                    'path_nodes': path['node_indices'],
                    'confidence': data['pair']['confidence']
                })
    
    print(f"📊 {target_disease_name.upper()} PATH ANALYSIS:")
    print(f"   • Total {target_disease_name} paths: {len(disease_paths)}")
    print(f"   • Unique drugs leading to {target_disease_name}: {len(disease_drugs)}")
    
    if len(disease_paths) == 0:
        print(f"❌ No paths found for {target_disease_name}")
        return None, None
    
    # Analyze gene usage across different drugs
    gene_usage_by_drug = defaultdict(lambda: defaultdict(int))
    gene_total_usage = defaultdict(int)
    gene_drug_count = defaultdict(set)
    
    for path_info in disease_paths:
        drug = path_info['drug']
        
        for node in path_info['path']['nodes']:
            if node['type'] == 'Gene':
                gene_name = node['name']
                gene_usage_by_drug[gene_name][drug] += 1
                gene_total_usage[gene_name] += 1
                gene_drug_count[gene_name].add(drug)
    
    # Calculate shortcut metrics
    shortcut_genes = []
    for gene, total_usage in gene_total_usage.items():
        num_drugs = len(gene_drug_count[gene])
        usage_per_drug = total_usage / num_drugs if num_drugs > 0 else 0
        
        # Custom shortcut score
        shortcut_score = num_drugs * usage_per_drug * (total_usage / len(disease_paths))
        
        # Find gene node in network for degree info
        gene_node = None
        gene_degree = 0
        distance_to_disease = "Unknown"
        
        # Access global variables
        global idx_to_name
        
        for node_idx in G.nodes():
            if idx_to_name.get(node_idx, '') == gene:
                gene_node = node_idx
                gene_degree = G.degree(gene_node)
                
                # Calculate distance to disease
                try:
                    if disease_idx:
                        distance_to_disease = nx.shortest_path_length(G, gene_node, disease_idx)
                except:
                    distance_to_disease = "No path"
                break
        
        # Classify by network percentiles if available
        percentile_class = "Unknown"
        if degree_stats and gene_degree > 0:
            percentiles = degree_stats['percentiles']
            if gene_degree >= percentiles[99]:
                percentile_class = "🔥 TOP 1%"
            elif gene_degree >= percentiles[95]:
                percentile_class = "🌟 TOP 5%"
            elif gene_degree >= percentiles[90]:
                percentile_class = "🔸 TOP 10%"
            elif gene_degree >= percentiles[75]:
                percentile_class = "⬆️ TOP 25%"
            else:
                percentile_class = "📊 BELOW 75%"
        
        shortcut_genes.append({
            'gene': gene,
            'total_usage': total_usage,
            'num_drugs': num_drugs,
            'usage_per_drug': usage_per_drug,
            'shortcut_score': shortcut_score,
            'drug_coverage': num_drugs / len(disease_drugs) if len(disease_drugs) > 0 else 0,
            'degree': gene_degree,
            'distance_to_disease': distance_to_disease,
            'percentile_class': percentile_class
        })
    
    # Sort by shortcut score
    shortcut_genes.sort(key=lambda x: x['shortcut_score'], reverse=True)
    
    print(f"\n🧬 TOP SHORTCUT GENES:")
    print(f"{'Gene':<15} {'Score':<8} {'Degree':<8} {'Percentile':<12} {'Distance':<8} {'Classification'}")
    print("-" * 80)
    
    for i, gene_info in enumerate(shortcut_genes[:25], 1):
        gene_short = gene_info['gene'][:14]
        print(f"{gene_short:<15} "
              f"{gene_info['shortcut_score']:<8.1f} "
              f"{gene_info['degree']:<8} "
              f"{gene_info['percentile_class']:<12} "
              f"{gene_info['distance_to_disease']:<8} "
              f"Score rank #{i}")
    
    return shortcut_genes, gene_usage_by_drug

def analyze_network_degree_distribution(network_data):
    """
    Analyze the degree distribution of your network
    """
    degrees = [node['degree'] for node in network_data['nodes']]
    
    # Calculate percentiles
    percentiles = {
        50: np.percentile(degrees, 50),
        75: np.percentile(degrees, 75),
        90: np.percentile(degrees, 90),
        95: np.percentile(degrees, 95),
        99: np.percentile(degrees, 99)
    }
    
    return {
        'degrees': degrees,
        'percentiles': percentiles,
        'stats': {
            'mean': np.mean(degrees),
            'median': np.median(degrees),
            'std': np.std(degrees),
            'min': min(degrees),
            'max': max(degrees)
        }
    }

def find_disease_index(target_disease_name):
    """
    Find the disease index in your mappings
    """
    # Access global variables
    global disease_list_name, disease_list, disease_key_mapping
    
    target_lower = target_disease_name.lower()
    
    for i, disease_name in enumerate(disease_list_name):
        if target_lower in disease_name.lower():
            disease_id = disease_list[i]
            disease_idx = disease_key_mapping[disease_id]
            return disease_idx, disease_name
    
    return None, None

def run_top_10_diseases_analysis(sample_size=10000, top_diseases_count=10):
    """
    Analyze gene shortcuts for the top 10 diseases from FP predictions
    
    Args:
        sample_size: Number of FP predictions to analyze (10000 recommended)
        top_diseases_count: Number of top diseases to analyze (10 recommended)
    """
    
    # Access global variables
    global approved_drugs_list_name, disease_list_name, disease_list, disease_key_mapping
    global idx_to_name, idx_to_type
    
    print("🧬 TOP 10 DISEASES GENE SHORTCUTS ANALYSIS")
    print("="*70)
    print(f"🎯 Analyzing top {top_diseases_count} diseases from {sample_size:,} FP predictions")
    
    # Load your data
    components = load_essential_components(results_path, datetime)
    if not components:
        print("❌ Failed to load components")
        return None
    
    graph = components['graph']
    transformer_fps = components['transformer_fps']
    
    # Get network data
    G = to_networkx(graph, to_undirected=True)
    network_data = {
        'nodes': [
            {
                'id': node_idx,
                'name': idx_to_name.get(node_idx, f"Node_{node_idx}"),
                'type': idx_to_type.get(node_idx, "Unknown"),
                'degree': G.degree(node_idx)
            }
            for node_idx in G.nodes()
        ]
    }
    
    # Analyze network degree distribution once
    print("📊 Analyzing network degree distribution...")
    degree_stats = analyze_network_degree_distribution(network_data)
    
    # Step 1: Discover all diseases from FP predictions
    sorted_diseases, disease_stats = discover_all_diseases_in_fp_predictions(
        transformer_fps, sample_size=sample_size
    )
    
    # Step 2: Check network mappings
    mappable_diseases, unmappable_diseases = check_diseases_in_network_mappings(
        sorted_diseases, disease_stats, top_n=50
    )
    
    # Step 3: Get top mappable diseases
    top_mappable_diseases = mappable_diseases[:top_diseases_count]
    
    print(f"\n🎯 ANALYZING TOP {len(top_mappable_diseases)} MAPPABLE DISEASES:")
    print("="*70)
    
    # Prepare FP pairs data for path analysis
    fp_pairs_data = []
    for i, (drug_name, disease_name, confidence) in enumerate(transformer_fps[:sample_size]):
        if drug_name in approved_drugs_list_name and disease_name in disease_list_name:
            drug_idx = approved_drugs_list_name.index(drug_name)
            disease_list_pos = disease_list_name.index(disease_name)
            disease_id = disease_list[disease_list_pos]
            disease_idx = disease_key_mapping[disease_id]
            
            fp_pairs_data.append({
                'drug_name': drug_name,
                'disease_name': disease_name,
                'drug_idx': drug_idx,
                'disease_idx': disease_idx,
                'confidence': confidence,
                'pair_name': f"{drug_name} -> {disease_name}"
            })
    
    # Find paths for analysis
    print("🛤️ Finding paths for gene shortcuts analysis...")
    paths_data = find_web_optimized_paths_filtered(G, fp_pairs_data, max_path_length=4, max_paths_per_pair=100)
    
    # Analyze each disease
    all_disease_results = {}
    
    for i, (disease_name, stats, mapped_name) in enumerate(top_mappable_diseases, 1):
        print(f"\n" + "="*80)
        print(f"DISEASE {i}/{len(top_mappable_diseases)}: {disease_name.upper()}")
        print("="*80)
        print(f"📊 Predictions: {stats['count']}, Drugs: {stats['num_drugs']}, Avg Confidence: {stats['avg_confidence']:.3f}")
        
        # Find disease index
        disease_idx, found_disease_name = find_disease_index(disease_name)
        
        if disease_idx is None:
            print(f"❌ Disease '{disease_name}' not found in network mappings")
            continue
        
        try:
            # Run gene shortcuts analysis
            shortcut_genes, gene_usage_by_drug = analyze_gene_shortcuts_for_disease_enhanced(
                G, paths_data, network_data, disease_name, disease_idx, degree_stats
            )
            
            if shortcut_genes:
                all_disease_results[disease_name] = {
                    'shortcut_genes': shortcut_genes,
                    'gene_usage_by_drug': gene_usage_by_drug,
                    'stats': stats,
                    'disease_idx': disease_idx,
                    'found_disease_name': found_disease_name
                }
                print(f"✅ Analysis completed for {disease_name}")
            else:
                print(f"❌ No gene shortcuts found for {disease_name}")
                
        except Exception as e:
            print(f"❌ Error analyzing {disease_name}: {e}")
    
    # Summary results
    print(f"\n🎊 TOP {top_diseases_count} DISEASES ANALYSIS COMPLETE!")
    print("="*70)
    print(f"✅ Successfully analyzed {len(all_disease_results)}/{len(top_mappable_diseases)} diseases")
    
    # Show summary of top genes across all diseases
    print(f"\n📊 SUMMARY: TOP GENES ACROSS ALL DISEASES")
    print("-" * 50)
    
    all_genes_summary = defaultdict(list)
    for disease_name, results in all_disease_results.items():
        for gene_info in results['shortcut_genes'][:5]:  # Top 5 per disease
            all_genes_summary[gene_info['gene']].append({
                'disease': disease_name,
                'score': gene_info['shortcut_score'],
                'degree': gene_info['degree'],
                'percentile': gene_info['percentile_class']
            })
    
    # Show genes that appear in multiple diseases
    multi_disease_genes = {gene: diseases for gene, diseases in all_genes_summary.items() if len(diseases) > 1}
    
    if multi_disease_genes:
        print(f"\n🔥 GENES APPEARING IN MULTIPLE DISEASES:")
        for gene, diseases in sorted(multi_disease_genes.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            disease_list = [f"{d['disease']} ({d['score']:.1f})" for d in diseases]
            print(f"   • {gene}: {len(diseases)} diseases - {', '.join(disease_list[:3])}{'...' if len(disease_list) > 3 else ''}")
    
    return {
        'disease_results': all_disease_results,
        'top_mappable_diseases': top_mappable_diseases,
        'degree_stats': degree_stats,
        'network_data': network_data,
        'multi_disease_genes': multi_disease_genes
    }

def save_top_diseases_analysis_report(results, results_path, datetime):
    """
    Save detailed report of top diseases analysis
    """
    
    print("📝 SAVING DETAILED ANALYSIS REPORT...")
    
    report_filename = f"Top_10_Diseases_Gene_Shortcuts_Analysis_{datetime}.txt"
    report_path = os.path.join(results_path, report_filename)
    
    # Function to convert emoji percentiles to text
    def clean_percentile(percentile_str):
        if "🔥 TOP 1%" in percentile_str:
            return "TOP 1%"
        elif "🌟 TOP 5%" in percentile_str:
            return "TOP 5%"
        elif "🔸 TOP 10%" in percentile_str:
            return "TOP 10%"
        elif "⬆️ TOP 25%" in percentile_str:
            return "TOP 25%"
        elif "📊 BELOW 75%" in percentile_str:
            return "BELOW 75%"
        else:
            return percentile_str.replace("🔥", "").replace("🌟", "").replace("🔸", "").replace("⬆️", "").replace("📊", "").strip()
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("TOP 10 DISEASES GENE SHORTCUTS ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Summary
        f.write("ANALYSIS SUMMARY:\n")
        f.write("-"*30 + "\n")
        f.write(f"Total diseases analyzed: {len(results['disease_results'])}\n")
        f.write(f"Network degree stats: {results['degree_stats']['stats']}\n\n")
        
        # Each disease analysis
        for disease_name, disease_data in results['disease_results'].items():
            f.write(f"\n{disease_name.upper()}\n")
            f.write("="*50 + "\n")
            f.write(f"Predictions: {disease_data['stats']['count']}\n")
            f.write(f"Unique drugs: {disease_data['stats']['num_drugs']}\n")
            f.write(f"Average confidence: {disease_data['stats']['avg_confidence']:.4f}\n\n")
            
            f.write("TOP SHORTCUT GENES:\n")
            f.write(f"{'Gene':<15} {'Score':<8} {'Degree':<8} {'Percentile':<12} {'Distance':<8}\n")
            f.write("-" * 70 + "\n")
            
            for i, gene_info in enumerate(disease_data['shortcut_genes'][:25], 1):
                clean_percentile_text = clean_percentile(gene_info['percentile_class'])
                f.write(f"{gene_info['gene']:<15} "
                       f"{gene_info['shortcut_score']:<8.1f} "
                       f"{gene_info['degree']:<8} "
                       f"{clean_percentile_text:<12} "
                       f"{gene_info['distance_to_disease']:<8}\n")
            f.write("\n")
        
        # Multi-disease genes
        if 'multi_disease_genes' in results and results['multi_disease_genes']:
            f.write("\nGENES APPEARING IN MULTIPLE DISEASES:\n")
            f.write("-"*40 + "\n")
            for gene, diseases in results['multi_disease_genes'].items():
                f.write(f"{gene}: {len(diseases)} diseases\n")
                for disease_info in diseases:
                    f.write(f"  - {disease_info['disease']}: score {disease_info['score']:.1f}\n")
                f.write("\n")
    
    print(f"✅ Report saved: {report_filename}")
    return report_path

# MAIN EXECUTION FUNCTION
if __name__ == "__main__":
    print("🧬 TOP 10 DISEASES GENE SHORTCUTS ANALYSIS")
    print("="*70)
    
    # Run the analysis
    results = run_top_10_diseases_analysis(sample_size=10000, top_diseases_count=10)
    
    if results:
        # Save detailed report
        report_path = save_top_diseases_analysis_report(results, results_path, datetime)
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"📁 Detailed report saved: {report_path}")
        print(f"🧬 Found gene shortcuts for {len(results['disease_results'])} diseases")
        print(f"🔥 {len(results['multi_disease_genes'])} genes appear in multiple diseases")