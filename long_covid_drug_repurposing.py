#!/usr/bin/env python3
"""
Long COVID Drug Repurposing Script

This script adds Long COVID (MONDO:0100320) to an existing trained graph and
predicts drug repurposing candidates using a trained GNN model.

Features:
- Predict all drugs or top-k candidates
- Look up drug names from ChEMBL database
- Generate visualisations of prediction distributions
- Export comprehensive reports

Usage:
    python long_covid_drug_repurposing.py --top-k 50 --lookup-names
    python long_covid_drug_repurposing.py --all-drugs --visualise

Author: Drug Repurposing Pipeline
Date: January 2026
"""

import os
import sys
import json
import glob
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import time
import urllib.request
import urllib.error

# Add src to path
sys.path.append(str(Path(__file__).parent))
from src.models import GCNModel, TransformerModel, SAGEModel, MODEL_CLASSES
from src.utils import set_seed, enable_full_reproducibility
from src.config import get_config


def find_latest_graph(results_dir='results'):
    """Auto-detect the latest graph file."""
    graph_patterns = [
        os.path.join(results_dir, 'graph_*.pt'),
        'graph_*.pt',
    ]
    
    graph_files = []
    for pattern in graph_patterns:
        graph_files.extend(glob.glob(pattern))
    
    if not graph_files:
        raise FileNotFoundError("No graph files found. Please create a graph first using script 1_create_graph.py")
    
    latest_graph = max(graph_files, key=os.path.getmtime)
    print(f"Auto-detected latest graph: {latest_graph}")
    return latest_graph


def find_latest_model(results_dir='results', model_type=None):
    """Auto-detect the latest trained model file."""
    if model_type:
        model_patterns = [
            os.path.join(results_dir, 'models', f'{model_type}_best_model_*.pt'),
            os.path.join(results_dir, f'{model_type}_best_model_*.pt'),
        ]
    else:
        model_patterns = [
            os.path.join(results_dir, 'models', '*_best_model_*.pt'),
            os.path.join(results_dir, '*_best_model_*.pt'),
        ]
    
    model_files = []
    for pattern in model_patterns:
        model_files.extend(glob.glob(pattern))
    
    if not model_files:
        raise FileNotFoundError("No trained models found. Please train models first using script 2_train_models.py")
    
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"Auto-detected latest model: {latest_model}")
    return latest_model


def lookup_chembl_drug_name(chembl_id: str, timeout: int = 10) -> Dict[str, str]:
    """
    Look up drug information from ChEMBL API
    
    Args:
        chembl_id: ChEMBL ID (e.g., 'CHEMBL25')
        timeout: Request timeout in seconds
    
    Returns:
        Dictionary with drug information
    """
    try:
        # ChEMBL API endpoint
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
        
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'DrugRepurposingPipeline/1.0 (Python/urllib)')
        req.add_header('Accept', 'application/json')
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode())
            
            # Extract key information
            pref_name = data.get('pref_name', None)
            if not pref_name or pref_name == 'None' or pref_name is None:
                pref_name = chembl_id
            
            # Try to find a common/generic name from synonyms
            synonyms = data.get('molecule_synonyms', [])
            common_name = pref_name
            
            if synonyms and isinstance(synonyms, list) and len(synonyms) > 0:
                # Prioritise shorter, readable names
                for syn in synonyms:
                    if isinstance(syn, dict):
                        syn_name = syn.get('molecule_synonym', '') or syn.get('synonym', '')
                    else:
                        continue
                    
                    # Skip empty, long chemical names, or CHEMBL IDs
                    if (syn_name and 
                        len(syn_name) < 50 and 
                        not syn_name.startswith('CHEMBL') and
                        not any(c in syn_name for c in ['[', ']', '{', '}']) and
                        len(syn_name) < len(common_name)):
                        common_name = syn_name
            
            # Get approval status
            max_phase = data.get('max_phase')
            if max_phase is None or max_phase == '' or max_phase == 'None':
                max_phase = 0
            else:
                try:
                    max_phase = int(max_phase)
                except (ValueError, TypeError):
                    max_phase = 0
            
            if max_phase == 4:
                status = 'Approved'
            elif max_phase >= 1:
                status = f'Clinical Phase {max_phase}'
            elif max_phase == 0:
                status = 'Preclinical'
            else:
                status = 'Unknown'
            
            first_approval = data.get('first_approval', None)
            molecule_type = data.get('molecule_type', 'Unknown')
            
            return {
                'preferred_name': pref_name,
                'common_name': common_name,
                'max_phase': max_phase,
                'molecule_type': molecule_type if molecule_type else 'Unknown',
                'first_approval': first_approval,
                'status': status
            }
    
    except urllib.error.HTTPError as e:
        # Drug not found in ChEMBL (404)
        return {
            'preferred_name': chembl_id, 
            'common_name': chembl_id, 
            'status': 'Not in ChEMBL', 
            'max_phase': 0,
            'molecule_type': 'Unknown',
            'first_approval': None
        }
    except Exception as e:
        # Any other error
        return {
            'preferred_name': chembl_id, 
            'common_name': chembl_id, 
            'status': 'Lookup Error', 
            'max_phase': 0,
            'molecule_type': 'Unknown',
            'first_approval': None
        }


def batch_lookup_drug_names(drug_ids: List[str], delay: float = 0.5, verbose: bool = True) -> pd.DataFrame:
    """
    Look up multiple drug names from ChEMBL API
    
    Args:
        drug_ids: List of ChEMBL IDs
        delay: Delay between API calls (seconds)
        verbose: Print progress
    
    Returns:
        DataFrame with drug information
    """
    if verbose:
        print(f"\nLooking up {len(drug_ids)} drug names from ChEMBL database...")
        print(f"   (This may take ~{len(drug_ids) * delay:.0f} seconds)")
    
    results = []
    success_count = 0
    error_count = 0
    
    for i, drug_id in enumerate(drug_ids, 1):
        if verbose and i % 10 == 0:
            print(f"   Progress: {i}/{len(drug_ids)} ({i/len(drug_ids)*100:.0f}%) - Success: {success_count}, Errors: {error_count}")
        
        info = lookup_chembl_drug_name(drug_id)
        info['chembl_id'] = drug_id
        results.append(info)
        
        # Track success/error
        if info['status'] not in ['Error', 'Network Error']:
            success_count += 1
        else:
            error_count += 1
        
        # Be polite to the API
        if i < len(drug_ids):
            time.sleep(delay)
    
    if verbose:
        print(f"   Lookup complete! Success: {success_count}/{len(drug_ids)} ({success_count/len(drug_ids)*100:.1f}%)")
        if error_count > 0:
            print(f"   WARNING: {error_count} lookups failed (may not exist in ChEMBL)")
    
    return pd.DataFrame(results)


def find_drug_in_predictions(drug_id: str, predictions_df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Find a specific drug in the predictions
    
    Args:
        drug_id: ChEMBL ID to search for
        predictions_df: DataFrame with all predictions
    
    Returns:
        Row with drug information, or None if not found
    """
    matches = predictions_df[predictions_df['drug_id'] == drug_id]
    if len(matches) > 0:
        return matches.iloc[0]
    return None


class LongCOVIDDrugRepurposing:
    """Modern Long COVID drug repurposing analyser"""
    
    def __init__(self, graph_path: str = None, model_path: str = None, 
                 data_path: str = "processed_data", results_path: str = "results/long_covid"):
        self.data_path = Path(data_path)
        self.results_path = Path(results_path)
        
        # Create results directory if it doesn't exist
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        self.mappings_path = self.data_path / "mappings"
        
        # Long COVID identifiers
        self.long_covid_id = "MONDO:0100320"  # Post-COVID-19 condition
        self.long_covid_name = "Long COVID (Post-COVID-19 condition)"
        
        # GWAS genes file
        self.gwas_genes_file = "gwas_genes_long_covid.txt"
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load mappings
        self.load_mappings()
        
        # Auto-detect or use provided paths
        if graph_path:
            self.graph_path = graph_path
            print(f"Using specified graph: {self.graph_path}")
        else:
            self.graph_path = find_latest_graph()
        
        if model_path:
            self.model_path = model_path
            print(f"Using specified model: {self.model_path}")
        else:
            self.model_path = find_latest_model()
        
        # Will be loaded later
        self.graph = None
        self.model = None
        self.long_covid_idx = None
    
    def load_mappings(self):
        """Load all mapping files"""
        print("\nLoading mappings...")
        
        # Load drug mappings
        drug_mapping_path = self.mappings_path / "drug_key_mapping.json"
        with open(drug_mapping_path, 'r') as f:
            self.drug_mapping = json.load(f)
        print(f"  Loaded {len(self.drug_mapping)} drugs")
        
        # Load gene mappings
        gene_mapping_path = self.mappings_path / "gene_key_mapping.json"
        with open(gene_mapping_path, 'r') as f:
            self.gene_mapping = json.load(f)
        print(f"  Loaded {len(self.gene_mapping)} genes")
        
        # Load disease mappings
        disease_mapping_path = self.mappings_path / "disease_key_mapping.json"
        with open(disease_mapping_path, 'r') as f:
            self.disease_mapping = json.load(f)
        print(f"  Loaded {len(self.disease_mapping)} diseases")
        
        # Load therapeutic area mappings
        therapeutic_mapping_path = self.mappings_path / "therapeutic_area_key_mapping.json"
        if therapeutic_mapping_path.exists():
            with open(therapeutic_mapping_path, 'r') as f:
                self.therapeutic_mapping = json.load(f)
            print(f"  Loaded {len(self.therapeutic_mapping)} therapeutic areas")
        else:
            self.therapeutic_mapping = {}
            print(f"  WARNING: No therapeutic area mappings found")
        
        # Create reverse mappings (idx -> id)
        self.idx_to_drug = {v: k for k, v in self.drug_mapping.items()}
        self.idx_to_gene = {v: k for k, v in self.gene_mapping.items()}
        self.idx_to_disease = {v: k for k, v in self.disease_mapping.items()}
        self.idx_to_therapeutic = {v: k for k, v in self.therapeutic_mapping.items()}
    
    def load_gwas_genes(self) -> List[str]:
        """Load GWAS genes and check which ones exist in the graph"""
        print(f"\nLoading GWAS genes from {self.gwas_genes_file}...")
        
        if not os.path.exists(self.gwas_genes_file):
            raise FileNotFoundError(f"GWAS genes file not found: {self.gwas_genes_file}")
        
        # Read genes from file
        with open(self.gwas_genes_file, 'r') as f:
            genes = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract Ensembl ID (before any comment)
                    gene_id = line.split('#')[0].strip()
                    if gene_id:
                        genes.append(gene_id)
        
        print(f"  Found {len(genes)} genes in file")
        
        # Check which genes exist in our graph
        found_genes = []
        missing_genes = []
        
        for gene in genes:
            if gene in self.gene_mapping:
                found_genes.append(gene)
            else:
                missing_genes.append(gene)
        
        print(f"  Found in graph: {len(found_genes)} genes ({len(found_genes)/len(genes)*100:.1f}%)")
        
        if missing_genes:
            print(f"  WARNING: Missing from graph: {len(missing_genes)} genes")
            if len(missing_genes) <= 10:
                for gene in missing_genes:
                    print(f"     - {gene}")
            else:
                for gene in missing_genes[:5]:
                    print(f"     - {gene}")
                print(f"     ... and {len(missing_genes)-5} more")
        
        return found_genes
    
    def load_graph(self):
        """Load the graph structure"""
        print(f"\nLoading graph from {self.graph_path}...")
        
        self.graph = torch.load(self.graph_path, map_location=self.device, weights_only=False)
        
        print(f"  Graph loaded successfully")
        print(f"     Nodes: {self.graph.x.shape[0]:,}")
        print(f"     Edges: {self.graph.edge_index.shape[1]:,}")
        print(f"     Features: {self.graph.x.shape[1]}")
        
        # Store original graph info
        self.original_num_nodes = self.graph.x.shape[0]
        self.original_num_edges = self.graph.edge_index.shape[1]
    
    def check_long_covid_exists(self) -> bool:
        """Check if Long COVID already exists in the graph"""
        exists = self.long_covid_id in self.disease_mapping
        
        if exists:
            self.long_covid_idx = self.disease_mapping[self.long_covid_id]
            print(f"\nLong COVID already exists in graph at index {self.long_covid_idx}")
        else:
            print(f"\nLong COVID not found in graph - will add it")
        
        return exists
    
    def add_long_covid_to_graph(self, gwas_genes: List[str]):
        """Add Long COVID node to the graph with proper edges and edge features"""
        print(f"\nAdding Long COVID to graph...")
        
        # Step 1: Add Long COVID to mappings
        self.long_covid_idx = len(self.disease_mapping)
        self.disease_mapping[self.long_covid_id] = self.long_covid_idx
        self.idx_to_disease[self.long_covid_idx] = self.long_covid_id
        
        print(f"  Assigned node index: {self.long_covid_idx}")
        
        # Step 2: Create new node features (average of similar diseases)
        similar_disease_ids = [
            "MONDO:0005263",   # Chronic fatigue syndrome
            "MONDO:0005002",   # COPD
            "MONDO:0004979",   # Asthma
            "MONDO:0007915",   # Systemic lupus erythematosus
            "MONDO:0011996",   # Fibromyalgia
        ]
        
        similar_disease_indices = []
        for disease_id in similar_disease_ids:
            if disease_id in self.disease_mapping:
                similar_disease_indices.append(self.disease_mapping[disease_id])
        
        if similar_disease_indices:
            # Average features from similar diseases
            similar_features = self.graph.x[similar_disease_indices]
            new_node_features = similar_features.mean(dim=0, keepdim=True)
            print(f"  Initialised features from {len(similar_disease_indices)} similar diseases")
        else:
            # Fallback: use mean of all disease features
            num_drugs = len(self.drug_mapping)
            num_genes = len(self.gene_mapping)
            disease_start = num_drugs + num_genes
            disease_features = self.graph.x[disease_start:disease_start + len(self.disease_mapping)]
            new_node_features = disease_features.mean(dim=0, keepdim=True)
            print(f"  WARNING: Using average of all disease features")
        
        # Add new node to graph
        self.graph.x = torch.cat([self.graph.x, new_node_features], dim=0)
        
        # Step 3: Create edges to GWAS genes (disease-gene edges)
        new_edges = []
        
        for gene_id in gwas_genes:
            gene_idx = self.gene_mapping[gene_id]
            # Bidirectional edges
            new_edges.append([self.long_covid_idx, gene_idx])
            new_edges.append([gene_idx, self.long_covid_idx])
        
        print(f"  Created {len(new_edges)} disease-gene edges ({len(gwas_genes)} genes, bidirectional)")
        
        # Step 4: Create edge to infectious disease therapeutic area
        infectious_disease_ta = "EFO_0005741"  # Infectious disease
        
        if infectious_disease_ta in self.therapeutic_mapping:
            ta_idx = self.therapeutic_mapping[infectious_disease_ta]
            # Bidirectional edges
            new_edges.append([self.long_covid_idx, ta_idx])
            new_edges.append([ta_idx, self.long_covid_idx])
            print(f"  Created edges to therapeutic area (Infectious disease)")
        else:
            print(f"  WARNING: Therapeutic area {infectious_disease_ta} not found")
        
        # Step 5: Add all new edges to graph (with edge features if needed)
        if new_edges:
            new_edge_tensor = torch.tensor(new_edges, dtype=torch.long).t()
            self.graph.edge_index = torch.cat([self.graph.edge_index, new_edge_tensor.to(self.device)], dim=1)
            
            # If graph has edge features, generate features for new edges
            if hasattr(self.graph, 'edge_attr') and self.graph.edge_attr is not None:
                num_new_edges = len(new_edges)
                edge_dim = self.graph.edge_attr.size(1)
                
                # Generate edge features as average of existing edge features
                # This ensures new edges have similar characteristics to existing edges
                avg_edge_features = self.graph.edge_attr.mean(dim=0, keepdim=True)
                new_edge_features = avg_edge_features.repeat(num_new_edges, 1)
                
                # Concatenate new edge features to existing ones
                self.graph.edge_attr = torch.cat([self.graph.edge_attr, new_edge_features.to(self.device)], dim=0)
                
                print(f"  Generated edge features for {num_new_edges} new edges (dim={edge_dim})")
            
            print(f"\nLong COVID added to graph:")
            print(f"   Node index: {self.long_covid_idx}")
            print(f"   Total nodes: {self.graph.x.shape[0]:,} (was {self.original_num_nodes:,})")
            print(f"   Total edges: {self.graph.edge_index.shape[1]:,} (was {self.original_num_edges:,})")
            print(f"   New edges: {self.graph.edge_index.shape[1] - self.original_num_edges:,}")
            if hasattr(self.graph, 'edge_attr') and self.graph.edge_attr is not None:
                print(f"   Total edge features: {self.graph.edge_attr.shape[0]:,}")
    
    def load_model(self):
        """Load the trained GNN model"""
        print(f"\nLoading model from {self.model_path}...")
        
        # Detect model type from filename
        model_filename = Path(self.model_path).name.lower()
        
        if 'gcn' in model_filename:
            model_class = GCNModel
            model_name = 'GCN'
        elif 'transformer' in model_filename:
            model_class = TransformerModel
            model_name = 'Transformer'
        elif 'sage' in model_filename:
            model_class = SAGEModel
            model_name = 'SAGE'
        else:
            print(f"  WARNING: Cannot detect model type, defaulting to SAGE")
            model_class = SAGEModel
            model_name = 'SAGE'
        
        print(f"  Model type: {model_name}")
        
        # Get model configuration
        config = get_config()
        model_config = config.get_model_config()
        
        # Check if graph has edge features and initialise model accordingly
        has_edge_attr = hasattr(self.graph, 'edge_attr') and self.graph.edge_attr is not None
        if has_edge_attr:
            print(f"  ✓ Graph has edge features: {self.graph.edge_attr.shape}")
            edge_dim = self.graph.edge_attr.size(1)
            
            # For TransformerModel, pass edge_dim to constructor
            if model_name == 'Transformer':
                self.model = model_class(
                    in_channels=self.graph.x.shape[1],
                    hidden_channels=model_config['hidden_channels'],
                    out_channels=model_config['out_channels'],
                    num_layers=model_config['num_layers'],
                    dropout_rate=model_config['dropout_rate'],
                    edge_dim=edge_dim
                ).to(self.device)
            else:
                # For other models, pass edge_dim if they support it
                self.model = model_class(
                    in_channels=self.graph.x.shape[1],
                    hidden_channels=model_config['hidden_channels'],
                    out_channels=model_config['out_channels'],
                    num_layers=model_config['num_layers'],
                    dropout_rate=model_config['dropout_rate'],
                    edge_dim=edge_dim
                ).to(self.device)
        else:
            print("  Note: No edge features found")
            # Initialise model without edge_dim
            self.model = model_class(
                in_channels=self.graph.x.shape[1],
                hidden_channels=model_config['hidden_channels'],
                out_channels=model_config['out_channels'],
                num_layers=model_config['num_layers'],
                dropout_rate=model_config['dropout_rate']
            ).to(self.device)
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"  Model loaded and ready for inference")
    
    def predict_drug_candidates(self, top_k: int = None, predict_all: bool = False) -> pd.DataFrame:
        """
        Predict drug repurposing candidates for Long COVID
        
        Args:
            top_k: Number of top candidates to return (None = return all)
            predict_all: If True, predict and store scores for ALL drugs
        
        Returns:
            DataFrame with predictions
        """
        num_drugs = len(self.drug_mapping)
        
        if predict_all or top_k is None:
            print(f"\nPredicting ALL {num_drugs:,} drug candidates for Long COVID...")
        else:
            print(f"\nPredicting top {top_k} drug candidates for Long COVID...")
        
        with torch.no_grad():
            # Get node embeddings from model
            # Pass edge features if they exist in the graph
            if hasattr(self.graph, 'edge_attr') and self.graph.edge_attr is not None:
                embeddings = self.model(self.graph.x, self.graph.edge_index, edge_attr=self.graph.edge_attr)
            else:
                embeddings = self.model(self.graph.x, self.graph.edge_index)
            
            # Normalise embeddings (consistent with training/testing)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Get Long COVID embedding
            long_covid_embedding = embeddings[self.long_covid_idx]
            
            # Get all drug embeddings (first N nodes are drugs)
            drug_embeddings = embeddings[:num_drugs]
            
            # Calculate similarity scores (dot product, now bounded to [-1, 1])
            scores = torch.matmul(drug_embeddings, long_covid_embedding)
            
            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(scores)
            
            # Store for later use (for visualisations)
            self.all_scores = scores.cpu().numpy()
            self.all_probabilities = probabilities.cpu().numpy()
            
            # Determine how many predictions to return
            if predict_all or top_k is None:
                # Return all drugs, sorted by probability
                sorted_indices = torch.argsort(probabilities, descending=True)
                top_probs = probabilities[sorted_indices]
                top_indices = sorted_indices
            else:
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probabilities, min(top_k, len(probabilities)))
            
            # Create results dataframe
            results = []
            for rank, (prob, idx) in enumerate(zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()), 1):
                drug_id = self.idx_to_drug[int(idx)]
                
                results.append({
                    'rank': rank,
                    'drug_id': drug_id,
                    'drug_name': drug_id,  # Will be updated if lookup is requested
                    'probability': float(prob),
                    'score': float(scores[idx].cpu()),
                    'confidence': 'High' if prob > 0.7 else 'Medium' if prob > 0.5 else 'Low'
                })
            
            df = pd.DataFrame(results)
            
            print(f"  Generated {len(df):,} predictions")
            
            # Print statistics
            print(f"\n Prediction Statistics:")
            print(f"   Total drugs:      {num_drugs:,}")
            print(f"   Probability range: {self.all_probabilities.min():.4f} - {self.all_probabilities.max():.4f}")
            print(f"   Mean probability:  {self.all_probabilities.mean():.4f}")
            print(f"   Median probability: {np.median(self.all_probabilities):.4f}")
            print(f"   Std deviation:     {self.all_probabilities.std():.4f}")
            
            # Count by confidence
            high_conf = np.sum(self.all_probabilities > 0.7)
            med_conf = np.sum((self.all_probabilities > 0.5) & (self.all_probabilities <= 0.7))
            low_conf = np.sum(self.all_probabilities <= 0.5)
            
            print(f"\n Confidence Distribution:")
            print(f"   High (>0.7):    {high_conf:,} drugs ({high_conf/num_drugs*100:.1f}%)")
            print(f"   Medium (0.5-0.7): {med_conf:,} drugs ({med_conf/num_drugs*100:.1f}%)")
            print(f"   Low (<0.5):     {low_conf:,} drugs ({low_conf/num_drugs*100:.1f}%)")
            
            print(f"\n Top 10 Drug Candidates:")
            print("=" * 80)
            
            for _, row in df.head(10).iterrows():
                print(f"{row['rank']:2d}. {row['drug_name']}")
                print(f"    Probability: {row['probability']:.4f} | Score: {row['score']:.4f} | Confidence: {row['confidence']}")
                print(f"    ID: {row['drug_id']}")
                print()
            
            return df
    
    def lookup_drug_names(self, results_df: pd.DataFrame, top_n: int = None) -> pd.DataFrame:
        """
        Look up drug names from ChEMBL for the predictions
        
        Args:
            results_df: DataFrame with predictions
            top_n: Only lookup top N drugs (None = lookup all, but can be slow!)
        
        Returns:
            DataFrame with drug names added
        """
        if top_n is not None:
            drug_ids = results_df.head(top_n)['drug_id'].tolist()
            print(f"\n Looking up names for top {top_n} drugs...")
        else:
            drug_ids = results_df['drug_id'].tolist()
            print(f"\n Looking up names for {len(drug_ids)} drugs...")
            if len(drug_ids) > 100:
                print(f"     This will take ~{len(drug_ids) * 0.3 / 60:.1f} minutes. Consider using --lookup-top-n instead.")
        
        # Lookup drug information
        drug_info_df = batch_lookup_drug_names(drug_ids)
        
        # Merge with results
        results_with_names = results_df.merge(
            drug_info_df,
            left_on='drug_id',
            right_on='chembl_id',
            how='left'
        )
        
        # Update drug_name column
        results_with_names['drug_name'] = results_with_names['common_name'].fillna(results_with_names['drug_id'])
        
        # Add additional columns
        results_with_names['preferred_name'] = results_with_names['preferred_name'].fillna(results_with_names['drug_id'])
        results_with_names['approval_status'] = results_with_names['status'].fillna('Unknown')
        results_with_names['max_phase'] = results_with_names['max_phase'].fillna(-1).astype(int)
        
        # Drop duplicate columns
        results_with_names = results_with_names.drop(columns=['chembl_id', 'status'], errors='ignore')
        
        # Show approved drugs in top candidates
        approved_in_top = results_with_names[results_with_names['approval_status'] == 'Approved'].head(20)
        
        if len(approved_in_top) > 0:
            print(f"\n FDA-Approved Drugs in Top Candidates:")
            print("=" * 80)
            for _, row in approved_in_top.head(10).iterrows():
                print(f"{row['rank']:2d}. {row['drug_name']}")
                print(f"    Probability: {row['probability']:.4f} | Approval: {row['first_approval']}")
                print(f"    Preferred name: {row['preferred_name']}")
                print()
        
        return results_with_names
    
    def create_visualisations(self, results_df: pd.DataFrame, timestamp: str = None):
        """
        Create visualisations of prediction results
        
        Args:
            results_df: DataFrame with predictions
            timestamp: Timestamp string for filenames
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n Creating visualisations...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Probability distribution (all drugs)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(self.all_probabilities, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(np.median(self.all_probabilities), color='red', linestyle='--', linewidth=2, 
                    label=f'Median: {np.median(self.all_probabilities):.3f}')
        ax1.axvline(self.all_probabilities.mean(), color='orange', linestyle='--', linewidth=2,
                    label=f'Mean: {self.all_probabilities.mean():.3f}')
        ax1.set_xlabel('Probability', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Drugs', fontsize=12, fontweight='bold')
        ax1.set_title('Probability Distribution (All Drugs)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        
        # 2. Top 20 candidates bar chart
        ax2 = fig.add_subplot(gs[0, 1])
        top_20 = results_df.head(20)
        colors = ['red' if c == 'High' else 'orange' if c == 'Medium' else 'gray' 
                  for c in top_20['confidence']]
        
        bars = ax2.barh(range(len(top_20)), top_20['probability'], color=colors, alpha=0.7, edgecolor='black')
        ax2.set_yticks(range(len(top_20)))
        ax2.set_yticklabels([f"{r['rank']}. {r['drug_name'][:30]}" for _, r in top_20.iterrows()], fontsize=9)
        ax2.set_xlabel('Probability', fontsize=12, fontweight='bold')
        ax2.set_title('Top 20 Drug Candidates', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(alpha=0.3, axis='x')
        ax2.axvline(0.7, color='red', linestyle='--', alpha=0.5, label='High confidence')
        ax2.axvline(0.5, color='orange', linestyle='--', alpha=0.5, label='Medium confidence')
        ax2.legend(fontsize=9)
        
        # 3. Confidence pie chart
        ax3 = fig.add_subplot(gs[1, 0])
        high_conf = np.sum(self.all_probabilities > 0.7)
        med_conf = np.sum((self.all_probabilities > 0.5) & (self.all_probabilities <= 0.7))
        low_conf = np.sum(self.all_probabilities <= 0.5)
        
        sizes = [high_conf, med_conf, low_conf]
        labels = [f'High\n({high_conf})', f'Medium\n({med_conf})', f'Low\n({low_conf})']
        colors_pie = ['#ff6b6b', '#ffd93d', '#95e1d3']
        
        ax3.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
        
        # 4. Top 50 vs Rest boxplot
        ax4 = fig.add_subplot(gs[1, 1])
        top_50_probs = results_df.head(50)['probability']
        rest_probs = results_df.iloc[50:]['probability'] if len(results_df) > 50 else []
        
        if len(rest_probs) > 0:
            box_data = [top_50_probs, rest_probs]
            bp = ax4.boxplot(box_data, tick_labels=['Top 50', 'Rest'], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightcoral')
            bp['boxes'][1].set_facecolor('lightblue')
        else:
            ax4.boxplot([top_50_probs], tick_labels=['Top 50'], patch_artist=True)
        
        ax4.set_ylabel('Probability', fontsize=10, fontweight='bold')
        ax4.set_title('Top 50 vs Rest', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3, axis='y')
        
        # Main title
        fig.suptitle('Long COVID Drug Repurposing Analysis', fontsize=18, fontweight='bold', y=0.995)
        
        # Save plot
        plot_file = self.results_path / f"long_covid_predictions_visualisation_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"   Saved visualisation: {plot_file}")
        
        plt.close()
    
    def save_results(self, results_df: pd.DataFrame, output_file: str = None):
        """Save prediction results"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.results_path / f"long_covid_drug_predictions_{timestamp}.csv"
        else:
            output_file = Path(output_file)
        
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        # Also save a summary JSON
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'disease_id': self.long_covid_id,
            'disease_name': self.long_covid_name,
            'graph_path': str(self.graph_path),
            'model_path': str(self.model_path),
            'num_predictions': len(results_df),
            'num_nodes': int(self.graph.x.shape[0]),
            'num_edges': int(self.graph.edge_index.shape[1]),
            'long_covid_node_index': int(self.long_covid_idx),
            'top_10_drugs': results_df.head(10)[['drug_id', 'drug_name', 'probability']].to_dict('records')
        }
        
        summary_file = output_file.parent / output_file.name.replace('.csv', '_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to: {summary_file}")
    
    def run_analysis(self, top_k: int = None, predict_all: bool = False, 
                    lookup_names: bool = False, lookup_top_n: int = None,
                    visualise: bool = False):
        """
        Run complete Long COVID drug repurposing analysis
        
        Args:
            top_k: Number of top candidates to export
            predict_all: Predict all drugs (not just top-k)
            lookup_names: Look up drug names from ChEMBL
            lookup_top_n: Only lookup names for top N drugs
            visualise: Create visualisation plots
        """
        print("=" * 80)
        print("Long COVID Drug Repurposing Analysis")
        print("=" * 80)
        print(f"Disease: {self.long_covid_id} - {self.long_covid_name}")
        
        if predict_all:
            print(f"Mode: Predict ALL drugs")
        elif top_k:
            print(f"Mode: Top {top_k} predictions")
        else:
            print(f"Mode: Top 50 predictions (default)")
            top_k = 50
        
        if lookup_names:
            if lookup_top_n:
                print(f"Drug name lookup: Top {lookup_top_n} drugs")
            else:
                print(f"Drug name lookup: Enabled for all predictions")
        
        if visualise:
            print(f"Visualisations: Enabled")
        
        print()
        
        # Step 1: Load graph
        self.load_graph()
        
        # Step 2: Load GWAS genes
        gwas_genes = self.load_gwas_genes()
        
        if not gwas_genes:
            raise ValueError("No GWAS genes found in graph. Please check your gene mappings.")
        
        # Step 3: Check if Long COVID exists, if not add it
        exists = self.check_long_covid_exists()
        
        if not exists:
            self.add_long_covid_to_graph(gwas_genes)
        
        # Step 4: Load model
        self.load_model()
        
        # Step 5: Predict drug candidates
        results_df = self.predict_drug_candidates(top_k=top_k, predict_all=predict_all)
        
        # Step 6: Look up drug names if requested
        if lookup_names:
            results_df = self.lookup_drug_names(results_df, top_n=lookup_top_n)
        
        # Step 7: Create visualisations if requested
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if visualise:
            self.create_visualisations(results_df, timestamp)
        
        # Step 8: Save results
        self.save_results(results_df)
        
        print("\n" + "=" * 80)
        print("Analysis completed successfully!")
        print("=" * 80)
        
        return results_df


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Long COVID Drug Repurposing Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: Get top 50 candidates
  python long_covid_drug_repurposing.py
  
  # Get top 100 candidates with drug names and visualisations
  python long_covid_drug_repurposing.py --top-k 100 --lookup-names --visualise
  
  # Predict ALL drugs with visualisations (no name lookup to save time)
  python long_covid_drug_repurposing.py --all-drugs --visualise
  
  # Get top 200 candidates but only lookup names for top 50
  python long_covid_drug_repurposing.py --top-k 200 --lookup-names --lookup-top-n 50 --visualise
  
  # Specify custom graph and model
  python long_covid_drug_repurposing.py --graph results/graph_latest.pt --model results/SAGEModel_best.pt --visualise

Note: Drug name lookup uses ChEMBL API and can be slow for many drugs.
      Use --lookup-top-n to limit lookups to top N candidates.
"""
    )
    
    # File paths
    parser.add_argument('--graph', type=str, help='Path to graph file (.pt)')
    parser.add_argument('--model', type=str, help='Path to trained model (.pt)')
    parser.add_argument('--data-path', type=str, default='processed_data',
                       help='Path to processed data directory')
    parser.add_argument('--results-path', type=str, default='results/long_covid',
                       help='Path to results directory (default: results/long_covid)')
    parser.add_argument('--output', type=str, 
                       help='Output CSV file path (default: auto-generated in results/long_covid/)')
    
    # Prediction options
    prediction_group = parser.add_argument_group('Prediction Options')
    prediction_group.add_argument('--top-k', type=int, default=50, 
                                  help='Number of top drug candidates to return (default: 50)')
    prediction_group.add_argument('--all-drugs', action='store_true',
                                  help='Predict ALL drugs instead of just top-k (returns all ~1,900 drugs)')
    
    # Drug name lookup options
    lookup_group = parser.add_argument_group('Drug Name Lookup Options')
    lookup_group.add_argument('--lookup-names', action='store_true',
                             help='Look up drug names from ChEMBL database (can be slow)')
    lookup_group.add_argument('--lookup-top-n', type=int, 
                             help='Only lookup names for top N drugs (default: lookup all in results)')
    
    # Visualisation options
    viz_group = parser.add_argument_group('Visualisation Options')
    viz_group.add_argument('--visualise', action='store_true',
                          help='Create visualisation plots of predictions')
    viz_group.add_argument('--no-visualise', action='store_true',
                          help='Skip visualisation generation')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    config = get_config()
    enable_full_reproducibility(config.seed)
    
    # Determine visualisation flag
    visualise = args.visualise and not args.no_visualise
    
    try:
        # Initialise analyser
        analyser = LongCOVIDDrugRepurposing(
            graph_path=args.graph,
            model_path=args.model,
            data_path=args.data_path,
            results_path=args.results_path
        )
        
        # Run analysis with all options
        results = analyser.run_analysis(
            top_k=None if args.all_drugs else args.top_k,
            predict_all=args.all_drugs,
            lookup_names=args.lookup_names,
            lookup_top_n=args.lookup_top_n,
            visualise=visualise
        )
        
        # Save with custom output path if provided
        if args.output:
            analyser.save_results(results, args.output)
        
        # Print summary
        print(f"\nSummary:")
        print(f"   Total predictions: {len(results):,}")
        print(f"   High confidence (>0.7): {len(results[results['confidence'] == 'High']):,}")
        print(f"   Medium confidence (0.5-0.7): {len(results[results['confidence'] == 'Medium']):,}")
        
        if args.lookup_names:
            approved = results[results.get('approval_status', '') == 'Approved']
            if len(approved) > 0:
                print(f"   FDA-approved drugs found: {len(approved):,}")
        
        print(f"\nNext steps:")
        print(f"   1. Review the top candidates in the CSV file")
        if visualise:
            print(f"   2. Check the visualisation plots in {args.results_path}/")
        if not args.lookup_names:
            print(f"   2. Re-run with --lookup-names to get drug names")
        print(f"   3. Research mechanisms of action for top candidates")
        print(f"   4. Consider clinical trial feasibility for approved drugs")
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())