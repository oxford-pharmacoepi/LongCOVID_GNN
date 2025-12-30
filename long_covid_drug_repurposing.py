#!/usr/bin/env python3
"""
Long COVID Drug Repurposing Script (Modern Version)

This script adds Long COVID (MONDO:0100320) to an existing trained graph and
predicts drug repurposing candidates using a trained GNN model.

Steps:
1. Load existing graph and trained model
2. Check if Long COVID exists; if not, add it with proper edges
3. Run drug repurposing predictions
4. Export top candidates

Usage:
    python long_covid_drug_repurposing.py --top-k 50
    python long_covid_drug_repurposing.py --graph results/graph_latest.pt --model results/models/SAGEModel_best.pt

Author: Drug Repurposing Pipeline
Date: December 2025
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
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

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


class LongCOVIDDrugRepurposing:
    """Modern Long COVID drug repurposing analyzer"""
    
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
        """Add Long COVID node to the graph with proper edges"""
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
            print(f"  Initialized features from {len(similar_disease_indices)} similar diseases")
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
        
        # Step 5: Add all new edges to graph
        if new_edges:
            new_edge_tensor = torch.tensor(new_edges, dtype=torch.long).t()
            self.graph.edge_index = torch.cat([self.graph.edge_index, new_edge_tensor.to(self.device)], dim=1)
            
            print(f"\nLong COVID added to graph:")
            print(f"   Node index: {self.long_covid_idx}")
            print(f"   Total nodes: {self.graph.x.shape[0]:,} (was {self.original_num_nodes:,})")
            print(f"   Total edges: {self.graph.edge_index.shape[1]:,} (was {self.original_num_edges:,})")
            print(f"   New edges: {self.graph.edge_index.shape[1] - self.original_num_edges:,}")
    
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
        
        # Initialize model
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
    
    def predict_drug_candidates(self, top_k: int = 50) -> pd.DataFrame:
        """Predict drug repurposing candidates for Long COVID"""
        print(f"\nPredicting top {top_k} drug candidates for Long COVID...")
        
        with torch.no_grad():
            # Get node embeddings from model (same as training)
            embeddings = self.model(self.graph.x, self.graph.edge_index)
            
            # DIAGNOSTIC: Check embedding statistics
            print(f"\nEMBEDDING DIAGNOSTICS:")
            print(f"  Embedding shape: {embeddings.shape}")
            print(f"  Embedding mean: {embeddings.mean():.4f}")
            print(f"  Embedding std: {embeddings.std():.4f}")
            print(f"  Embedding min: {embeddings.min():.4f}")
            print(f"  Embedding max: {embeddings.max():.4f}")
            
            # Get Long COVID embedding
            long_covid_embedding = embeddings[self.long_covid_idx]
            print(f"\n  Long COVID embedding mean: {long_covid_embedding.mean():.4f}")
            print(f"  Long COVID embedding std: {long_covid_embedding.std():.4f}")
            
            # Get drug embeddings (first N nodes are drugs)
            num_drugs = len(self.drug_mapping)
            drug_embeddings = embeddings[:num_drugs]
            
            print(f"\n  Drug embeddings mean: {drug_embeddings.mean():.4f}")
            print(f"  Drug embeddings std: {drug_embeddings.std():.4f}")
            
            # IMPORTANT: Use the SAME edge prediction as during training
            # Edge score = dot product of embeddings (element-wise multiply then sum)
            edge_scores = (drug_embeddings * long_covid_embedding).sum(dim=-1)
            
            # Apply sigmoid to get probability of edge existing (just like training)
            probabilities = torch.sigmoid(edge_scores)
            
            # DIAGNOSTIC: Show score distribution across ALL drugs
            print(f"\nEDGE SCORE DISTRIBUTION (across all {num_drugs} drugs):")
            percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
            percentile_values = torch.quantile(edge_scores, torch.tensor([p/100 for p in percentiles]))
            for p, val in zip(percentiles, percentile_values):
                prob = torch.sigmoid(val)
                print(f"  {p:3d}th percentile: edge_score={val:7.4f} -> prob={prob:.4f}")
            
            print(f"\nPROBABILITY DISTRIBUTION:")
            prob_percentile_values = torch.quantile(probabilities, torch.tensor([p/100 for p in percentiles]))
            for p, val in zip(percentiles, prob_percentile_values):
                print(f"  {p:3d}th percentile: prob={val:.4f}")
            
            # Check if embeddings are actually different (variance check)
            embedding_variance = drug_embeddings.var(dim=0).mean()
            print(f"\n  Average variance across embedding dimensions: {embedding_variance:.6f}")
            if embedding_variance < 0.01:
                print("  ⚠️  WARNING: Very low variance! Embeddings may have collapsed.")
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, min(top_k, len(probabilities)))
            
            # Create results dataframe
            results = []
            for rank, (prob, idx) in enumerate(zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()), 1):
                drug_id = self.idx_to_drug[int(idx)]
                
                # Try to get drug name from ChEMBL or use ID
                drug_name = drug_id  # Default to ID
                
                results.append({
                    'rank': rank,
                    'drug_id': drug_id,
                    'drug_name': drug_name,
                    'probability': float(prob),
                    'edge_score': float(edge_scores[idx].cpu()),
                    'confidence': 'High' if prob > 0.7 else 'Medium' if prob > 0.5 else 'Low'
                })
            
            df = pd.DataFrame(results)
            
            print(f"\nTop 10 Drug Candidates:")
            print("=" * 80)
            
            for _, row in df.head(10).iterrows():
                print(f"{row['rank']:2d}. {row['drug_name']}")
                print(f"    Probability: {row['probability']:.4f} | Edge Score: {row['edge_score']:.4f} | Confidence: {row['confidence']}")
                print(f"    ID: {row['drug_id']}")
                print()
            
            return df
    
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
    
    def run_analysis(self, top_k: int = 50):
        """Run complete Long COVID drug repurposing analysis"""
        print("=" * 80)
        print("Long COVID Drug Repurposing Analysis")
        print("=" * 80)
        print(f"Disease: {self.long_covid_id} - {self.long_covid_name}")
        print(f"Requested predictions: {top_k}")
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
        results_df = self.predict_drug_candidates(top_k)
        
        # Step 6: Save results
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
  # Run with auto-detected latest files
  python long_covid_drug_repurposing.py --top-k 50
  
  # Specify specific graph and model
  python long_covid_drug_repurposing.py --graph results/graph_latest.pt --model results/SAGEModel_best.pt
  
  # Get top 100 candidates
  python long_covid_drug_repurposing.py --top-k 100 --output results/long_covid/top100.csv
"""
    )
    
    parser.add_argument('--graph', type=str, help='Path to graph file (.pt)')
    parser.add_argument('--model', type=str, help='Path to trained model (.pt)')
    parser.add_argument('--top-k', type=int, default=50, 
                       help='Number of top drug candidates to return (default: 50)')
    parser.add_argument('--output', type=str, 
                       help='Output CSV file path (default: auto-generated in results/long_covid/)')
    parser.add_argument('--data-path', type=str, default='processed_data',
                       help='Path to processed data directory')
    parser.add_argument('--results-path', type=str, default='results/long_covid',
                       help='Path to results directory (default: results/long_covid)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    enable_full_reproducibility(42)
    
    try:
        # Initialize analyzer
        analyzer = LongCOVIDDrugRepurposing(
            graph_path=args.graph,
            model_path=args.model,
            data_path=args.data_path,
            results_path=args.results_path
        )
        
        # Run analysis
        results = analyzer.run_analysis(top_k=args.top_k)
        
        # Save with custom output path if provided
        if args.output:
            analyzer.save_results(results, args.output)
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())