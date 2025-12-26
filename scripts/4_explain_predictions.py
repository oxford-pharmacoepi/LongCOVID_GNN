#!/usr/bin/env python3
"""
GNN Explainer Script for Drug-Disease Prediction
Generates explanations for model predictions using GNNExplainer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, wilcoxon
import networkx as nx
from torch_geometric.utils import to_networkx, k_hop_subgraph
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ExplanationType, ModelMode, ModelTaskLevel, MaskType
from collections import defaultdict
import random
import time
import math
import argparse
import json
import os
from pathlib import Path
import datetime as dt
import warnings
import glob

warnings.filterwarnings('ignore')

# Import from shared modules
from src.models import GCNModel, TransformerModel, SAGEModel, MODEL_CLASSES
from src.utils import set_seed, enable_full_reproducibility
from src.config import get_config, create_custom_config
from src.mlflow_tracker import ExperimentTracker


def find_latest_graph(results_dir='results'):
    """Auto-detect the latest graph file."""
    graph_patterns = [
        os.path.join(results_dir, 'graph_*.pt'),
        'graph_*.pt',
        os.path.join('data', 'processed', '*.pt'),
    ]
    
    graph_files = []
    for pattern in graph_patterns:
        graph_files.extend(glob.glob(pattern))
    
    if not graph_files:
        raise FileNotFoundError("No graph files found. Please create a graph first using script 1_create_graph.py")
    
    # Sort by modification time (most recent first)
    latest_graph = max(graph_files, key=os.path.getmtime)
    print(f"Auto-detected latest graph: {latest_graph}")
    return latest_graph


def find_latest_model(results_dir='results'):
    """Auto-detect the latest trained model file."""
    model_patterns = [
        os.path.join(results_dir, '*_best_model_*.pt'),
        os.path.join(results_dir, 'models', '*_best_model_*.pt'),
        '*_best_model_*.pt',
    ]
    
    model_files = []
    for pattern in model_patterns:
        model_files.extend(glob.glob(pattern))
    
    if not model_files:
        raise FileNotFoundError("No trained models found. Please train models first using script 2_train_models.py")
    
    # Sort by modification time (most recent first)
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"Auto-detected latest model: {latest_model}")
    return latest_model


def find_latest_predictions(results_dir='results'):
    """Auto-detect the latest FP predictions file."""
    predictions_patterns = [
        os.path.join(results_dir, 'predictions', '*_FP_predictions_*.csv'),
        os.path.join(results_dir, 'predictions', '*_FP_predictions_*.pt'),
        os.path.join(results_dir, '*_FP_predictions_*.csv'),
        os.path.join(results_dir, '*_FP_predictions_*.pt'),
        'predictions/*_FP_predictions_*.csv',
        'predictions/*_FP_predictions_*.pt',
    ]
    
    prediction_files = []
    for pattern in predictions_patterns:
        prediction_files.extend(glob.glob(pattern))
    
    if not prediction_files:
        raise FileNotFoundError("No FP prediction files found. Please run script 3_test_evaluate.py first with --export-fp flag")
    
    # Sort by modification time (most recent first)
    latest_predictions = max(prediction_files, key=os.path.getmtime)
    print(f"Auto-detected latest predictions: {latest_predictions}")
    return latest_predictions


class LinkPredictor(nn.Module):
    """Wrapper to make GNN suitable for edge-level explanations."""
    
    def __init__(self, gnn):
        super().__init__()
        self.gnn = gnn
        self.target_edge = None
    
    def set_target_edge(self, drug_idx, disease_idx):
        """Set the target drug-disease pair for explanation."""
        self.target_edge = (drug_idx, disease_idx)
    
    def forward(self, x, edge_index, index=None):
        """Forward pass compatible with GNNExplainer."""
        z = self.gnn(x, edge_index)  # Get node embeddings
        
        # Use provided index or fall back to stored target_edge
        if index is not None:
            src, dst = index
        elif self.target_edge is not None:
            src, dst = self.target_edge
        else:
            raise ValueError("No target edge specified. Call set_target_edge() first.")
        
        # Compute prediction for this specific pair
        logit = (z[src] * z[dst]).sum(-1)
        return logit


class ExplanationCache:
    """Global cache for explanations to avoid recomputation."""
    
    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0
    
    def get_key(self, drug_idx, disease_idx):
        return f"{drug_idx}_{disease_idx}"
    
    def get(self, drug_idx, disease_idx):
        key = self.get_key(drug_idx, disease_idx)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, drug_idx, disease_idx, explanation):
        key = self.get_key(drug_idx, disease_idx)
        self.cache[key] = explanation
    
    def stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return f"Cache stats: {self.hits}/{total} hits ({hit_rate:.2%})"


class GNNExplainerAnalyzer:
    """Main GNNExplainer implementation for drug-disease predictions."""
    
    def __init__(self, model, graph, device, config=None):
        self.model = model
        self.graph = graph.to(device)
        self.device = device
        self.config = config or {}
        self.cache = ExplanationCache()
        
        # Wrap model for edge-level predictions
        self.link_predictor = LinkPredictor(model).to(device)
        
        # Create explainer with optimized configuration
        self.explainer = Explainer(
            model=self.link_predictor,
            algorithm=GNNExplainer(epochs=self.config.get('epochs', 50)),
            explanation_type=ExplanationType.model,
            node_mask_type=MaskType.object,
            edge_mask_type=MaskType.object,
            model_config=dict(
                mode=ModelMode.binary_classification,
                task_level=ModelTaskLevel.edge,
                return_type='raw'
            ),
        )
    
    def explain_drug_disease_prediction(self, drug_idx, disease_idx, k_hop=2):
        """Generate explanation for a drug-disease pair."""
        # Check cache first
        cached_result = self.cache.get(drug_idx, disease_idx)
        if cached_result is not None:
            return cached_result
        
        try:
            # Validate node indices first
            num_nodes = self.graph.x.size(0)
            if drug_idx >= num_nodes or disease_idx >= num_nodes:
                print(f"Skipping invalid indices: drug_idx={drug_idx}, disease_idx={disease_idx}, max_nodes={num_nodes}")
                return None
            
            # Extract subgraph for efficiency
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                node_idx=[drug_idx, disease_idx],
                num_hops=k_hop,
                edge_index=self.graph.edge_index,
                relabel_nodes=True
            )
            
            # Get subgraph features and ensure float32 dtype
            sub_x = self.graph.x[subset].float().to(self.device)
            edge_index = edge_index.to(self.device)
            
            # Map drug and disease indices to subgraph
            drug_sub_idx = mapping[0].item()
            disease_sub_idx = mapping[1].item()
            
            # Set target for subgraph
            self.link_predictor.set_target_edge(drug_sub_idx, disease_sub_idx)
            
            # Explain on subgraph
            explanation = self.explainer(x=sub_x, edge_index=edge_index)
            
            # Process results
            explanation_data = self._process_explanation(
                explanation, drug_idx, disease_idx, subset, drug_sub_idx, disease_sub_idx
            )
            
            # Cache the result
            if explanation_data:
                self.cache.put(drug_idx, disease_idx, explanation_data)
            
            return explanation_data
            
        except Exception as e:
            print(f"Error generating explanation for Drug {drug_idx} -> Disease {disease_idx}: {e}")
            return None
    
    def _process_explanation(self, explanation, orig_drug_idx, orig_disease_idx, 
                           subset, sub_drug_idx, sub_disease_idx):
        """Process explanation with both edge and node masks."""
        try:
            edge_mask = explanation.get('edge_mask')
            node_mask = explanation.get('node_mask')

            if edge_mask is None and node_mask is None:
                return None

            def _to_1d_float_tensor(t):
                if t is None:
                    return torch.empty(0, dtype=torch.float32, device=self.device)
                if t.ndim > 1:
                    if t.shape[-1] == 1:
                        t = t.squeeze(-1)
                    else:
                        t = t.mean(dim=-1)
                return t.float()

            edge_mask = _to_1d_float_tensor(edge_mask)
            node_mask = _to_1d_float_tensor(node_mask)

            # Process edge masks
            important_edges = []
            if edge_mask.numel() > 0:
                finite = torch.isfinite(edge_mask)
                valid = edge_mask[finite]
                if valid.numel() > 0 and (valid.max() > valid.min()):
                    edge_threshold = torch.quantile(valid, 0.8)
                    cmp_mask = edge_mask.clone()
                    cmp_mask[~finite] = float('-inf')
                    important_edge_indices = torch.where(cmp_mask >= edge_threshold)[0]
                    
                    # Map back to original indices
                    _, sub_edge_index, _, _ = k_hop_subgraph(
                        node_idx=[orig_drug_idx, orig_disease_idx],
                        num_hops=2,
                        edge_index=self.graph.edge_index,
                        relabel_nodes=True
                    )
                    
                    for eidx in important_edge_indices.tolist():
                        if eidx < sub_edge_index.size(1):
                            sub_source = int(sub_edge_index[0, eidx])
                            sub_target = int(sub_edge_index[1, eidx])
                            
                            orig_source = int(subset[sub_source])
                            orig_target = int(subset[sub_target])
                            
                            imp = float(edge_mask[eidx])
                            if math.isfinite(imp):
                                important_edges.append({
                                    'source': orig_source,
                                    'target': orig_target,
                                    'importance': imp,
                                    'edge_idx': int(eidx),
                                })

            # Process node masks
            node_importance_scores = {}
            important_nodes = {orig_drug_idx, orig_disease_idx}
            
            if node_mask.numel() > 0:
                for sub_idx in range(len(subset)):
                    if sub_idx < node_mask.size(0):
                        orig_node_idx = int(subset[sub_idx])
                        score = float(node_mask[sub_idx])
                        if math.isfinite(score):
                            node_importance_scores[orig_node_idx] = score
                            important_nodes.add(orig_node_idx)

            if len(important_edges) == 0 and len(node_importance_scores) == 0:
                return None

            return {
                'drug_idx': orig_drug_idx,
                'disease_idx': orig_disease_idx,
                'important_nodes': list(important_nodes),
                'important_edges': important_edges,
                'num_important_edges': len(important_edges),
                'node_importance_scores': node_importance_scores,
            }

        except Exception as e:
            print(f"Error processing explanation: {e}")
            return None
    
    def explain_multiple_predictions(self, fp_pairs, max_explanations=500):
        """Generate explanations for multiple predictions."""
        explanations = {}
        print(f"Generating explanations for {min(len(fp_pairs), max_explanations)} predictions...")
        
        for i, pair in enumerate(fp_pairs[:max_explanations]):
            if i % 50 == 0:
                print(f"   Progress: {i}/{min(len(fp_pairs), max_explanations)}")
            
            pair_key = f"{pair['drug_name']} -> {pair['disease_name']}"
            
            explanation = self.explain_drug_disease_prediction(pair['drug_idx'], pair['disease_idx'])
            
            if explanation:
                explanations[pair_key] = {
                    'pair': pair,
                    'explanation': explanation,
                    'has_explanation': True
                }
            else:
                explanations[pair_key] = {
                    'pair': pair,
                    'explanation': None,
                    'has_explanation': False
                }
        
        successful_explanations = len([e for e in explanations.values() if e['has_explanation']])
        print(f"Generated {successful_explanations} successful explanations")
        print(f"Cache performance: {self.cache.stats()}")
        
        return explanations


def load_actual_mappings(mappings_dir):
    """Load the real drug/disease mappings from processed data."""
    import json
    import os
    
    print(f"Loading actual mappings from {mappings_dir}")
    
    try:
        # Load drug mappings
        drug_mapping_path = os.path.join(mappings_dir, "drug_key_mapping.json")
        with open(drug_mapping_path, 'r') as f:
            drug_name_to_idx = json.load(f)
        
        # Load disease mappings  
        disease_mapping_path = os.path.join(mappings_dir, "disease_key_mapping.json")
        with open(disease_mapping_path, 'r') as f:
            disease_name_to_idx = json.load(f)
        
        print(f"Loaded {len(drug_name_to_idx)} drug mappings and {len(disease_name_to_idx)} disease mappings")
        return drug_name_to_idx, disease_name_to_idx
        
    except FileNotFoundError as e:
        print(f"Error loading mappings: {e}")
        print("Available files in mappings directory:")
        for f in os.listdir(mappings_dir):
            print(f"  {f}")
        return None, None
    except Exception as e:
        print(f"Error loading mappings: {e}")
        return None, None


def load_fp_predictions(predictions_path, mappings_dir=None):
    """Load false positive predictions from CSV or PyTorch file with proper index mapping."""
    print(f"Loading FP predictions from {predictions_path}")
    
    # Load actual mappings if directory provided
    drug_name_to_idx = None
    disease_name_to_idx = None
    if mappings_dir and os.path.exists(mappings_dir):
        drug_name_to_idx, disease_name_to_idx = load_actual_mappings(mappings_dir)
    
    if predictions_path.endswith('.csv'):
        df = pd.read_csv(predictions_path)
        
        # Handle different possible column names
        confidence_col = None
        for col in ['confident_score', 'confidence_score', 'prediction', 'probability']:
            if col in df.columns:
                confidence_col = col
                break
        
        if confidence_col is None:
            raise ValueError(f"No confidence column found. Available: {df.columns.tolist()}")
        
        fp_pairs = []
        skipped_count = 0
        
        for _, row in df.iterrows():
            drug_name = row['drug_name']
            disease_name = row['disease_name']
            
            # Try to get actual indices from mappings first
            if drug_name_to_idx is not None and disease_name_to_idx is not None:
                drug_idx = drug_name_to_idx.get(drug_name)
                disease_idx = disease_name_to_idx.get(disease_name)
                
                # If direct lookup fails, try to extract index from generic name
                if drug_idx is None and drug_name.startswith('Drug_'):
                    try:
                        potential_idx = int(drug_name.split('_')[1])
                        # Check if this index exists in the range of mapped drugs
                        drug_indices = list(drug_name_to_idx.values())
                        if potential_idx in drug_indices:
                            drug_idx = potential_idx
                    except (ValueError, IndexError):
                        pass
                
                if disease_idx is None and disease_name.startswith('Disease_'):
                    try:
                        potential_idx = int(disease_name.split('_')[1])
                        # Check if this index exists in the range of mapped diseases
                        disease_indices = list(disease_name_to_idx.values())
                        if potential_idx in disease_indices:
                            disease_idx = potential_idx
                    except (ValueError, IndexError):
                        pass
                
                if drug_idx is None or disease_idx is None:
                    if skipped_count < 5:  # Only print first few warnings
                        print(f"Warning: Skipping {drug_name} -> {disease_name} (not found in mappings)")
                    skipped_count += 1
                    continue
                
                drug_idx = int(drug_idx)
                disease_idx = int(disease_idx)
            else:
                # Last resort fallback: extract indices directly from generic names only
                print("Warning: No mappings found. Attempting to extract indices from names with 'Drug_' or 'Disease_' prefix.")
                try:
                    if drug_name.startswith('Drug_'):
                        drug_idx = int(drug_name.split('_')[1])
                    else:
                        print(f"Error: Cannot extract drug index from '{drug_name}' (must start with 'Drug_')")
                        skipped_count += 1
                        continue
                        
                    if disease_name.startswith('Disease_'):
                        disease_idx = int(disease_name.split('_')[1])
                    else:
                        print(f"Error: Cannot extract disease index from '{disease_name}' (must start with 'Disease_'")
                        skipped_count += 1
                        continue
                except (ValueError, IndexError) as e:
                    print(f"Error: Could not extract indices from {drug_name} -> {disease_name}: {e}")
                    skipped_count += 1
                    continue
            
            fp_pairs.append({
                'drug_name': drug_name,
                'disease_name': disease_name,
                'drug_idx': drug_idx,
                'disease_idx': disease_idx,
                'confidence': float(row[confidence_col])
            })
        
        if skipped_count > 5:
            print(f"Warning: Skipped {skipped_count} total predictions")
        
        print(f"Loaded {len(fp_pairs)} FP predictions from CSV")
        
    elif predictions_path.endswith('.pt'):
        predictions = torch.load(predictions_path, weights_only=False)
        
        fp_pairs = []
        skipped_count = 0
        
        for i, pred in enumerate(predictions):
            if len(pred) >= 3:
                drug_name, disease_name, confidence = pred[0], pred[1], float(pred[2])
                
                # Try to get actual indices from mappings first
                if drug_name_to_idx is not None and disease_name_to_idx is not None:
                    drug_idx = drug_name_to_idx.get(drug_name)
                    disease_idx = disease_name_to_idx.get(disease_name)
                    
                    # If direct lookup fails, try to extract index from generic name
                    if drug_idx is None and drug_name.startswith('Drug_'):
                        try:
                            potential_idx = int(drug_name.split('_')[1])
                            drug_indices = list(drug_name_to_idx.values())
                            if potential_idx in drug_indices:
                                drug_idx = potential_idx
                        except (ValueError, IndexError):
                            pass
                    
                    if disease_idx is None and disease_name.startswith('Disease_'):
                        try:
                            potential_idx = int(disease_name.split('_')[1])
                            disease_indices = list(disease_name_to_idx.values())
                            if potential_idx in disease_indices:
                                disease_idx = potential_idx
                        except (ValueError, IndexError):
                            pass
                    
                    if drug_idx is None or disease_idx is None:
                        if skipped_count < 5:
                            print(f"Warning: Skipping {drug_name} -> {disease_name} (not found in mappings)")
                        skipped_count += 1
                        continue
                    
                    drug_idx = int(drug_idx)
                    disease_idx = int(disease_idx)
                else:
                    # Last resort fallback
                    try:
                        if drug_name.startswith('Drug_'):
                            drug_idx = int(drug_name.split('_')[1])
                        else:
                            skipped_count += 1
                            continue
                            
                        if disease_name.startswith('Disease_'):
                            disease_idx = int(disease_name.split('_')[1])
                        else:
                            skipped_count += 1
                            continue
                    except (ValueError, IndexError):
                        skipped_count += 1
                        continue
                
                fp_pairs.append({
                    'drug_name': drug_name,
                    'disease_name': disease_name,
                    'drug_idx': drug_idx,
                    'disease_idx': disease_idx,
                    'confidence': confidence
                })
        
        if skipped_count > 5:
            print(f"Warning: Skipped {skipped_count} total predictions")
        
        print(f"Loaded {len(fp_pairs)} FP predictions from PyTorch file")
    
    else:
        raise ValueError(f"Unsupported file format: {predictions_path}")
    
    return fp_pairs


def calculate_hub_bias_analysis(explanations_dict, graph, idx_to_type=None):
    """Calculate hub bias using node attribution scores."""
    print("HUB BIAS ANALYSIS:")
    print("=" * 50)
    
    G = to_networkx(graph, to_undirected=True)
    
    all_scores = []
    all_degrees = []
    all_types = []
    
    for data in explanations_dict.values():
        if data['has_explanation'] and 'node_importance_scores' in data['explanation']:
            for node_idx, score in data['explanation']['node_importance_scores'].items():
                if node_idx in G and math.isfinite(score):
                    degree = G.degree(node_idx)
                    node_type = idx_to_type.get(node_idx, "Unknown") if idx_to_type else "Unknown"
                    
                    all_scores.append(score)
                    all_degrees.append(degree)
                    all_types.append(node_type)
    
    if len(all_scores) > 10:
        correlation, p_value = spearmanr(all_scores, all_degrees)
        
        print(f"Node attribution-degree correlation: ρ = {correlation:.3f}, p = {p_value:.3f}")
        print(f"Analysis based on {len(all_scores)} node attributions")
        
        if abs(correlation) < 0.1:
            print("Result: Minimal hub bias detected")
        elif correlation > 0.3:
            print("Result: Significant hub bias detected")
        else:
            print("Result: Moderate correlation with node degree")
    else:
        correlation = float('nan')
        p_value = float('nan')
        print("Result: Cannot assess hub bias - insufficient node attributions")
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'n_nodes': len(all_scores),
        'scores': all_scores,
        'degrees': all_degrees,
        'types': all_types
    }


def calculate_type_importance_analysis(explanations_dict, graph, idx_to_type=None):
    """Calculate importance by node type using actual attribution scores."""
    print("\nTYPE IMPORTANCE ANALYSIS:")
    print("=" * 50)
    
    node_type_scores = defaultdict(list)
    
    for data in explanations_dict.values():
        if data['has_explanation'] and 'node_importance_scores' in data['explanation']:
            for node_idx, score in data['explanation']['node_importance_scores'].items():
                node_type = idx_to_type.get(node_idx, "Unknown") if idx_to_type else "Unknown"
                if math.isfinite(score):
                    node_type_scores[node_type].append(score)
    
    print("Node Type Attribution Statistics:")
    node_stats = {}
    for node_type, scores in node_type_scores.items():
        if scores:
            mean_score = np.mean(scores)
            median_score = np.median(scores)
            std_score = np.std(scores)
            max_score = np.max(scores)
            min_score = np.min(scores)
            
            node_stats[node_type] = {
                'mean': mean_score,
                'median': median_score,
                'std': std_score,
                'max': max_score,
                'min': min_score,
                'count': len(scores)
            }
            
            print(f"  {node_type}: mean={mean_score:.4f}, median={median_score:.4f}, "
                  f"std={std_score:.4f}, range=[{min_score:.4f}, {max_score:.4f}] ({len(scores)} nodes)")
    
    return {'node_type_stats': node_stats}


def analyze_edge_types(explanations_dict, graph, idx_to_type=None):
    """
    Analyze which edge types are most important in predictions.
    Classifies edges by the types of nodes they connect.
    """
    print("\nEDGE TYPE ANALYSIS:")
    print("=" * 50)
    
    edge_type_importance = defaultdict(list)
    edge_type_counts = defaultdict(int)
    
    for data in explanations_dict.values():
        if data['has_explanation'] and 'important_edges' in data['explanation']:
            for edge in data['explanation']['important_edges']:
                source_idx = edge['source']
                target_idx = edge['target']
                importance = edge['importance']
                
                # Get node types
                source_type = idx_to_type.get(source_idx, "Unknown") if idx_to_type else "Unknown"
                target_type = idx_to_type.get(target_idx, "Unknown") if idx_to_type else "Unknown"
                
                # Create edge type label (alphabetically sorted for consistency)
                edge_type = f"{min(source_type, target_type)}-{max(source_type, target_type)}"
                
                if math.isfinite(importance):
                    edge_type_importance[edge_type].append(importance)
                    edge_type_counts[edge_type] += 1
    
    # Calculate statistics for each edge type
    edge_type_stats = {}
    print("Edge Type Importance Statistics:")
    
    for edge_type in sorted(edge_type_importance.keys(), 
                           key=lambda x: np.mean(edge_type_importance[x]), 
                           reverse=True):
        scores = edge_type_importance[edge_type]
        
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        std_score = np.std(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        count = len(scores)
        
        edge_type_stats[edge_type] = {
            'mean': mean_score,
            'median': median_score,
            'std': std_score,
            'max': max_score,
            'min': min_score,
            'count': count
        }
        
        print(f"  {edge_type}: mean={mean_score:.4f}, median={median_score:.4f}, "
              f"count={count}, range=[{min_score:.4f}, {max_score:.4f}]")
    
    # Identify most important edge types
    if edge_type_stats:
        top_edge_type = max(edge_type_stats.items(), key=lambda x: x[1]['mean'])
        print(f"\nMost important edge type: {top_edge_type[0]} (mean importance: {top_edge_type[1]['mean']:.4f})")
    
    return {
        'edge_type_stats': edge_type_stats,
        'edge_type_counts': dict(edge_type_counts)
    }


def real_faithfulness_test(explanations_dict, model, graph, n_tests=50):
    """Real faithfulness test with actual performance measurement."""
    print("\nREAL FAITHFULNESS VALIDATION:")
    print("=" * 50)
    
    performance_drops_attributed = []
    performance_drops_random = []
    
    valid_explanations = [(k, v) for k, v in explanations_dict.items() 
                         if v['has_explanation']]
    test_explanations = valid_explanations[:n_tests]
    
    link_predictor = LinkPredictor(model)
    link_predictor.eval()
    
    for i, (pair_key, exp_data) in enumerate(test_explanations):
        if i % 10 == 0:
            print(f"   Testing faithfulness {i+1}/{len(test_explanations)}")
        
        explanation = exp_data['explanation']
        drug_idx = explanation['drug_idx']
        disease_idx = explanation['disease_idx']
        
        # Set target edge
        link_predictor.set_target_edge(drug_idx, disease_idx)
        
        # Get original prediction
        with torch.no_grad():
            original_logit = link_predictor(graph.x, graph.edge_index)
            original_pred = torch.sigmoid(original_logit).item()
        
        # Remove top attributed nodes
        node_scores = explanation.get('node_importance_scores', {})
        if len(node_scores) == 0:
            continue
        
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        num_to_remove = max(1, len(sorted_nodes) // 5)
        top_nodes = [node_idx for node_idx, _ in sorted_nodes[:num_to_remove]]
        
        # Test with attributed nodes removed
        modified_x = graph.x.clone()
        modified_x[top_nodes] = 0
        
        with torch.no_grad():
            modified_logit = link_predictor(modified_x, graph.edge_index)
            modified_pred_attributed = torch.sigmoid(modified_logit).item()
        
        # Test with random nodes removed
        all_nodes = list(node_scores.keys())
        random_nodes = np.random.choice(all_nodes, size=len(top_nodes), replace=False)
        
        modified_x_random = graph.x.clone()
        modified_x_random[random_nodes] = 0
        
        with torch.no_grad():
            modified_logit_random = link_predictor(modified_x_random, graph.edge_index)
            modified_pred_random = torch.sigmoid(modified_logit_random).item()
        
        # Calculate performance drops
        drop_attributed = abs(original_pred - modified_pred_attributed)
        drop_random = abs(original_pred - modified_pred_random)
        
        performance_drops_attributed.append(drop_attributed)
        performance_drops_random.append(drop_random)
    
    # Statistical test
    if len(performance_drops_attributed) > 5:
        try:
            stat, p_value = wilcoxon(performance_drops_attributed, 
                                    performance_drops_random, 
                                    alternative='greater')
        except ValueError:
            p_value = 1.0
        
        print(f"Mean performance drop (attributed nodes): {np.mean(performance_drops_attributed):.3f}")
        print(f"Mean performance drop (random nodes): {np.mean(performance_drops_random):.3f}")
        print(f"Wilcoxon test p-value: {p_value:.3f}")
        
        if p_value < 0.05:
            print("RESULT: FAITHFUL - Removing attributed nodes hurts performance significantly more")
        else:
            print("RESULT: NOT FAITHFUL - No significant difference")
    else:
        print("WARNING: Insufficient data for statistical test")
        p_value = 1.0
    
    return {
        'attributed_drops': performance_drops_attributed,
        'random_drops': performance_drops_random,
        'p_value': p_value
    }


def create_visualizations(results, output_dir):
    """Create visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Hub bias plot
    hub_data = results.get('hub_bias_results', {})
    if hub_data and len(hub_data.get('scores', [])) > 0:
        scores = hub_data['scores']
        degrees = hub_data['degrees']
        types = hub_data['types']
        correlation = hub_data['correlation']
        p_value = hub_data['p_value']
        
        plt.figure(figsize=(10, 6))
        
        type_colors = {'Drug': '#1f77b4', 'Disease': '#ff7f0e', 'Gene': '#2ca02c', 
                       'Pathway': '#17becf', 'DrugType': '#9467bd', 'TherapeuticArea': '#8c564b'}
        
        for node_type in set(types):
            mask = [nt == node_type for nt in types]
            plt.scatter([d for d, m in zip(degrees, mask) if m], 
                       [s for s, m in zip(scores, mask) if m],
                       c=type_colors.get(node_type, '#999999'), 
                       label=node_type, alpha=0.6, s=20)
        
        plt.xlabel('Node Degree')
        plt.ylabel('GNNExplainer Attribution Score')
        plt.title(f'Hub Bias Analysis: Attribution vs Node Degree\n(Spearman ρ = {correlation:.3f}, p = {p_value:.3f}, n = {len(scores)})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_dir / 'hub_bias_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Faithfulness plot
    faithfulness_data = results.get('faithfulness_results', {})
    if faithfulness_data and len(faithfulness_data.get('attributed_drops', [])) > 0:
        attributed_drops = faithfulness_data['attributed_drops']
        random_drops = faithfulness_data['random_drops']
        p_value = faithfulness_data['p_value']
        
        plt.figure(figsize=(8, 6))
        
        data_to_plot = [attributed_drops, random_drops]
        labels = ['Attributed Nodes\nRemoved', 'Random Nodes\nRemoved']
        
        box_plot = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('#ff7f0e')
        box_plot['boxes'][1].set_facecolor('#1f77b4')
        
        plt.ylabel('Model Performance Drop')
        plt.title(f'Faithfulness Validation: Performance Drop After Node Removal\n(p = {p_value:.3f})')
        
        # Add significance markers
        if p_value < 0.001:
            plt.text(1.5, max(max(attributed_drops), max(random_drops)) * 1.1, '***', ha='center', fontsize=16)
        elif p_value < 0.01:
            plt.text(1.5, max(max(attributed_drops), max(random_drops)) * 1.1, '**', ha='center', fontsize=16)
        elif p_value < 0.05:
            plt.text(1.5, max(max(attributed_drops), max(random_drops)) * 1.1, '*', ha='center', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'faithfulness_test.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def save_results(results, output_dir):
    """Save results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Save main results (excluding non-serializable objects)
    results_to_save = {k: v for k, v in results.items() if k != 'explainer'}
    
    results_file = output_dir / f'gnn_explainer_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)
        
    print(f"Results saved to {results_file}")


def create_node_type_mapping(graph):
    """Create a basic node type mapping based on node indices."""
    # This is a simplified mapping - in practice you'd use actual mappings
    idx_to_type = {}
    
    # Basic heuristic based on node indices
    num_nodes = graph.x.size(0)
    
    # Assume first 1000 nodes are drugs
    for i in range(min(1000, num_nodes)):
        idx_to_type[i] = "Drug"
    
    # Next 100 are drug types
    for i in range(1000, min(1100, num_nodes)):
        idx_to_type[i] = "DrugType"
    
    # Next 3000 are genes
    for i in range(1100, min(4100, num_nodes)):
        idx_to_type[i] = "Gene"
    
    # Next 500 are pathways
    for i in range(4100, min(4600, num_nodes)):
        idx_to_type[i] = "Pathway"
    
    # Remaining are diseases and therapeutic areas
    for i in range(4600, num_nodes):
        if i < num_nodes - 100:
            idx_to_type[i] = "Disease"
        else:
            idx_to_type[i] = "TherapeuticArea"
    
    return idx_to_type


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='GNN Explainer for drug-disease predictions')
    parser.add_argument('--graph', type=str, help='Path to graph file (.pt)')
    parser.add_argument('--model', type=str, help='Path to trained model (.pt)')
    parser.add_argument('--predictions', type=str, help='Path to FP predictions (CSV or PT)')
    parser.add_argument('--output-dir', type=str, default='results/explainer/', help='Output directory')
    parser.add_argument('--max-explanations', type=int, default=1000, help='Maximum explanations to generate')
    parser.add_argument('--mappings-dir', type=str, help='Path to directory containing drug/disease mappings')
    
    args = parser.parse_args()
    
    # Load configuration from config.py
    config = get_config()
    
    explainer_config = config.get_explainer_config()
    
    # Set reproducibility
    enable_full_reproducibility(42)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize MLflow tracking
    tracker = ExperimentTracker(
        experiment_name="GNN_Explainer_Analysis",
        tracking_uri="./mlruns"
    )
    
    # Start MLflow run (not as context manager)
    tracker.start_run(run_name=f"explainer_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    try:
        # Log parameters
        tracker.log_param('graph_path', args.graph)
        tracker.log_param('model_path', args.model)
        tracker.log_param('predictions_path', args.predictions)
        tracker.log_param('max_explanations', args.max_explanations)
        tracker.log_param('explainer_epochs', explainer_config.get('epochs', 50))
        tracker.log_param('device', str(device))
        
        # Load graph
        if args.graph:
            graph_path = args.graph
        else:
            graph_path = find_latest_graph()
        
        print(f"Loading graph from {graph_path}")
        graph = torch.load(graph_path, map_location=device, weights_only=False)
        print(f"Loaded graph: {graph.x.size(0)} nodes, {graph.edge_index.size(1)} edges")
        
        tracker.log_param('num_nodes', graph.x.size(0))
        tracker.log_param('num_edges', graph.edge_index.size(1))
        tracker.log_param('num_features', graph.x.size(1))
        
        # Detect model type and load
        if args.model:
            model_path = args.model
        else:
            model_path = find_latest_model()
        
        model_filename = Path(model_path).name.lower()
        
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
            print(f"Warning: Cannot detect model type from {model_filename}, defaulting to TransformerModel")
            model_class = TransformerModel
            model_name = 'Transformer'
        
        print(f"Loading {model_name} model from {model_path}")
        tracker.log_param('model_type', model_name)
        
        # Get model configuration from config
        model_config = config.get_model_config()
        
        model = model_class(
            in_channels=graph.x.size(1),
            hidden_channels=model_config['hidden_channels'],
            out_channels=model_config['out_channels'],
            num_layers=model_config['num_layers'],
            dropout_rate=model_config['dropout_rate']
        ).to(device)
        
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.eval()
        print("Model loaded successfully")
        
        # Load FP predictions with proper mappings
        mappings_dir = args.mappings_dir
        if not mappings_dir:
            # Try to auto-detect mappings directory relative to workspace
            workspace_root = Path(__file__).parent.parent
            potential_mappings_dir = workspace_root / "processed_data" / "mappings"
            if potential_mappings_dir.exists():
                mappings_dir = str(potential_mappings_dir)
                print(f"Auto-detected mappings directory: {mappings_dir}")
            else:
                print("Warning: Could not auto-detect mappings directory")
        
        if args.predictions:
            predictions_path = args.predictions
        else:
            predictions_path = find_latest_predictions()
        
        fp_pairs = load_fp_predictions(predictions_path, mappings_dir=mappings_dir)
        print(f"Loaded {len(fp_pairs)} false positive predictions")
        tracker.log_metric('num_fp_predictions', len(fp_pairs))
        
        # Initialize GNNExplainer
        print("Initializing GNNExplainer...")
        analyzer = GNNExplainerAnalyzer(model, graph, device, config=explainer_config)
        
        # Generate explanations
        start_time = time.time()
        explanations_dict = analyzer.explain_multiple_predictions(
            fp_pairs, 
            max_explanations=args.max_explanations
        )
        explanation_time = time.time() - start_time
        
        tracker.log_metric('explanation_time_seconds', explanation_time)
        tracker.log_metric('explanations_per_second', len(explanations_dict) / explanation_time if explanation_time > 0 else 0)
        
        # Create node type mapping (simplified - in practice use actual mappings)
        idx_to_type = create_node_type_mapping(graph)
        
        # Run analysis
        hub_bias_results = calculate_hub_bias_analysis(explanations_dict, graph, idx_to_type)
        type_importance_results = calculate_type_importance_analysis(explanations_dict, graph, idx_to_type)
        edge_type_results = analyze_edge_types(explanations_dict, graph, idx_to_type)
        faithfulness_results = real_faithfulness_test(explanations_dict, model, graph, n_tests=50)
        
        # Compile results
        results = {
            'config': explainer_config,
            'n_explanations': len(explanations_dict),
            'n_successful': len([e for e in explanations_dict.values() if e['has_explanation']]),
            'hub_bias_results': hub_bias_results,
            'type_importance_results': type_importance_results,
            'edge_type_results': edge_type_results,
            'faithfulness_results': faithfulness_results,
            'timestamp': dt.datetime.now().isoformat()
        }
        
        # Log key metrics to MLflow
        tracker.log_metric('n_explanations', results['n_explanations'])
        tracker.log_metric('n_successful_explanations', results['n_successful'])
        tracker.log_metric('success_rate', results['n_successful'] / results['n_explanations'] if results['n_explanations'] > 0 else 0)
        
        # Log hub bias metrics
        if hub_bias_results and 'correlation' in hub_bias_results and math.isfinite(hub_bias_results['correlation']):
            tracker.log_metric('hub_bias_correlation', hub_bias_results['correlation'])
            tracker.log_metric('hub_bias_p_value', hub_bias_results['p_value'])
            tracker.log_metric('hub_bias_n_nodes', hub_bias_results['n_nodes'])
        
        # Log faithfulness metrics
        if faithfulness_results and 'p_value' in faithfulness_results:
            tracker.log_metric('faithfulness_p_value', faithfulness_results['p_value'])
            if len(faithfulness_results['attributed_drops']) > 0:
                tracker.log_metric('mean_attributed_drop', np.mean(faithfulness_results['attributed_drops']))
                tracker.log_metric('mean_random_drop', np.mean(faithfulness_results['random_drops']))
        
        # Log node type importance
        if type_importance_results and 'node_type_stats' in type_importance_results:
            for node_type, stats in type_importance_results['node_type_stats'].items():
                tracker.log_metric(f'node_importance_{node_type}_mean', stats['mean'])
                tracker.log_metric(f'node_importance_{node_type}_count', stats['count'])
        
        # Log edge type importance
        if edge_type_results and 'edge_type_stats' in edge_type_results:
            for edge_type, stats in edge_type_results['edge_type_stats'].items():
                safe_edge_type = edge_type.replace('-', '_')
                tracker.log_metric(f'edge_importance_{safe_edge_type}_mean', stats['mean'])
                tracker.log_metric(f'edge_importance_{safe_edge_type}_count', stats['count'])
        
        # Save and visualize
        create_visualizations(results, args.output_dir)
        save_results(results, args.output_dir)
        
        # Log visualizations as artifacts
        output_path = Path(args.output_dir)
        if (output_path / 'hub_bias_analysis.png').exists():
            tracker.log_artifact(str(output_path / 'hub_bias_analysis.png'))
        if (output_path / 'faithfulness_test.png').exists():
            tracker.log_artifact(str(output_path / 'faithfulness_test.png'))
        
        # Log results JSON as artifact
        results_files = list(output_path.glob('gnn_explainer_results_*.json'))
        if results_files:
            tracker.log_artifact(str(results_files[-1]))
        
        print("\n" + "="*60)
        print("EXPLAINER ANALYSIS COMPLETE")
        print("="*60)
        print(f"Total explanations: {results['n_explanations']}")
        print(f"Successful: {results['n_successful']}")
        print(f"Results saved to: {args.output_dir}")
        print(f"MLflow run completed")
    
    finally:
        # Always end the run, even if there's an error
        tracker.end_run()


if __name__ == "__main__":
    main()
