#!/usr/bin/env python3
"""
Leave-One-Out Validation Script for Drug-Disease Prediction

This script performs rigorous leave-one-out cross-validation by:
1. Removing known drug-disease edges from the graph
2. Optionally removing the drug or disease node entirely (not just edges)
3. Retraining the model on the modified graph
4. Testing if the model can recover the held-out associations

Uses APR (Average Precision) as the primary evaluation metric.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import datetime as dt
import argparse
import json
import sys
import os
import glob
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
import copy

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.negative_sampling import get_sampler
from src.training.losses import get_loss_function
from src.utils.common import set_seed
from src.models import GCNModel, SAGEModel, TransformerModel, GATModel, MODEL_CLASSES, LinkPredictor
from src.features.heuristic_scores import compute_heuristic_edge_features
from src.training.tracker import ExperimentTracker
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve


def find_latest_graph(results_dir='results'):
    """Auto-detect the latest graph file."""
    graph_files = glob.glob(f'{results_dir}/graph_*.pt')
    if not graph_files:
        raise FileNotFoundError(f"No graph files found in {results_dir}")
    latest = max(graph_files, key=os.path.getctime)
    print(f"  Auto-detected: {latest}")
    return latest




class LeaveOneOutValidator:
    """
    Leave-One-Out Cross-Validation for Drug Repurposing.
    
    Supports two validation modes:
    1. Edge removal only: Remove drug-disease edge, keep nodes
    2. Node removal: Remove the drug OR disease node entirely (and all its edges)
    """
    
    def __init__(self, config, removal_mode='edge', remove_node_type='drug', 
                 num_folds=None, epochs_per_fold=50, device=None):
        """
        Args:
            config: Configuration object
            removal_mode: 'edge' (remove only the drug-disease edge) or 
                         'node' (remove entire node and all its connections)
            remove_node_type: When removal_mode='node', which node to remove: 'drug' or 'disease'
            num_folds: Number of drug-disease pairs to test (None = all)
            epochs_per_fold: Training epochs for each fold
            device: torch device
        """
        self.config = config
        self.removal_mode = removal_mode
        self.remove_node_type = remove_node_type
        self.num_folds = num_folds
        self.epochs_per_fold = epochs_per_fold
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Results storage
        self.results = []
        self.fold_metrics = []
        
        print(f"\n{'='*80}")
        print(f"LEAVE-ONE-OUT VALIDATION")
        print(f"{'='*80}")
        print(f"Removal mode: {removal_mode}")
        if removal_mode == 'node':
            print(f"Node type to remove: {remove_node_type}")
        print(f"Device: {self.device}")
        print(f"Epochs per fold: {epochs_per_fold}")
        print(f"{'='*80}\n")
    
    def find_latest_graph(self, processed_path: Path) -> Path:
        """Find the latest graph file in the processed directory."""
        graph_files = glob.glob(str(processed_path / 'graph_*.pt'))
        if not graph_files:
            raise FileNotFoundError("No graph files found in the processed directory.")
        latest_graph = max(graph_files, key=os.path.getctime)
        return Path(latest_graph)
    
    def load_graph_data(self):
        """Load the base graph and mappings."""
        print("Loading graph data...")
        
        results_path = Path(self.config.paths['results'])
        
        # Load graph
        latest_graph_path = self.find_latest_graph(results_path)
        self.graph = torch.load(latest_graph_path, weights_only=False)
        print(f"  Graph: {self.graph.num_nodes} nodes, {self.graph.edge_index.shape[1]} edges")
        
        # --- FEATURE ABLATION ---
        ablation_mode = getattr(self.config, 'feature_ablation', None)
        if ablation_mode and ablation_mode not in ['none', 'None', 'standard', 'original']:
            print(f"\n[!!] FEATURE ABLATION ACTIVE: {ablation_mode}")
            
            num_nodes = self.graph.num_nodes
            
            feat_dim = self.graph.num_node_features
            
            if ablation_mode == 'constant':
                print(f"  Replacing {feat_dim}-dim features with ONES")
                self.graph.x = torch.ones((num_nodes, feat_dim))
                
            elif ablation_mode == 'random':
                print(f"  Replacing {feat_dim}-dim features with RANDOM noise")
                self.graph.x = torch.randn((num_nodes, feat_dim))
                
            elif ablation_mode == 'degree':
                print(f"  Replacing features with DEGREE ONE-HOT (max_degree=100)")
                # Calculate degrees
                from torch_geometric.utils import degree
                d = degree(self.graph.edge_index[0], num_nodes=num_nodes).long()
                # Cap at 100 to keep dimension reasonable
                d = d.clamp(max=99)
                self.graph.x = torch.nn.functional.one_hot(d, num_classes=100).float()
                # Update input dimension references
                self.config.model_config['in_channels'] = 100
                print(f"  New feature dimension: {self.graph.num_node_features}")
                
            elif ablation_mode == 'one_hot_id':
                # This is risky for memory but theoretically useful
                # Only use for small graphs
                if num_nodes > 10000:
                    print("  WARNING: one_hot_id requested but graph too large. Falling back to constant.")
                    self.graph.x = torch.ones((num_nodes, feat_dim))
                else:
                    print("  Replacing features with NODE ID ONE-HOT")
                    self.graph.x = torch.eye(num_nodes)
                    self.config.model_config['in_channels'] = num_nodes
            
            else:
                print(f"  Unknown ablation mode '{ablation_mode}', using original features.")
        
        # Load mappings
        processed_data_path = Path('processed_data')
        
        # Check if mapings exist next to the graph (preferred)
        graph_mappings_dir = str(latest_graph_path).replace('.pt', '_mappings')
        
        if os.path.isdir(graph_mappings_dir):
            print(f"Loading graph-specific mappings from: {graph_mappings_dir}")
            mappings_path = Path(graph_mappings_dir)
        else:
            # Fallback: Use global processed_data/mappings (risky if graph was subsetted)
            mappings_path = processed_data_path / 'mappings'
            print(f"Loading global mappings from: {mappings_path}")
            print("WARNING: Global mappings may not match graph indices if nodes were filtered.")
        
        # Load JSON mappings
        import json
        with open(mappings_path / 'drug_key_mapping.json', 'r') as f:
            self.drug_key_mapping = json.load(f)
        with open(mappings_path / 'disease_key_mapping.json', 'r') as f:
            self.disease_key_mapping = json.load(f)
        with open(mappings_path / 'gene_key_mapping.json', 'r') as f:
            self.gene_key_mapping = json.load(f)
        
        # Create reverse mappings
        self.idx_to_drug = {v: k for k, v in self.drug_key_mapping.items()}
        self.idx_to_disease = {v: k for k, v in self.disease_key_mapping.items()}
        self.idx_to_gene = {v: k for k, v in self.gene_key_mapping.items()}
        
        # Extract drug-disease edges from graph itself
        print("  Extracting drug-disease edges from graph...")
        
        # Create masks for efficient filtering
        num_nodes = self.graph.num_nodes
        is_drug = torch.zeros(num_nodes, dtype=torch.bool)
        is_disease = torch.zeros(num_nodes, dtype=torch.bool)
        
        # Mark drug and disease nodes
        # Note: We use the values from mappings which are guaranteed to match the graph indices
        drug_indices = list(self.drug_key_mapping.values())
        disease_indices = list(self.disease_key_mapping.values())
        
        is_drug[drug_indices] = True
        is_disease[disease_indices] = True
        
        # --- EDGE ABLATION ---
        edge_ablation = getattr(self.config, 'edge_ablation', None)
        if edge_ablation and edge_ablation not in ['none', 'None', 'standard', 'original']:
            print(f"\n[!!] EDGE ABLATION ACTIVE: {edge_ablation}")
            
            src, dst = self.graph.edge_index[0], self.graph.edge_index[1]
            mask = torch.ones(src.size(0), dtype=torch.bool)
            
            # Create gene mask
            gene_indices = list(self.gene_key_mapping.values())
            is_gene = torch.zeros(self.graph.num_nodes, dtype=torch.bool)
            is_gene[gene_indices] = True
            
            if edge_ablation == 'no_ppi' or edge_ablation == 'no_gene_gene':
                # Remove Gene-Gene edges
                # Both src and dst are genes
                gg_mask = is_gene[src] & is_gene[dst]
                print(f"  Removing {gg_mask.sum().item()} Gene-Gene edges (PPI)")
                mask = mask & (~gg_mask)
                
            elif edge_ablation == 'no_disease_struct' or edge_ablation == 'no_disease_disease':
                # Remove Disease-Disease edges
                # Both src and dst are diseases
                dd_mask = is_disease[src] & is_disease[dst]
                print(f"  Removing {dd_mask.sum().item()} Disease-Disease edges")
                mask = mask & (~dd_mask)
                
            elif edge_ablation == 'no_drug_target':
                # Remove Drug-Gene and Gene-Drug edges
                dg_mask = (is_drug[src] & is_gene[dst]) | (is_gene[src] & is_drug[dst])
                print(f"  Removing {dg_mask.sum().item()} Drug-Gene edges")
                mask = mask & (~dg_mask)
            
            elif edge_ablation == 'no_disease_gene':
                # Remove Disease-Gene and Gene-Disease edges
                # Note: This removes association scores too
                dis_g_mask = (is_disease[src] & is_gene[dst]) | (is_gene[src] & is_disease[dst])
                print(f"  Removing {dis_g_mask.sum().item()} Disease-Gene edges")
                mask = mask & (~dis_g_mask)
                
            # Apply mask
            self.graph.edge_index = self.graph.edge_index[:, mask]
            if self.graph.edge_attr is not None:
                self.graph.edge_attr = self.graph.edge_attr[mask]
            
            print(f"  Edges remaining: {self.graph.edge_index.shape[1]}")
            
            # Handle feature score ablation separately
            if edge_ablation == 'no_scores':
                if self.graph.edge_attr is not None and self.graph.edge_attr.shape[1] > 6:
                    print("  Zeroing out association scores in edge attributes (col 7)")
                    pass 
        
        # Refetch edge_index after potential modification
        edge_index = self.graph.edge_index
        src, dst = edge_index[0], edge_index[1]
        
        # Find edges where (Src is Drug and Dst is Disease)
        mask = is_drug[src] & is_disease[dst]
        
        filtered_edges = edge_index[:, mask]
        self.drug_disease_edges = filtered_edges
        
        print(f"  Drug-disease edges: {self.drug_disease_edges.shape[1]}")
        
        # Get unique drugs and diseases with connections
        self.connected_drugs = set(self.drug_disease_edges[0].numpy())
        self.connected_diseases = set(self.drug_disease_edges[1].numpy())
        print(f"  Connected drugs: {len(self.connected_drugs)}")
        print(f"  Connected diseases: {len(self.connected_diseases)}")
        
        # Create edge set for fast lookup
        self.positive_edge_set = set(
            zip(self.drug_disease_edges[0].numpy(), 
                self.drug_disease_edges[1].numpy())
        )
        
    def get_all_edges_for_node(self, node_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find all edges connected to a specific node.
        
        Returns:
            mask: Boolean mask of edges to remove
            edge_indices: Indices of edges to remove
        """
        edge_index = self.graph.edge_index
        
        # Find edges where node appears as source or target
        source_mask = edge_index[0] == node_idx
        target_mask = edge_index[1] == node_idx
        edge_mask = source_mask | target_mask
        
        return edge_mask, torch.where(edge_mask)[0]
    
    def remove_edge(self, graph: torch.Tensor, drug_idx: int, disease_idx: int) -> torch.Tensor:
        """Remove a specific drug-disease edge from the graph."""
        edge_index = graph.edge_index
        
        # Find and remove the edge (both directions for undirected graph)
        mask1 = ~((edge_index[0] == drug_idx) & (edge_index[1] == disease_idx))
        mask2 = ~((edge_index[0] == disease_idx) & (edge_index[1] == drug_idx))
        mask = mask1 & mask2
        
        new_edge_index = edge_index[:, mask]
        
        # Create modified graph
        modified_graph = copy.copy(graph)
        modified_graph.edge_index = new_edge_index
        
        return modified_graph
    
    def remove_node(self, graph: torch.Tensor, node_idx: int) -> Tuple[torch.Tensor, Dict[int, int]]:
        """
        Remove a node and all its edges from the graph.
        
        Returns:
            modified_graph: Graph with node removed
            idx_mapping: Mapping from old indices to new indices
        """
        edge_index = graph.edge_index
        x = graph.x
        num_nodes = graph.num_nodes
        
        # Remove all edges connected to this node
        source_mask = edge_index[0] != node_idx
        target_mask = edge_index[1] != node_idx
        edge_mask = source_mask & target_mask
        new_edge_index = edge_index[:, edge_mask]
        
        # Create index mapping (nodes after removed node shift down by 1)
        idx_mapping = {}
        for old_idx in range(num_nodes):
            if old_idx < node_idx:
                idx_mapping[old_idx] = old_idx
            elif old_idx > node_idx:
                idx_mapping[old_idx] = old_idx - 1
            # Removed node doesn't get a mapping
        
        # Update edge indices
        def remap_idx(idx):
            if idx < node_idx:
                return idx
            elif idx > node_idx:
                return idx - 1
            else:
                return -1  # Should not happen after filtering
        
        new_edge_index[0] = torch.tensor([remap_idx(i.item()) for i in new_edge_index[0]])
        new_edge_index[1] = torch.tensor([remap_idx(i.item()) for i in new_edge_index[1]])
        
        # Remove node features
        new_x = torch.cat([x[:node_idx], x[node_idx+1:]], dim=0)
        
        # Create modified graph
        modified_graph = copy.copy(graph)
        modified_graph.edge_index = new_edge_index
        modified_graph.x = new_x
        modified_graph.num_nodes = num_nodes - 1
        
        return modified_graph, idx_mapping
    
    def create_model(self, in_channels: int, hidden_channels: int = None) -> torch.nn.Module:
        """Create a fresh model instance with link prediction capability using Config."""
        model_choice = self.config.model_choice
        model_config = self.config.model_config
        
        # Use config hidden_channels if not overridden
        if hidden_channels is None:
            hidden_channels = model_config.get('hidden_channels', 64)
        
        # Get the encoder model class
        ModelClass = MODEL_CLASSES.get(model_choice, TransformerModel)
        
        # Standardise args from config
        kwargs = {
            'in_channels': in_channels, 
            'hidden_channels': hidden_channels, 
            'out_channels': model_config.get('out_channels', 32),
            'num_layers': model_config.get('num_layers', 2),
            'dropout_rate': model_config.get('dropout_rate', 0.5)
        }
        
        # Add model-specific args
        if model_choice in ['Transformer', 'TransformerModel']:
            kwargs['heads'] = model_config.get('heads', 2)
            kwargs['concat'] = model_config.get('concat', True)
            
            # Check for edge features
            if hasattr(self, 'graph') and hasattr(self.graph, 'edge_attr') and self.graph.edge_attr is not None:
                kwargs['edge_dim'] = self.graph.edge_attr.size(1)
        
        # Create encoder
        # Explicit initialisation for Transformer to ensure edge_dim is passed correctly
        if ModelClass.__name__ == 'TransformerModel':
             encoder = TransformerModel(
                 in_channels=kwargs['in_channels'],
                 hidden_channels=kwargs['hidden_channels'],
                 out_channels=kwargs['out_channels'],
                 num_layers=kwargs['num_layers'],
                 dropout_rate=kwargs['dropout_rate'],
                 heads=kwargs.get('heads', 4),
                 concat=kwargs.get('concat', False),
                 edge_dim=kwargs.get('edge_dim')
             )
        else:
             encoder = ModelClass(**kwargs)
        
        # Wrap with link predictor
        # Use MLP decoder with configured type
        # hidden_channels for decoder = out_channels of encoder (the embedding dimension)
        out_channels = kwargs['out_channels']
        decoder_type = model_config.get('decoder_type', 'mlp_interaction')
        model = LinkPredictor(encoder, hidden_channels=out_channels, decoder_type=decoder_type, num_neighbor_features=3)
        
        return model.to(self.device)
    
    def train_model(self, model: torch.nn.Module, graph: torch.Tensor, 
                    train_edges: torch.Tensor, train_labels: torch.Tensor, log_metrics_file: str = None) -> torch.nn.Module:
        """Train model on the modified graph using Config settings."""
        model.train()
        
        # Optimiser from config
        lr = self.config.model_config.get('learning_rate', 0.001)
        weight_decay = self.config.model_config.get('weight_decay', 1e-5)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Loss from config
        loss_config = self.config.get_loss_config()
        try:
            criterion = get_loss_function(loss_config['loss_function'], **loss_config['params'])
            print(f"  Using loss function: {loss_config['loss_function']}")
        except Exception as e:
            print(f"  Warning: Failed to load custom loss {loss_config['loss_function']} ({e}). Using standard BCE.")
            criterion = torch.nn.BCEWithLogitsLoss()
        
        graph = graph.to(self.device)
        train_edges = train_edges.to(self.device)
        train_labels = train_labels.to(self.device)
        
        print(f"  Training on {train_edges.shape[1]} edges (Positive: {torch.sum(train_labels==1)}, Negative: {torch.sum(train_labels==0)})")
        
        # Precompute heuristic features for training edges (Phase 1)
        print("  Computing heuristic features for training edges...")
        # We need a CPU copy of graph for heuristic computation
        graph_cpu = graph.cpu()
        train_edges_cpu = train_edges.cpu()
        # Do not normalise, as max values differ between train/inference sets
        heuristic_feats = compute_heuristic_edge_features(graph_cpu, train_edges_cpu, normalise=False)
        heuristic_feats_gpu = heuristic_feats.to(self.device)
        print(f"  Heuristic features shape: {heuristic_feats.shape} (CN, AA, Jaccard)")
        
        if log_metrics_file:
            # Initialise CSV
            with open(log_metrics_file, 'w') as f:
                f.write("epoch,train_loss,val_rank_median,val_hits_20\n")

        pbar = tqdm(range(self.epochs_per_fold), desc="Training", leave=False)
        for epoch in pbar:
            model.train() # Ensure train mode
            optimiser.zero_grad()
            
            # Forward pass
            edge_attr = getattr(graph, 'edge_attr', None)
            z = model.encode(graph.x, graph.edge_index, edge_attr=edge_attr)
            
            # Use precomputed heuristics
            # We assume training edges don't change, so heuristics computed once at start are valid
            pass # Already computed
            
            pred = model.decode(z, train_edges, heuristic_features=heuristic_feats_gpu)
            
            # Loss
            try:
                # train_edges[1] contains target node indices (groups/diseases)
                loss = criterion(pred, train_labels.float(), groups=train_edges[1])
            except TypeError:
                loss = criterion(pred, train_labels.float())
            
            # Backward pass
            loss.backward()
            optimiser.step()
            
            current_loss = loss.item()
            pbar.set_postfix({'loss': f"{current_loss:.4f}"})
            
            # --- Diagnostic Logging ---
            if log_metrics_file and (epoch % 5 == 0 or epoch == self.epochs_per_fold - 1):
                # Run fast validation on target node if available
                if hasattr(self, 'target_node_idx') and self.target_node_idx is not None:
                    # Very simple inference: only rank connected drugs
                    try:
                        val_metrics = self.run_fast_validation_snapshot(model, z, self.target_node_idx)
                        with open(log_metrics_file, 'a') as f:
                            f.write(f"{epoch},{current_loss:.6f},{val_metrics['median_rank']},{val_metrics['hits_20']}\n")
                    except Exception as e:
                        print(f"Validation snapshot failed: {e}")
            # --------------------------
        
        return model
    
    def run_fast_validation_snapshot(self, model: torch.nn.Module, z: torch.Tensor, target_node_idx: int) -> Dict:
        """
        Run a quick validation snapshot during training for diagnostics.
        Only ranks drugs connected to the specific target node.
        """
        model.eval()
        with torch.no_grad():
            # Get drug indices
            drug_indices = list(self.drug_key_mapping.values())
            num_drugs = len(drug_indices)
            
            src = torch.tensor(drug_indices, dtype=torch.long, device=self.device)
            dst = torch.tensor([target_node_idx] * num_drugs, dtype=torch.long, device=self.device)
            query_edges = torch.stack([src, dst])
            
            # Compute heuristics (using CPU graph reference if available, else recompute)
            # For speed, we just recompute normalisation=False
            if hasattr(self, 'graph_cpu') and hasattr(self, 'query_edges_cpu'):
                inf_heuristics = compute_heuristic_edge_features(self.graph_cpu, self.query_edges_cpu, normalise=False, show_progress=False)
            else:
                # Fallback
                graph_cpu = self.graph.cpu()
                inf_heuristics = compute_heuristic_edge_features(graph_cpu, query_edges.cpu(), normalise=False, show_progress=False)
            
            inf_heuristics_gpu = inf_heuristics.to(self.device)
            
            # Decode
            scores = model.decode(z, query_edges, heuristic_features=inf_heuristics_gpu)
            
            # Compute Rank and Hits
            # We need to know which drugs are TRUE positives for this target
            # self.target_true_drugs should be set during setup
            if not hasattr(self, 'target_true_drugs'):
                return {'median_rank': -1, 'hits_20': -1}
                
            true_drug_indices = self.target_true_drugs
            
            # Sort scores descending
            sorted_indices = torch.argsort(scores, descending=True)
            sorted_drugs = src[sorted_indices].cpu().numpy()
            
            # Find ranks
            ranks = []
            for true_drug in true_drug_indices:
                try:
                    # Find where true drug is in sorted list
                    # This is O(N) but N is small (1500 drugs)
                    rank = np.where(sorted_drugs == true_drug)[0][0] + 1
                    ranks.append(rank)
                except IndexError:
                    pass
            
            if not ranks:
                return {'median_rank': -1, 'hits_20': 0}
                
            median_rank = np.median(ranks)
            hits_20 = sum(r <= 20 for r in ranks)
            
            return {'median_rank': median_rank, 'hits_20': hits_20}
    
    def predict_edge(self, model: torch.nn.Module, graph: torch.Tensor,
                    source_idx: int, target_idx: int, 
                    idx_mapping: Optional[Dict[int, int]] = None) -> float:
        """
        Predict the probability of an edge.
        
        Args:
            model: Trained model
            graph: Graph (possibly modified)
            source_idx: Original source node index
            target_idx: Original target node index
            idx_mapping: If node was removed, mapping from old to new indices
        """
        model.eval()
        graph = graph.to(self.device)
        
        with torch.no_grad():
            z = model.encode(graph.x, graph.edge_index)
            
            # Handle index remapping if node was removed
            if idx_mapping is not None:
                # If the node we're predicting for was removed, we can't make a prediction
                if source_idx not in idx_mapping or target_idx not in idx_mapping:
                    return None
                source_idx = idx_mapping[source_idx]
                target_idx = idx_mapping[target_idx]
            
            edge = torch.tensor([[source_idx], [target_idx]], device=self.device)
            pred = model.decode(z, edge)
            prob = torch.sigmoid(pred).item()
        
        return prob
    
    def generate_negative_samples(self, drug_idx: int, disease_idx: int, 
                                  num_negatives: int = 100) -> List[Tuple[int, int]]:
        """Generate negative samples for evaluation."""
        negatives = []
        
        all_drugs = list(self.connected_drugs)
        all_diseases = list(self.connected_diseases)
        
        # Generate negative samples by pairing with random nodes
        attempts = 0
        while len(negatives) < num_negatives and attempts < num_negatives * 10:
            # Randomly choose to corrupt drug or disease
            if np.random.random() < 0.5:
                # Corrupt drug
                neg_drug = np.random.choice(all_drugs)
                neg_disease = disease_idx
            else:
                # Corrupt disease
                neg_drug = drug_idx
                neg_disease = np.random.choice(all_diseases)
            
            # Check it's not a positive edge
            if (neg_drug, neg_disease) not in self.positive_edge_set:
                negatives.append((neg_drug, neg_disease))
            
            attempts += 1
        
        return negatives
    
    def calculate_apr_for_fold(self, pos_score: float, neg_scores: List[float]) -> Dict[str, float]:
        """Calculate APR and other metrics for a single fold."""
        # Combine scores and labels
        all_scores = [pos_score] + neg_scores
        all_labels = [1] + [0] * len(neg_scores)
        
        # Convert to numpy arrays
        scores = np.array(all_scores)
        labels = np.array(all_labels)
        
        # Calculate metrics
        apr = average_precision_score(labels, scores)
        auc = roc_auc_score(labels, scores)
        
        # Calculate rank of positive sample
        sorted_indices = np.argsort(scores)[::-1]
        pos_rank = np.where(sorted_indices == 0)[0][0] + 1
        
        # Calculate Hits@K
        hits_at_1 = 1 if pos_rank == 1 else 0
        hits_at_5 = 1 if pos_rank <= 5 else 0
        hits_at_10 = 1 if pos_rank <= 10 else 0
        
        # Mean Reciprocal Rank
        mrr = 1.0 / pos_rank
        
        return {
            'apr': apr,
            'auc': auc,
            'pos_rank': pos_rank,
            'hits@1': hits_at_1,
            'hits@5': hits_at_5,
            'hits@10': hits_at_10,
            'mrr': mrr,
            'pos_score': pos_score,
            'neg_score_mean': np.mean(neg_scores),
            'neg_score_std': np.std(neg_scores),
            'num_negatives': len(neg_scores)
        }
    
    def run_validation(self, sample_size: Optional[int] = None):
        """
        Run leave-one-out validation.
        
        Args:
            sample_size: Number of edges to sample for validation (None = all)
        """
        self.load_graph_data()
        
        # Get list of drug-disease pairs to validate
        drug_disease_pairs = list(zip(
            self.drug_disease_edges[0].numpy(),
            self.drug_disease_edges[1].numpy()
        ))
        
        # Sample if requested
        if sample_size and sample_size < len(drug_disease_pairs):
            np.random.seed(42)
            indices = np.random.choice(len(drug_disease_pairs), sample_size, replace=False)
            drug_disease_pairs = [drug_disease_pairs[i] for i in indices]
        
        if self.num_folds:
            drug_disease_pairs = drug_disease_pairs[:self.num_folds]
        
        print(f"\nRunning validation on {len(drug_disease_pairs)} drug-disease pairs...")
        print(f"Removal mode: {self.removal_mode}")
        if self.removal_mode == 'node':
            print(f"Removing: {self.remove_node_type} node")
        
        # Progress tracking
        all_metrics = []
        
        for fold_idx, (drug_idx, disease_idx) in enumerate(tqdm(drug_disease_pairs, desc="Leave-one-out folds")):
            drug_id = self.idx_to_drug.get(drug_idx, f"drug_{drug_idx}")
            disease_id = self.idx_to_disease.get(disease_idx, f"disease_{disease_idx}")
            
            try:
                # Modify graph based on removal mode
                idx_mapping = None
                
                if self.removal_mode == 'edge':
                    # Only remove the drug-disease edge
                    modified_graph = self.remove_edge(self.graph, drug_idx, disease_idx)
                    test_drug_idx = drug_idx
                    test_disease_idx = disease_idx
                    
                elif self.removal_mode == 'node':
                    # Remove entire node and all its edges
                    if self.remove_node_type == 'drug':
                        modified_graph, idx_mapping = self.remove_node(self.graph, drug_idx)
                        # Can't predict for removed node - skip this fold
                        # Instead, we test if model can still predict disease's other drugs
                        continue  # For now, skip node removal for the removed node itself
                    else:
                        modified_graph, idx_mapping = self.remove_node(self.graph, disease_idx)
                        continue  # Skip disease removal folds
                
                # Create training data from remaining edges
                train_edges = modified_graph.edge_index
                train_labels = torch.ones(train_edges.shape[1])
                
                # Add some negative samples for training
                num_train_neg = min(1000, train_edges.shape[1])
                neg_edges = []
                for _ in range(num_train_neg):
                    neg_drug = np.random.choice(list(self.connected_drugs))
                    neg_disease = np.random.choice(list(self.connected_diseases))
                    if (neg_drug, neg_disease) not in self.positive_edge_set:
                        neg_edges.append([neg_drug, neg_disease])
                
                if neg_edges:
                    neg_edge_tensor = torch.tensor(neg_edges, dtype=torch.long).t()
                    train_edges = torch.cat([train_edges, neg_edge_tensor], dim=1)
                    train_labels = torch.cat([train_labels, torch.zeros(len(neg_edges))])
                
                # Create and train model
                model = self.create_model(modified_graph.x.shape[1])
                model = self.train_model(model, modified_graph, train_edges, train_labels)
                
                # Predict held-out edge
                pos_score = self.predict_edge(model, modified_graph, test_drug_idx, test_disease_idx, idx_mapping)
                
                if pos_score is None:
                    continue
                
                # Generate negative samples and predict
                negatives = self.generate_negative_samples(test_drug_idx, test_disease_idx, num_negatives=100)
                neg_scores = []
                for neg_drug, neg_disease in negatives:
                    neg_score = self.predict_edge(model, modified_graph, neg_drug, neg_disease, idx_mapping)
                    if neg_score is not None:
                        neg_scores.append(neg_score)
                
                if len(neg_scores) < 10:
                    continue
                
                # Calculate metrics
                fold_metrics = self.calculate_apr_for_fold(pos_score, neg_scores)
                fold_metrics['fold'] = fold_idx
                fold_metrics['drug_id'] = drug_id
                fold_metrics['disease_id'] = disease_id
                fold_metrics['drug_idx'] = drug_idx
                fold_metrics['disease_idx'] = disease_idx
                
                all_metrics.append(fold_metrics)
                
                # Log progress every 10 folds
                if (fold_idx + 1) % 10 == 0:
                    mean_apr = np.mean([m['apr'] for m in all_metrics])
                    mean_mrr = np.mean([m['mrr'] for m in all_metrics])
                    print(f"\n  Fold {fold_idx + 1}: Running Mean APR={mean_apr:.4f}, MRR={mean_mrr:.4f}")
                
            except Exception as e:
                print(f"\n  Error in fold {fold_idx} (drug={drug_id}, disease={disease_id}): {e}")
                continue
        
        # Store results
        self.fold_metrics = all_metrics
        self.calculate_summary_metrics()
        
        return self.summary_metrics
    
    def calculate_summary_metrics(self):
        """Calculate summary metrics across all folds."""
        if not self.fold_metrics:
            self.summary_metrics = {}
            return
        
        df = pd.DataFrame(self.fold_metrics)
        
        self.summary_metrics = {
            'mean_apr': df['apr'].mean(),
            'std_apr': df['apr'].std(),
            'median_apr': df['apr'].median(),
            'mean_auc': df['auc'].mean(),
            'std_auc': df['auc'].std(),
            'mean_mrr': df['mrr'].mean(),
            'std_mrr': df['mrr'].std(),
            'hits@1': df['hits@1'].mean(),
            'hits@5': df['hits@5'].mean(),
            'hits@10': df['hits@10'].mean(),
            'mean_pos_rank': df['pos_rank'].mean(),
            'median_pos_rank': df['pos_rank'].median(),
            'mean_pos_score': df['pos_score'].mean(),
            'mean_neg_score': df['neg_score_mean'].mean(),
            'num_folds': len(df),
            'removal_mode': self.removal_mode,
            'remove_node_type': self.remove_node_type if self.removal_mode == 'node' else None
        }
        
        print("\n" + "="*80)
        print("LEAVE-ONE-OUT VALIDATION SUMMARY")
        print("="*80)
        print(f"\nRemoval Mode: {self.removal_mode}")
        if self.removal_mode == 'node':
            print(f"Node Type Removed: {self.remove_node_type}")
        print(f"Number of Folds: {len(df)}")
        print(f"\n{'Metric':<20} {'Mean':>10} {'Std':>10} {'Median':>10}")
        print("-"*50)
        print(f"{'APR':<20} {self.summary_metrics['mean_apr']:>10.4f} {self.summary_metrics['std_apr']:>10.4f} {self.summary_metrics['median_apr']:>10.4f}")
        print(f"{'AUC':<20} {self.summary_metrics['mean_auc']:>10.4f} {self.summary_metrics['std_auc']:>10.4f}")
        print(f"{'MRR':<20} {self.summary_metrics['mean_mrr']:>10.4f} {self.summary_metrics['std_mrr']:>10.4f}")
        print(f"\n{'Hits@K Metrics:':<20}")
        print(f"  {'Hits@1':<18} {self.summary_metrics['hits@1']:>10.4f}")
        print(f"  {'Hits@5':<18} {self.summary_metrics['hits@5']:>10.4f}")
        print(f"  {'Hits@10':<18} {self.summary_metrics['hits@10']:>10.4f}")
        print(f"\n{'Ranking:':<20}")
        print(f"  {'Mean Rank':<18} {self.summary_metrics['mean_pos_rank']:>10.2f}")
        print(f"  {'Median Rank':<18} {self.summary_metrics['median_pos_rank']:>10.2f}")
        print(f"\n{'Scores:':<20}")
        print(f"  {'Mean Pos Score':<18} {self.summary_metrics['mean_pos_score']:>10.4f}")
        print(f"  {'Mean Neg Score':<18} {self.summary_metrics['mean_neg_score']:>10.4f}")
        print("="*80)
    
    def save_results(self, output_dir: str = 'results/validation'):
        """Save validation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        mode_suffix = f"{self.removal_mode}"
        if self.removal_mode == 'node':
            mode_suffix += f"_{self.remove_node_type}"
        
        # Save fold-level results
        if self.fold_metrics:
            fold_df = pd.DataFrame(self.fold_metrics)
            fold_file = output_path / f"loo_fold_results_{mode_suffix}_{timestamp}.csv"
            fold_df.to_csv(fold_file, index=False)
            print(f"\nFold results saved to: {fold_file}")
        
        # Save summary metrics
        summary_file = output_path / f"loo_summary_{mode_suffix}_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(self.summary_metrics, f, indent=2, default=str)
        print(f"Summary saved to: {summary_file}")
        
        return output_path

    def validate_multi_disease(self, k: int = 20):
        """
        Run LOO validation across diseases specified in config.
        
        Uses config.optimisation_config.loo_validation_diseases list.
        Returns aggregate metrics suitable for Bayesian optimisation.
        """
        self.load_graph_data()
        
        # Get disease list from config
        diseases = self.config.optimisation_config.get('loo_validation_diseases', [])
        
        if not diseases:
            print("No diseases specified in config.optimisation_config.loo_validation_diseases")
            return None
        
        print(f"\n{'='*80}")
        print(f"MULTI-DISEASE LOO VALIDATION ({len(diseases)} diseases)")
        print(f"{'='*80}\n")
        
        metrics = []
        skipped = []
        
        for i, disease_id in enumerate(diseases):
            # Check if disease exists in graph
            if disease_id not in self.disease_key_mapping:
                print(f"[{i+1}/{len(diseases)}] SKIPPED: {disease_id} not found in graph")
                skipped.append(disease_id)
                continue
            
            disease_idx = self.disease_key_mapping[disease_id]
            drug_count = len(self.get_drug_neighbors(disease_idx))
            
            print(f"\n[{i+1}/{len(diseases)}] Validating {disease_id} ({drug_count} known drugs)...")
            
            if drug_count == 0:
                print(f"  SKIPPED: No drug connections")
                skipped.append(disease_id)
                continue
            
            metric = self.validate_specific_node(disease_id, k=k, return_metrics=True)
            if metric:
                metric['disease_id'] = disease_id
                metric['drug_count'] = drug_count
                metrics.append(metric)
        
        if not metrics:
            print("\nNo valid metrics collected.")
            return None
        
        # Calculate aggregate metrics
        total_hits = sum(m['hits'] for m in metrics)
        total_true = sum(m.get('drug_count', 0) for m in metrics)
        avg_hits = np.mean([m['hits'] for m in metrics])
        avg_mean_rank = np.mean([m['mean_rank'] for m in metrics])
        avg_median_rank = np.mean([m['median_rank'] for m in metrics])
        
        # Hits@K rate (for optimisation)
        hits_at_k_rate = total_hits / total_true if total_true > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"MULTI-DISEASE SUMMARY ({len(metrics)}/{len(diseases)} diseases validated)")
        print(f"{'='*80}")
        print(f"\nPer-disease averages:")
        print(f"  Avg Hits@{k}: {avg_hits:.2f}")
        print(f"  Avg Mean Rank: {avg_mean_rank:.1f}")
        print(f"  Avg Median Rank: {avg_median_rank:.1f}")
        print(f"\nAggregate:")
        print(f"  Total Hits@{k}: {total_hits} / {total_true}")
        print(f"  Hits@{k} Rate: {hits_at_k_rate:.4f}")
        
        if skipped:
            print(f"\nSkipped diseases: {skipped}")
        
        return {
            'hits_at_k_rate': hits_at_k_rate,
            'total_hits': total_hits,
            'total_true': total_true,
            'avg_hits': avg_hits,
            'avg_mean_rank': avg_mean_rank,
            'avg_median_rank': avg_median_rank,
            'n_diseases': len(metrics),
            'per_disease': metrics
        }
    
    def get_drug_neighbors(self, disease_idx: int) -> set:
        """Get set of drug indices connected to a disease."""
        drug_indices = set(self.drug_key_mapping.values())
        neighbors = set()
        
        edge_index = self.graph.edge_index
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if dst == disease_idx and src in drug_indices:
                neighbors.add(src)
            if src == disease_idx and dst in drug_indices:
                neighbors.add(dst)
        
        return neighbors

    def validate_top_diseases(self, n: int = 5, k: int = 20):
        """
        Run validation on the top N diseases with the most known drugs.
        This provides a representative 'Global' validation score without the
        computational cost of full Leave-One-Out.
        """
        self.load_graph_data()
        
        # Count connections for each disease
        disease_counts = {}
        for i in range(self.drug_disease_edges.shape[1]):
            disease_idx = self.drug_disease_edges[1, i].item()
            disease_counts[disease_idx] = disease_counts.get(disease_idx, 0) + 1
            
        # Sort by count
        sorted_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)
        top_n = sorted_diseases[:n]
        
        print(f"\n{'='*80}")
        print(f"VALIDATING TOP {n} DISEASES (by connected drug count)")
        print(f"{'='*80}\n")
        
        metrics = []
        
        for i, (disease_idx, count) in enumerate(top_n):
            disease_id = self.idx_to_disease.get(disease_idx, f"Unknown_{disease_idx}")
            print(f"\n[{i+1}/{n}] Validating {disease_id} ({count} known drugs)...")
            
            # Use the existing single-node validation method
            # Note: We need to adapt it slightly to return metrics instead of just printing
            metric = self.validate_specific_node(disease_id, k=k, return_metrics=True)
            if metric:
                 metric['disease_id'] = disease_id
                 metrics.append(metric)
        
        if not metrics:
            print("No metrics collected.")
            return

        # Calculate average metrics
        avg_hits = np.mean([m['hits'] for m in metrics])
        avg_mean_rank = np.mean([m['mean_rank'] for m in metrics])
        avg_median_rank = np.mean([m['median_rank'] for m in metrics])
        
        print(f"\n{'='*80}")
        print(f"SUMMARY FOR TOP {n} DISEASES")
        print(f"{'='*80}")
        print(f"Average Hits@{k}: {avg_hits:.1f}")
        print(f"Average Mean Rank: {avg_mean_rank:.1f}")
        print(f"Average Median Rank: {avg_median_rank:.1f}")
        
    def load_safety_data(self):
        """Load drug warnings into a dictionary: ChemBL ID -> Warning details"""
        import glob
        import pyarrow.dataset as ds
        
        print("\nLoading drug safety data for inference filtering...")
        try:
            path = self.config.paths['drugWarnings']
            parquet_files = glob.glob(f"{path}/*.parquet")
            
            if not parquet_files:
                print("  Warning: No drug warning files found.")
                return {}
            
            dataset = ds.dataset(parquet_files, format="parquet")
            table = dataset.to_table()
            df = table.to_pandas()
            
            warnings_map = {}
            for _, row in df.iterrows():
                ids = row['chemblIds']
                w_type = row['warningType']
                
                if not isinstance(ids, (list, np.ndarray)):
                    continue
                    
                for chembl_id in ids:
                    if chembl_id not in warnings_map:
                        warnings_map[chembl_id] = []
                    if w_type not in warnings_map[chembl_id]:
                        warnings_map[chembl_id].append(w_type)
            
            print(f"  Loaded warnings for {len(warnings_map)} drugs.")
            return warnings_map
        except Exception as e:
            print(f"  Failed to load safety data: {e}")
            return {}

    def validate_specific_node(self, target_node_name: str, k: int = 20, return_metrics: bool = False, log_metrics_file: str = None):
        safety_map = self.load_safety_data()
        """
        Validate ability to recover drug-disease edges for a specific disease.
        
        This removes ONLY the drug-disease edges for the target disease, keeps the
        disease node with all its other connections (genes, pathways, etc.), then
        tests if the model can predict which drugs should be connected.
        
        Args:
            target_node_name: Name of disease to test (e.g. 'EFO_0003854')
            k: Top k predictions to show
            log_metrics_file: Optional path to save CSV with per-epoch metrics
        """
        if not hasattr(self, 'graph'):
            self.load_graph_data()
        
        # 1. Find node index
        target_cur = target_node_name.lower().strip()
        node_idx = None
        node_type = None
        true_name = None
        
        # Search in disease mapping
        for name, idx in self.disease_key_mapping.items():
            if name.lower().strip() == target_cur:
                node_idx = idx
                node_type = 'disease'
                true_name = name
                break
        
        # Search in drug mapping if not found
        if node_idx is None:
            for name, idx in self.drug_key_mapping.items():
                if name.lower().strip() == target_cur:
                    node_idx = idx
                    node_type = 'drug'
                    true_name = name
                    break
                    
        if node_idx is None:
            print(f"Error: Node '{target_node_name}' not found in mappings.")
            return None
            
        print(f"\nAnalysing target node: {true_name} (ID: {node_idx}, Type: {node_type})")
        
        # 2. Identify ground truth DRUG-DISEASE edges only
        true_drug_neighbors = set()
        for i in range(self.drug_disease_edges.shape[1]):
            drug_idx = self.drug_disease_edges[0, i].item()
            disease_idx = self.drug_disease_edges[1, i].item()
            
            if node_type == 'disease' and disease_idx == node_idx:
                true_drug_neighbors.add(drug_idx)
            elif node_type == 'drug' and drug_idx == node_idx:
                true_drug_neighbors.add(disease_idx)
        
        print(f"Node has {len(true_drug_neighbors)} drug-disease connections.")
        
        # Cache for fast validation snapshot
        self.target_true_drugs = true_drug_neighbors
        self.target_node_idx = node_idx
        
        if len(true_drug_neighbors) == 0:
            print("Warning: No drug-disease edges found for this node.")
            return None
        
        # 3. Create Training Graph - Remove ONLY the drug-disease edges for this node
        print("Creating training graph (removing drug-disease edges only, keeping disease node)...")
        train_graph = copy.copy(self.graph)
        edge_index = train_graph.edge_index
        
        # Create a set of drug-disease edge pairs for fast lookup
        drug_disease_edge_set = set()
        for i in range(self.drug_disease_edges.shape[1]):
            drug_idx = self.drug_disease_edges[0, i].item()
            disease_idx = self.drug_disease_edges[1, i].item()
            
            # Only add edges involving our target node
            if (node_type == 'disease' and disease_idx == node_idx) or \
               (node_type == 'drug' and drug_idx == node_idx):
                # Add both directions for undirected graph
                drug_disease_edge_set.add((drug_idx, disease_idx))
                drug_disease_edge_set.add((disease_idx, drug_idx))
        
        print(f"Target node has {len(drug_disease_edge_set) // 2} drug-disease edges to remove")
        
        # Filter edges efficiently using the set
        edges_to_keep = []
        removed_count = 0
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            
            # Check if this edge is in our removal set
            if (src, dst) not in drug_disease_edge_set:
                edges_to_keep.append(i)
            else:
                removed_count += 1
        
        # Update edge attributes (if present) to match filtered edges
        if hasattr(train_graph, 'edge_attr') and train_graph.edge_attr is not None:
            train_graph.edge_attr = train_graph.edge_attr[edges_to_keep]
            
        if hasattr(train_graph, 'edge_type') and train_graph.edge_type is not None:
            train_graph.edge_type = train_graph.edge_type[edges_to_keep]
            
        train_graph.edge_index = edge_index[:, edges_to_keep]
        print(f"Training graph: {train_graph.num_nodes} nodes, {train_graph.edge_index.shape[1]} edges")
        print(f"Removed {removed_count} edges (disease node kept with other connections)")
        
        # 4. Train Model
        # 4. Train Model
        print(f"Training model on modified graph...")
        
        # Train only on known drug-disease edges (minus the held out ones)
        
        # Filter global drug-disease edges to exclude the held-out ones
        src_dd = self.drug_disease_edges[0]
        dst_dd = self.drug_disease_edges[1]
        
        # Keep edges where neither source nor dest is the target node
        keep_mask = (src_dd != node_idx) & (dst_dd != node_idx)
        train_pos_edges = self.drug_disease_edges[:, keep_mask]
        
        print(f"Defining training set: {train_pos_edges.shape[1]} drug-disease edges (excluding target connections)")
        
        # Add negative samples using config sampler: matches standard training
        neg_config = self.config.get_negative_sampling_config()
        strategy = neg_config['strategy']
        params = neg_config['params']
        neg_ratio = neg_config['train_neg_ratio']
        
        num_neg = train_pos_edges.shape[1] * neg_ratio
        print(f"Sampling negatives using '{strategy}' strategy (Ratio 1:{neg_ratio}, Target: {num_neg})...")
        
        # Prepare data for sampler
        # 1. Positive edges set
        pos_edges_list = train_pos_edges.t().tolist()
        pos_edges_set = set((r[0], r[1]) for r in pos_edges_list)
        
        # 2. All possible pairs (Drug x Disease)
        drug_indices = list(self.drug_key_mapping.values())
        disease_indices = list(self.disease_key_mapping.values())
        
        # Use itertools to generate pairs lazily if sampler supported it, but it expects list
        import itertools
        all_possible_pairs = list(itertools.product(drug_indices, disease_indices))
        
        # These are the edges we're trying to predict - we must not sample them as negatives!
        test_positive_edges = set()
        for drug_idx in true_drug_neighbors:
            if node_type == 'disease':
                test_positive_edges.add((drug_idx, node_idx))
            else:  # node_type == 'drug'
                test_positive_edges.add((node_idx, drug_idx))
        
        print(f"  Excluding {len(test_positive_edges)} held-out test edges from negative sampling...")
        
        # 3. Init and Run Sampler with future_positives
        params_with_future = params.copy()
        params_with_future['future_positives'] = test_positive_edges
        sampler = get_sampler(strategy, **params_with_future)
        
        neg_samples_list = sampler.sample(
            positive_edges=pos_edges_set,
            all_possible_pairs=all_possible_pairs,
            num_samples=num_neg,
            edge_index=train_graph.edge_index, # Use full graph structure for structural sampling
            node_features=train_graph.x
        )
        
        neg_edges = torch.tensor(neg_samples_list, dtype=torch.long).t()
        
        # Ensure label count matches actual samples received
        actual_num_neg = len(neg_samples_list)
        if actual_num_neg != int(num_neg):
            print(f"Warning: Requested {int(num_neg)} negatives, but received {actual_num_neg}")
            
        train_edges = torch.cat([train_pos_edges, neg_edges], dim=1)
        train_labels = torch.cat([torch.ones(train_pos_edges.shape[1]), torch.zeros(actual_num_neg)])
        
        model = self.create_model(train_graph.x.shape[1])
        model = self.train_model(model, train_graph, train_edges, train_labels, log_metrics_file=log_metrics_file)
        
        # 5. Inference:  Predict drug-disease links
        print("Running inference (ranking drugs for target disease)...")
        model.eval()
        
        # Use the training graph for inference (disease node is still there, just without drug edges)
        inf_graph = train_graph.to(self.device)
        
        with torch.no_grad():
            edge_attr = getattr(inf_graph, 'edge_attr', None)
            z = model.encode(inf_graph.x, inf_graph.edge_index, edge_attr=edge_attr)
            # Construct query edges
            drug_indices = list(self.drug_key_mapping.values())
            num_drugs = len(drug_indices)
            
            src = torch.tensor(drug_indices, dtype=torch.long, device=self.device)
            dst = torch.tensor([node_idx] * num_drugs, dtype=torch.long, device=self.device)
            query_edges = torch.stack([src, dst])
            
            # Compute heuristics for inference query
            # Use training graph as reference (CPU copy needed for heuristic algo)
            inf_graph_cpu = inf_graph.cpu()
            query_edges_cpu = query_edges.cpu()
            
            inf_heuristics = compute_heuristic_edge_features(inf_graph_cpu, query_edges_cpu, normalise=False)
            inf_heuristics_gpu = inf_heuristics.to(self.device)
            
            # Decode in one batch
            scores = model.decode(z, query_edges, heuristic_features=inf_heuristics_gpu)
            probs = torch.sigmoid(scores)
            
            drug_results = []
            for i, drug_idx in enumerate(drug_indices):
                score = scores[i].item()
                prob = probs[i].item()
                cn = inf_heuristics[i, 0].item()  # Common Neighbors
                aa = inf_heuristics[i, 1].item()  # Adamic-Adar
                drug_results.append((drug_idx, score, prob, cn, aa))
            
            # Sort by score
            sorted_drugs = sorted(drug_results, key=lambda x: x[1], reverse=True)
            
            print(f"\nTop {k} Drug Predictions for {true_name}:")
            print(f"{'Rank':<5} {'Score':<10} {'Prob':<10} {'CN':<8} {'AA':<8} {'Drug ID':<20} {'Warning':<20} {'Is True?'}")
            print("-" * 120)
            
            # Build top-20 list for display and JSON
            top20_list = []
            for rank, (drug_idx, score, prob, cn, aa) in enumerate(sorted_drugs[:20], 1):
                drug_id = self.idx_to_drug.get(drug_idx, "Unknown")
                is_true = drug_idx in true_drug_neighbors

                # Check warnings using ChemBL ID
                warnings = safety_map.get(drug_id, [])
                safe_str = ", ".join(warnings) if warnings else "Safe"
                if "Black Box Warning" in warnings:
                    safe_str = "BLACK BOX"
                elif "Withdrawn" in warnings:
                    safe_str = "WITHDRAWN"

                top20_list.append({
                    "rank": rank, "drug_id": drug_id, "score": round(score, 4),
                    "prob": round(prob, 4), "is_true": is_true,
                })
                mark = "✓ YES (RECOVERED)" if is_true else ""
                print(f"{rank:<5} {score:<10.4f} {prob:<10.4f} {cn:<8.1f} {aa:<8.2f} {drug_id:<20} {safe_str:<20} {mark}")
            
            # Calculate ranks for all true neighbours
            print(f"\nAnalysis of True Drug Connections (Total: {len(true_drug_neighbors)}):")
            print(f"{'Drug ID':<20} {'Rank':<8} {'Score':<10} {'GNN':<10} {'Heur':<10} {'Prob':<10} {'Warning'}")
            print("-" * 120)
            
            ranks = []
            rank_map = {d: (r, s, p, g, h) for r, (d, s, p, g, h) in enumerate(sorted_drugs, 1)}
            
            for drug_idx in true_drug_neighbors:
                if drug_idx in rank_map:
                    rank, score, prob, gnn_s, h_s = rank_map[drug_idx]
                    ranks.append(rank)
                    drug_id = self.idx_to_drug.get(drug_idx, "Unknown")
                    
                    warnings = safety_map.get(drug_id, [])
                    safe_str = ", ".join(warnings) if warnings else "Safe"
                    if "Black Box Warning" in warnings: safe_str = "BLACK BOX"
                    elif "Withdrawn" in warnings: safe_str = "WITHDRAWN"
                    
                    print(f"{drug_id:<20} {rank:<8} {score:<10.4f} {gnn_s:<10.4f} {h_s:<10.4f} {prob:<10.4f} {safe_str}")
                else:
                    ranks.append(len(sorted_drugs) + 1)
            
            ranks = np.array(ranks)
            n = len(ranks)
            total_drugs = len(sorted_drugs)
            mean_rank = float(np.mean(ranks)) if n > 0 else 0
            median_rank = float(np.median(ranks)) if n > 0 else 0
            
            # Standardised metrics (matching SEAL output)
            hits_at_10 = int(np.sum(ranks <= 10))
            hits_at_20 = int(np.sum(ranks <= 20))
            hits_at_50 = int(np.sum(ranks <= 50))
            hits_at_100 = int(np.sum(ranks <= 100))
            mrr = float(np.mean(1.0 / ranks)) if n > 0 else 0.0
            
            print(f"\n{'=' * 60}")
            print(f"GNN PERFORMANCE SUMMARY FOR {true_name}")
            print(f"{'=' * 60}")
            print(f"  Test Edges (True Positives): {n}")
            print(f"  Total Drugs Ranked: {total_drugs}")
            print(f"  Model: {self.config.model_choice}")
            if n > 0:
                print(f"  Hits@10:  {hits_at_10} / {n} ({hits_at_10 / n * 100:.1f}%)")
                print(f"  Hits@20:  {hits_at_20} / {n} ({hits_at_20 / n * 100:.1f}%)")
                print(f"  Hits@50:  {hits_at_50} / {n} ({hits_at_50 / n * 100:.1f}%)")
                print(f"  Hits@100: {hits_at_100} / {n} ({hits_at_100 / n * 100:.1f}%)")
            print(f"  Median Rank: {median_rank:.1f}")
            print(f"  Mean Rank: {mean_rank:.1f}")
            print(f"  MRR: {mrr:.4f}")
            print(f"{'=' * 60}")
            
            # Save standardised JSON (matching SEAL format)
            results_dir = Path("results/gnn_results")
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            results_json = {
                "method": "gnn",
                "target_disease": true_name,
                "timestamp": timestamp,
                "config": {
                    "model": self.config.model_choice,
                    "epochs": self.epochs_per_fold,
                    "removal_mode": self.removal_mode,
                    "num_layers": self.config.model_config.get('num_layers'),
                    "hidden_channels": self.config.model_config.get('hidden_channels'),
                    "decoder_type": self.config.model_config.get('decoder_type'),
                    "seed": getattr(self, '_seed', 42),
                },
                "metrics": {
                    "hits_at_10": hits_at_10, "hits_at_20": hits_at_20,
                    "hits_at_50": hits_at_50, "hits_at_100": hits_at_100,
                    "total_true": n, "total_drugs": total_drugs,
                    "median_rank": round(median_rank, 1),
                    "mean_rank": round(mean_rank, 1),
                    "mrr": round(mrr, 4),
                },
                "top20": top20_list,
                "all_ranks": sorted(ranks.tolist()),
            }
            
            json_path = results_dir / f"gnn_{true_name}_{timestamp}.json"
            with open(json_path, "w") as f:
                json.dump(results_json, f, indent=2)
            print(f"Results saved to: {json_path}")
            
            # MLflow tracking (if tracker available on validator)
            if hasattr(self, 'tracker') and self.tracker:
                self.tracker.log_metric("hits_at_10", hits_at_10)
                self.tracker.log_metric("hits_at_20", hits_at_20)
                self.tracker.log_metric("hits_at_50", hits_at_50)
                self.tracker.log_metric("hits_at_100", hits_at_100)
                self.tracker.log_metric("median_rank", median_rank)
                self.tracker.log_metric("mean_rank", mean_rank)
                self.tracker.log_metric("mrr", mrr)
                self.tracker.log_metric("total_true", n)
                self.tracker.log_metric("total_drugs", total_drugs)
                self.tracker.log_artifact(str(json_path))
            
            if return_metrics:
                return {
                    'hits': hits_at_20,
                    'hits_at_10': hits_at_10,
                    'hits_at_20': hits_at_20,
                    'hits_at_50': hits_at_50,
                    'hits_at_100': hits_at_100,
                    'mean_rank': mean_rank,
                    'median_rank': median_rank,
                    'mrr': mrr,
                }

def main():
    parser = argparse.ArgumentParser(description='Leave-One-Out Validation for Drug Repurposing')
    parser.add_argument('--graph', type=str, default=None,
                       help='Path to graph file (if not provided, will auto-detect latest)')
    parser.add_argument('--removal-mode', type=str, default='edge', 
                       choices=['edge', 'node'],
                       help='Removal mode: edge (remove only edge) or node (remove entire node)')
    parser.add_argument('--remove-node-type', type=str, default='drug',
                       choices=['drug', 'disease'],
                       help='When using node removal, which node type to remove')
    parser.add_argument('--num-folds', type=int, default=None,
                       help='Number of folds to run (None = all)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of edges to sample for validation')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Training epochs per fold (default: from config)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--log-metrics', type=str, default=None,
                       help='Path to CSV file to log per-epoch validation metrics')
    parser.add_argument('--model', type=str, default='TransformerModel',
                       choices=['GCN', 'SAGE', 'Transformer', 'GAT', 'GCNModel', 'SAGEModel', 'TransformerModel', 'GATModel'],
                       help='Model to use')
    parser.add_argument('--target-node', type=str, default=None,
                       help='Specific node to validate (e.g. "EFO_0003854").')
    parser.add_argument('--top-diseases', type=int, default=0,
                        help='Number of top diseases (by connections) to validate automatically. Defaults to 5 if no target specified and no global options set.')
    parser.add_argument('--multi-disease', action='store_true',
                        help='Run LOO validation across diseases defined in config.optimisation_config.loo_validation_diseases')
    parser.add_argument('--override-config', nargs='*', help='Override config values (key=value)')
    parser.add_argument("--layers", type=int, help="Number of GNN layers")
    parser.add_argument("--decoder-type", type=str, choices=['dot', 'mlp', 'mlp_interaction', 'mlp_neighbor'], 
                      help="Type of decoder to use")
    parser.add_argument("--no-mlflow", action="store_true",
                      help="Disable MLflow experiment tracking")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get config
    config = get_config()
    config.model_choice = args.model
    
    # Apply CLI args
    if args.layers:
        config.model_config['num_layers'] = args.layers
        print(f"Using CLI layers: {args.layers}")
        
    if args.decoder_type:
        config.model_config['decoder_type'] = args.decoder_type
        print(f"Using CLI decoder: {args.decoder_type}")
    
    # Apply type-safe overrides
    if args.override_config:
        print("\nApplying config overrides:")
        for item in args.override_config:
            try:
                key, value = item.split('=')
                # Attempt to convert value to appropriate type
                if value.isdigit():
                    value = int(value)
                elif value.replace('.','',1).isdigit():
                    value = float(value)
                elif value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                
                # Navigate nested keys
                parts = key.split('.')
                target = config
                for part in parts[:-1]:
                    if isinstance(target, dict):
                        target = target[part]
                    else:
                        target = getattr(target, part)
                
                # Set value
                last_key = parts[-1]
                if isinstance(target, dict):
                    target[last_key] = value
                else:
                    setattr(target, last_key, value)
                    
                print(f"  - {key} = {value}")
            except Exception as e:
                print(f"  ! Failed to set {item}: {e}")
    # End overrides
    
    # Set epochs from config if not provided
    if args.epochs is None:
        args.epochs = config.model_config.get('num_epochs', 50)
        print(f"Using config epochs: {args.epochs}")
    else:
        print(f"Using CLI epochs: {args.epochs}")
    
    # Auto-detect graph if not provided
    if args.graph is None:
        args.graph = find_latest_graph()
    
    print(f"Using graph: {args.graph}")
    
    # Create validator
    validator = LeaveOneOutValidator(
        config=config,
        removal_mode=args.removal_mode,
        remove_node_type=args.remove_node_type,
        num_folds=args.num_folds,
        epochs_per_fold=args.epochs
    )
    
    # Store graph path and seed in validator
    validator.graph_path = args.graph
    validator._seed = args.seed
    
    # MLflow tracking
    if not args.no_mlflow and args.target_node:
        tracker = ExperimentTracker(
            experiment_name=f"GNN-{args.target_node}",
        )
        run_name = f"gnn_{args.model}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tracker.start_run(run_name=run_name)
        print(f"MLflow tracking enabled: GNN-{args.target_node} / {run_name}")
        
        # Log parameters
        tracker.log_param("method", "gnn")
        tracker.log_param("target_disease", args.target_node)
        tracker.log_param("model", args.model)
        tracker.log_param("epochs", args.epochs)
        tracker.log_param("seed", args.seed)
        tracker.log_param("removal_mode", args.removal_mode)
        tracker.log_param("num_layers", config.model_config.get('num_layers'))
        tracker.log_param("hidden_channels", config.model_config.get('hidden_channels'))
        tracker.log_param("decoder_type", config.model_config.get('decoder_type'))
        tracker.log_param("graph_path", args.graph)
        
        validator.tracker = tracker
    else:
        validator.tracker = None
    
    if args.target_node:
        # 1. Run specific node validation
        validator.validate_specific_node(args.target_node, log_metrics_file=args.log_metrics)
        
        # End MLflow run
        if validator.tracker:
            validator.tracker.end_run()
            print(f"Ended MLflow run.")
    
    elif args.multi_disease:
        # 2. Run multi-disease LOO (diseases from config)
        k = config.optimisation_config.get('loo_k', 20)
        validator.validate_multi_disease(k=k)
        
    elif args.top_diseases > 0:
        # 3. Run top N diseases validation
        validator.validate_top_diseases(n=args.top_diseases)
        
    elif not args.sample_size and not args.num_folds:
        # 3. Default behavior if no arguments: Validate top 5 diseases
        print("\nNo specific validation arguments provided.")
        print("Defaulting to: Validate Top 5 Diseases (Recommended for quick check)")
        validator.validate_top_diseases(n=5)
        
    else:
        # 4. Run edge-based LOO
        print("\nWARNING: Running edge-by-edge Leave-One-Out validation.")
        print("This is computationally expensive (approx 1 min per edge).")
        input("Press Enter to continue or Ctrl+C to cancel...")
        
        results = validator.run_validation(sample_size=args.sample_size)
        
        # Save results
        validator.save_results()
        
        print("\n" + "="*80)
        print("VALIDATION COMPLETE")
        print("="*80)
        print(f"\nPrimary Metric (APR): {results.get('mean_apr', results.get('apr', 'N/A')):.4f}")

if __name__ == "__main__":
    main()
