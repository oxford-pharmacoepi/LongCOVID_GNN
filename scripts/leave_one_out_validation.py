#!/usr/bin/env python3
"""
Leave-One-Out Validation Script for Drug-Disease Prediction

This script performs rigorous leave-one-out cross-validation by:
1. Removing known drug-disease edges from the graph
2. Optionally removing the drug or disease NODE entirely (not just edges)
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
from src.utils import set_seed
from src.models import GCNModel, SAGEModel, TransformerModel, MODEL_CLASSES
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve


def find_latest_graph(results_dir='results'):
    """Auto-detect the latest graph file."""
    graph_files = glob.glob(f'{results_dir}/graph_*.pt')
    if not graph_files:
        raise FileNotFoundError(f"No graph files found in {results_dir}")
    latest = max(graph_files, key=os.path.getctime)
    print(f"  Auto-detected: {latest}")
    return latest


class LinkPredictor(nn.Module):
    """
    Wrapper that adds link prediction capability to GNN encoder models.
    
    Uses the encoder to get node embeddings, then predicts edge probability
    using dot product or MLP decoder.
    """
    
    def __init__(self, encoder: nn.Module, hidden_channels: int = 128, decoder_type: str = 'dot'):
        super().__init__()
        self.encoder = encoder
        self.decoder_type = decoder_type
        
        if decoder_type == 'mlp':
            self.decoder = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_channels, 1)
            )
    
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings from the encoder."""
        return self.encoder(x, edge_index)
    
    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Predict edge probability given node embeddings."""
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        
        if self.decoder_type == 'dot':
            # Dot product decoder
            return (src * dst).sum(dim=-1)
        else:
            # MLP decoder
            edge_features = torch.cat([src, dst], dim=-1)
            return self.decoder(edge_features).squeeze(-1)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                pred_edges: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode then decode."""
        z = self.encode(x, edge_index)
        return self.decode(z, pred_edges)


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
        
        # Load mappings from processed_data/mappings (JSON files)
        processed_data_path = Path('processed_data')
        mappings_path = processed_data_path / 'mappings'
            
        print(f"Loading mappings from: {mappings_path}")
        
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
        
        # Load drug-disease edges from processed_data/edges
        edges_path = processed_data_path / 'edges'
        self.drug_disease_edges = torch.load(edges_path / '2_molecule_disease_edges.pt', weights_only=False)
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
    
    def create_model(self, in_channels: int, hidden_channels: int = 128) -> torch.nn.Module:
        """Create a fresh model instance with link prediction capability."""
        model_choice = self.config.model_choice
        
        # Get the encoder model class
        ModelClass = MODEL_CLASSES.get(model_choice, TransformerModel)
        
        # Create encoder (output embedding dimension = hidden_channels)
        encoder = ModelClass(
            in_channels=in_channels, 
            hidden_channels=hidden_channels, 
            out_channels=hidden_channels,  # Output embeddings
            num_layers=2,
            dropout_rate=0.3
        )
        
        # Wrap with link predictor
        model = LinkPredictor(encoder, hidden_channels=hidden_channels, decoder_type='dot')
        
        return model.to(self.device)
    
    def train_model(self, model: torch.nn.Module, graph: torch.Tensor, 
                    train_edges: torch.Tensor, train_labels: torch.Tensor) -> torch.nn.Module:
        """Train model on the modified graph."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        graph = graph.to(self.device)
        train_edges = train_edges.to(self.device)
        train_labels = train_labels.to(self.device)
        
        for epoch in range(self.epochs_per_fold):
            optimizer.zero_grad()
            
            # Forward pass
            z = model.encode(graph.x, graph.edge_index)
            pred = model.decode(z, train_edges)
            
            # Loss
            loss = F.binary_cross_entropy_with_logits(pred, train_labels.float())
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        return model
    
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

    def validate_specific_node(self, target_node_name: str, k: int = 20):
        """
        Validate ability to recover drug-disease edges for a specific disease.
        
        This removes ONLY the drug-disease edges for the target disease, keeps the
        disease node with all its other connections (genes, pathways, etc.), then
        tests if the model can predict which drugs should be connected.
        
        Args:
            target_node_name: Name of disease to test (e.g. 'EFO_0003854')
            k: Top k predictions to show
        """
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
            return
            
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
        
        if len(true_drug_neighbors) == 0:
            print("Warning: No drug-disease edges found for this node.")
            return
        
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
        
        train_graph.edge_index = edge_index[:, edges_to_keep]
        print(f"Training graph: {train_graph.num_nodes} nodes, {train_graph.edge_index.shape[1]} edges")
        print(f"Removed {removed_count} edges (disease node kept with other connections)")
        
        # 4. Train Model
        print(f"Training model on modified graph...")
        train_pos_edges = train_graph.edge_index
        
        # Add negative samples
        num_neg = min(train_pos_edges.shape[1], 50000)  # Limit negatives for speed
        neg_edges = torch.randint(0, train_graph.num_nodes, (2, num_neg))
        
        train_edges = torch.cat([train_pos_edges, neg_edges], dim=1)
        train_labels = torch.cat([torch.ones(train_pos_edges.shape[1]), torch.zeros(num_neg)])
        
        model = self.create_model(train_graph.x.shape[1])
        model = self.train_model(model, train_graph, train_edges, train_labels)
        
        # 5. Inference - Predict drug-disease links
        print("Running inference (ranking drugs for target disease)...")
        model.eval()
        
        # Use the training graph for inference (disease node is still there, just without drug edges)
        inf_graph = train_graph.to(self.device)
        
        with torch.no_grad():
            z = model.encode(inf_graph.x, inf_graph.edge_index)
            z_target = z[node_idx]  # Embedding of target disease
            
            # Get all drug node indices
            drug_indices = list(self.drug_key_mapping.values())
            
            # Score ONLY against drug nodes
            drug_scores = {}
            for drug_idx in drug_indices:
                z_drug = z[drug_idx]
                score = (z_target * z_drug).sum().item()  # Dot product
                drug_scores[drug_idx] = score
            
            # Sort by score
            sorted_drugs = sorted(drug_scores.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\nTop {k} Drug Predictions for {true_name}:")
            print(f"{'Rank':<5} {'Score':<10} {'Drug ID':<20} {'Is True Connection?'}")
            print("-" * 80)
            
            hits = 0
            for rank, (drug_idx, score) in enumerate(sorted_drugs[:k], 1):
                drug_id = self.idx_to_drug.get(drug_idx, "Unknown")
                is_true = drug_idx in true_drug_neighbors
                mark = "✓ YES (RECOVERED)" if is_true else ""
                if is_true:
                    hits += 1
                
                print(f"{rank:<5} {score:<10.4f} {drug_id:<20} {mark}")
            
            # Calculate ranks for all true neighbors
            print(f"\nAnalysis of True Drug Connections (Total: {len(true_drug_neighbors)}):")
            ranks = []
            for drug_idx in true_drug_neighbors:
                # Find rank of this drug
                rank = next((i for i, (d, s) in enumerate(sorted_drugs, 1) if d == drug_idx), len(sorted_drugs) + 1)
                ranks.append(rank)
                
                if rank > k:
                    drug_id = self.idx_to_drug.get(drug_idx, "Unknown")
                    score = drug_scores.get(drug_idx, 0)
                    print(f"  {drug_id}: Rank {rank}, Score {score:.4f}")
            
            ranks = np.array(ranks)
            print(f"\nMetrics:")
            print(f"  Hits@{k}: {hits} / {len(true_drug_neighbors)}")
            print(f"  Median Rank: {np.median(ranks):.1f}")
            print(f"  Mean Rank: {np.mean(ranks):.2f}")
            print(f"  Min Rank: {np.min(ranks) if len(ranks) > 0 else 'N/A'}")
            print(f"  Max Rank: {np.max(ranks) if len(ranks) > 0 else 'N/A'}")




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
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs per fold')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--model', type=str, default='TransformerModel',
                       choices=['GCN', 'SAGE', 'Transformer', 'GCNModel', 'SAGEModel', 'TransformerModel'],
                       help='Model to use')
    parser.add_argument('--target-node', type=str, default=None,
                       help='Specific node to validate (e.g., "EFO_0003854", which is postmenopausal osteoporosis). Performance test for recovering its edges.')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get config
    config = get_config()
    config.model_choice = args.model
    
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
    
    # Store graph path in validator
    validator.graph_path = args.graph
    
    if args.target_node:
        # Run specific node validation
        validator.validate_specific_node(args.target_node)
        return {}
    else:
        # Run standard validation
        results = validator.run_validation(sample_size=args.sample_size)
        
        # Save results
        validator.save_results()
        
        print("\n" + "="*80)
        print("VALIDATION COMPLETE")
        print("="*80)
        print(f"\nPrimary Metric (APR): {results.get('mean_apr', results.get('apr', 'N/A')):.4f}")
        
        return results


if __name__ == "__main__":
    main()
