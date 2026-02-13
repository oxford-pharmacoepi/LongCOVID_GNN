#!/usr/bin/env python3
"""
Visualize GNN embeddings to diagnose what the model is learning.

Creates t-SNE and UMAP plots of node embeddings, colored by node type.
Helps identify:
- Are embeddings collapsed? (all nodes in one blob = bad)
- Do node types separate? (drugs vs genes vs diseases)
- Do true connections cluster together?

Usage:
    uv run scripts/visualise_embeddings.py
    uv run scripts/visualise_embeddings.py --highlight-disease EFO_0003854
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import get_config
from src.models import TransformerModel, GCNModel, SAGEModel


def find_latest_model(results_path: Path, model_type: str = "TransformerModel"):
    """Find most recent model file."""
    model_dir = results_path / "models"
    pattern = f"{model_type}_best_model_*.pt"
    model_files = list(model_dir.glob(pattern))
    if not model_files:
        # Try without model type prefix
        model_files = list(model_dir.glob("*_best_model_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    return max(model_files, key=lambda p: p.stat().st_mtime)


def find_latest_graph(results_path: Path):
    """Find most recent graph file."""
    graph_files = list(results_path.glob("graph_*.pt"))
    if not graph_files:
        raise FileNotFoundError(f"No graph files found in {results_path}")
    return max(graph_files, key=lambda p: p.stat().st_mtime)


def load_mappings(graph_path: Path):
    """Load node mappings."""
    mappings_dir = str(graph_path).replace('.pt', '_mappings')
    
    if os.path.isdir(mappings_dir):
        with open(f"{mappings_dir}/drug_key_mapping.json") as f:
            drug_mapping = json.load(f)
        with open(f"{mappings_dir}/disease_key_mapping.json") as f:
            disease_mapping = json.load(f)
        with open(f"{mappings_dir}/gene_key_mapping.json") as f:
            gene_mapping = json.load(f)
    else:
        # Fallback
        mappings_path = Path('processed_data/mappings')
        with open(mappings_path / 'drug_key_mapping.json') as f:
            drug_mapping = json.load(f)
        with open(mappings_path / 'disease_key_mapping.json') as f:
            disease_mapping = json.load(f)
        with open(mappings_path / 'gene_key_mapping.json') as f:
            gene_mapping = json.load(f)
    
    return drug_mapping, disease_mapping, gene_mapping


def get_node_types(num_nodes, drug_mapping, disease_mapping, gene_mapping):
    """Create array of node types (0=other, 1=drug, 2=disease, 3=gene)."""
    node_types = np.zeros(num_nodes, dtype=np.int32)
    
    for idx in drug_mapping.values():
        if idx < num_nodes:
            node_types[idx] = 1
    for idx in disease_mapping.values():
        if idx < num_nodes:
            node_types[idx] = 2
    for idx in gene_mapping.values():
        if idx < num_nodes:
            node_types[idx] = 3
    
    return node_types


def main():
    parser = argparse.ArgumentParser(description="Visualize GNN embeddings")
    parser.add_argument('--highlight-disease', type=str, default=None,
                        help='Disease ID to highlight its drug connections')
    parser.add_argument('--sample-size', type=int, default=5000,
                        help='Number of nodes to sample for visualization (default: 5000)')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity (default: 30)')
    parser.add_argument('--output', type=str, default='results/embedding_viz.png',
                        help='Output file path')
    
    args = parser.parse_args()
    
    config = get_config()
    results_path = Path(config.paths['results'])
    
    # Load graph
    graph_path = find_latest_graph(results_path)
    print(f"Loading graph: {graph_path}")
    graph = torch.load(graph_path, weights_only=False)
    print(f"  Nodes: {graph.num_nodes:,}, Edges: {graph.edge_index.shape[1]:,}")
    
    # Load model
    model_path = find_latest_model(results_path)
    print(f"Loading model: {model_path}")
    
    # Determine model type from filename
    model_name = model_path.stem
    if 'Transformer' in model_name:
        ModelClass = TransformerModel
    elif 'SAGE' in model_name:
        ModelClass = SAGEModel
    else:
        ModelClass = GCNModel
    
    # Create model with same architecture
    model_config = config.model_config
    edge_dim = graph.edge_attr.size(1) if hasattr(graph, 'edge_attr') and graph.edge_attr is not None else None
    
    # Load weights first to infer architecture
    state_dict = torch.load(model_path, weights_only=False)
    
    # Infer architecture from state_dict
    in_channels = graph.x.shape[1]
    final_layer_shape = state_dict['final_layer.weight'].shape
    out_channels = final_layer_shape[0]
    
    # ln.weight shape tells us the head_out_channels
    ln_shape = state_dict['ln.weight'].shape[0]
    
    # conv1.lin_key.weight is [hidden * heads, in_channels]
    conv1_key_shape = state_dict['conv1.lin_key.weight'].shape
    
    # We need to figure out heads and hidden_channels
    # The final_layer input matches ln output: head_out_channels
    # If concat=True: head_out_channels = hidden * heads
    # If concat=False: head_out_channels = hidden
    
    # Try common head values
    heads = 4  # Default
    for h in [4, 2, 8, 1]:
        if conv1_key_shape[0] % h == 0:
            potential_hidden = conv1_key_shape[0] // h
            # Check if ln matches expected size
            if ln_shape == potential_hidden:  # concat=False
                heads = h
                hidden_channels = potential_hidden
                concat = False
                break
            elif ln_shape == potential_hidden * h:  # concat=True
                heads = h
                hidden_channels = potential_hidden
                concat = True
                break
    else:
        # Fallback
        heads = 4
        hidden_channels = conv1_key_shape[0] // heads
        concat = ln_shape == hidden_channels * heads
    
    print(f"  Inferred architecture: hidden={hidden_channels}, heads={heads}, out={out_channels}, concat={concat}")
    
    model = ModelClass(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=model_config.get('num_layers', 2),
        dropout_rate=model_config.get('dropout_rate', 0.5),
        heads=heads,
        concat=concat,
        edge_dim=edge_dim
    )
    model.load_state_dict(state_dict)
    model.eval()
    
    # Get embeddings
    print("Computing embeddings...")
    with torch.no_grad():
        edge_attr = getattr(graph, 'edge_attr', None)
        embeddings = model(graph.x, graph.edge_index, edge_attr=edge_attr)
    
    embeddings = embeddings.numpy()
    print(f"  Embedding shape: {embeddings.shape}")
    
    # Check embedding statistics
    print(f"\nEmbedding statistics:")
    print(f"  Mean: {embeddings.mean():.4f}")
    print(f"  Std: {embeddings.std():.4f}")
    print(f"  Min: {embeddings.min():.4f}")
    print(f"  Max: {embeddings.max():.4f}")
    
    # Check for collapsed embeddings
    pairwise_distances = np.linalg.norm(
        embeddings[:100, None] - embeddings[None, :100], axis=2
    )
    avg_distance = pairwise_distances.mean()
    print(f"  Avg pairwise distance (sample): {avg_distance:.4f}")
    
    if avg_distance < 0.1:
        print("\n⚠️  WARNING: Embeddings appear COLLAPSED (very small pairwise distances)")
        print("    This suggests over-smoothing or training issues.")
    elif avg_distance < 1.0:
        print("\n⚠️  Embeddings have relatively low variance - may indicate partial collapse")
    else:
        print("\n✓ Embeddings have reasonable variance")
    
    # Load mappings
    drug_mapping, disease_mapping, gene_mapping = load_mappings(graph_path)
    node_types = get_node_types(graph.num_nodes, drug_mapping, disease_mapping, gene_mapping)
    
    # Sample nodes for visualisation
    n_samples = min(args.sample_size, graph.num_nodes)
    
    # Stratified sampling: ensure we get some of each type
    indices = []
    for node_type in [1, 2, 3]:  # drug, disease, gene
        type_indices = np.where(node_types == node_type)[0]
        n_type = min(len(type_indices), n_samples // 3)
        indices.extend(np.random.choice(type_indices, n_type, replace=False))
    
    indices = np.array(indices)
    print(f"\nSampling {len(indices)} nodes for visualization...")
    
    sampled_embeddings = embeddings[indices]
    sampled_types = node_types[indices]
    
    # t-SNE
    print(f"Running t-SNE (perplexity={args.perplexity})...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=42, max_iter=1000)
    coords_tsne = tsne.fit_transform(sampled_embeddings)
    
    # UMAP
    print("Running UMAP...")
    import umap
    reducer = umap.UMAP(random_state=42)
    coords_umap = reducer.fit_transform(sampled_embeddings)
    
    # Plot both
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    type_names = {1: 'Drug', 2: 'Disease', 3: 'Gene'}
    colors = {1: '#2ecc71', 2: '#e74c3c', 3: '#3498db'}
    
    # Plot t-SNE
    for node_type, name in type_names.items():
        mask = sampled_types == node_type
        ax1.scatter(
            coords_tsne[mask, 0], coords_tsne[mask, 1],
            c=colors[node_type], label=name, alpha=0.6, s=20
        )
    ax1.set_title(f't-SNE (perplexity={args.perplexity})')
    
    # Plot UMAP
    for node_type, name in type_names.items():
        mask = sampled_types == node_type
        ax2.scatter(
            coords_umap[mask, 0], coords_umap[mask, 1],
            c=colors[node_type], label=name, alpha=0.6, s=20
        )
    ax2.set_title('UMAP')

    # Highlight specific disease's drugs if requested
    if args.highlight_disease and args.highlight_disease in disease_mapping:
        disease_idx = disease_mapping[args.highlight_disease]
        
        # Find connected drugs
        edge_index = graph.edge_index.numpy()
        drug_indices = set(drug_mapping.values())
        
        connected_drugs = set()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if dst == disease_idx and src in drug_indices:
                connected_drugs.add(src)
            if src == disease_idx and dst in drug_indices:
                connected_drugs.add(dst)
        
        # Find these in our sample
        highlight_mask = np.array([
            indices[i] in connected_drugs for i in range(len(indices))
        ])
        
        if highlight_mask.sum() > 0:
            # Highlight on t-SNE
            ax1.scatter(
                coords_tsne[highlight_mask, 0], coords_tsne[highlight_mask, 1],
                c='yellow', edgecolors='black', s=100, linewidths=2,
                label=f'True drugs', zorder=10
            )
            # Highlight on UMAP
            ax2.scatter(
                coords_umap[highlight_mask, 0], coords_umap[highlight_mask, 1],
                c='yellow', edgecolors='black', s=100, linewidths=2,
                label=f'True drugs', zorder=10
            )
            print(f"Highlighted {highlight_mask.sum()} true drugs for {args.highlight_disease}")
    
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
