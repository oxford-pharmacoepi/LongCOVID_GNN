#!/usr/bin/env python3
"""
Visualise SEAL Subgraphs

Extracts and visualises the local subgraph around a drug-disease pair,
colouring nodes by their DRNL labels and biological types.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import torch

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models_seal import extract_enclosing_subgraph


def visualise_link_subgraph(drug_id: str, disease_id: str, num_hops: int = 2,
                            save_path: str = None):
    """Extract and plot the enclosing subgraph for a (drug, disease) pair."""

    # Load latest graph and mappings
    graph_files = sorted(glob.glob("results/graph_*_processed_*.pt"))
    graph_path = graph_files[-1]
    graph_data = torch.load(graph_path, weights_only=False)

    mappings_path = graph_path.replace(".pt", "_mappings")
    with open(f"{mappings_path}/drug_key_mapping.json") as f:
        drug_mapping = json.load(f)
    with open(f"{mappings_path}/disease_key_mapping.json") as f:
        disease_mapping = json.load(f)

    drug_idx = int(drug_mapping[drug_id])
    disease_idx = int(disease_mapping[disease_id])

    # Extract subgraph
    subgraph_data, src_rel, dst_rel = extract_enclosing_subgraph(
        drug_idx, disease_idx, graph_data.edge_index, graph_data.x, num_hops=num_hops
    )

    # Convert to NetworkX
    G = nx.Graph()
    ei = subgraph_data.edge_index.numpy()
    for i in range(ei.shape[1]):
        G.add_edge(ei[0, i], ei[1, i])

    # Node labels and colours
    labels = {}
    node_colours = []
    for i in range(subgraph_data.num_nodes):
        z = subgraph_data.z[i].item()
        if i == src_rel:
            labels[i] = f"{drug_id}\n(DRUG)"
            node_colours.append("#ff4d4d")
        elif i == dst_rel:
            labels[i] = f"{disease_id}\n(DISEASE)"
            node_colours.append("#4d79ff")
        else:
            labels[i] = f"z={z}"
            node_colours.append("#e0e0e0")

    # Draw
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color=node_colours, node_size=2000, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color="#bdbdbd", alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight="bold")
    plt.title(f"SEAL Subgraph: {drug_id} → {disease_id} (h={num_hops})", fontsize=15)
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Visualisation saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise SEAL subgraphs")
    parser.add_argument("--drug", type=str, default="CHEMBL1237023", help="Drug CHEMBL ID")
    parser.add_argument("--disease", type=str, default="EFO_0003854", help="Disease EFO/MONDO ID")
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--save", type=str, default=None, help="Save path for the figure")
    args = parser.parse_args()

    visualise_link_subgraph(
        args.drug, args.disease, num_hops=args.hops,
        save_path=args.save or f"results/seal_subgraph_{args.drug}.png",
    )
