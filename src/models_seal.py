"""
SEAL: Subgraph Extraction and Labelling for Link Prediction

Implementation of the SEAL method which extracts local enclosing subgraphs
around candidate links and uses structural node labelling (DRNL) to enable
GNNs to learn link prediction from local topological patterns.

Reference
---------
Zhang & Chen (2018) — "Link Prediction Based on Graph Neural Networks" (NeurIPS)
"""

from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_max_pool, global_mean_pool
from torch_geometric.nn.aggr import SortAggregation
from torch_geometric.utils import k_hop_subgraph


# ═══════════════════════════════════════════════════════════════════════════
# Subgraph extraction utilities
# ═══════════════════════════════════════════════════════════════════════════

def drnl_node_labeling(
    num_nodes: int,
    src_node: int,
    dst_node: int,
    edge_index: torch.Tensor,
    max_z: int = 1000,
) -> torch.Tensor:
    """Compute Double Radius Node Labelling (DRNL) for a local subgraph.

    Each node is labelled based on its shortest-path distances to *src_node*
    and *dst_node* within the subgraph.
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.t().tolist())

    try:
        src_dist = nx.single_source_shortest_path_length(G, src_node)
    except Exception:
        src_dist = {}
    try:
        dst_dist = nx.single_source_shortest_path_length(G, dst_node)
    except Exception:
        dst_dist = {}

    z = []
    for node in range(num_nodes):
        d_s = src_dist.get(node, max_z)
        d_t = dst_dist.get(node, max_z)
        d = d_s + d_t

        if d_s == 0 and d_t == 0:
            label = 0
        elif d_s == 0 or d_t == 0:
            label = 1
        elif d >= max_z:
            label = 0
        else:
            label = 1 + min(d_s, d_t) + (d // 2) * ((d // 2) + (d % 2) - 1)
        z.append(label)

    return torch.tensor(z, dtype=torch.long)


def extract_enclosing_subgraph(
    src: int,
    dst: int,
    edge_index: torch.Tensor,
    node_features: Optional[torch.Tensor] = None,
    num_hops: int = 2,
    max_nodes_per_hop: Optional[int] = None,
    max_z: int = 100,
) -> Tuple[Data, int, int]:
    """Extract the *k*-hop enclosing subgraph around a candidate link.

    Returns ``(subgraph_data, rel_src, rel_dst)`` where ``rel_src`` and
    ``rel_dst`` are the *local* indices of the source and destination nodes.
    """
    # Remove the target link to ensure clean extraction
    mask = ~((edge_index[0] == src) & (edge_index[1] == dst))
    mask &= ~((edge_index[0] == dst) & (edge_index[1] == src))
    clean_edge_index = edge_index[:, mask]

    # k-hop neighbourhood
    nodes, local_edge_index, mapping, _ = k_hop_subgraph(
        node_idx=[src, dst],
        num_hops=num_hops,
        edge_index=clean_edge_index,
        relabel_nodes=True,
    )

    rel_src = mapping[0].item()
    rel_dst = mapping[1].item()
    n = nodes.size(0)

    # DRNL labels
    z = drnl_node_labeling(n, rel_src, rel_dst, local_edge_index, max_z=max_z)

    # Node features: concatenate original features with one-hot DRNL encoding
    z_onehot = F.one_hot(z.clamp(max=max_z - 1), num_classes=max_z).float()
    if node_features is not None:
        x = torch.cat([node_features[nodes].float(), z_onehot], dim=1)
    else:
        x = z_onehot

    subgraph_data = Data(x=x, edge_index=local_edge_index, z=z, num_nodes=n)
    return subgraph_data, rel_src, rel_dst


# ═══════════════════════════════════════════════════════════════════════════
# SEAL model
# ═══════════════════════════════════════════════════════════════════════════

class SEALModel(nn.Module):
    """SEAL: link prediction via subgraph classification.

    Applies a multi-layer GNN to each extracted subgraph, pools the node
    representations into a single graph-level vector, and classifies whether
    the candidate link exists.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        num_layers: int = 3,
        dropout_rate: float = 0.5,
        pooling: str = "sort",
        k: int = 30,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        self.k = k

        # GNN layers
        self.convs = nn.ModuleList([SAGEConv(in_channels, hidden_channels)])
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # Batch normalisation
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)

        # Readout layer
        if pooling == "sort":
            self.aggr = SortAggregation(k=k)
        elif pooling == "mean":
            self.aggr = global_mean_pool
        elif pooling == "max":
            self.aggr = global_max_pool
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        # MLP classifier
        mlp_in = k * hidden_channels if pooling == "sort" else hidden_channels
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass on a batch of subgraphs.

        Returns a 1-D tensor of logits (one per subgraph).
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)

        # Graph-level readout
        if self.pooling == "sort":
            x = self.aggr(x, batch)
        else:
            x = self.aggr(x, batch)

        return self.mlp(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# On-the-fly dataset
# ═══════════════════════════════════════════════════════════════════════════

class SEALDataset(TorchDataset):
    """Memory-efficient dataset that extracts SEAL subgraphs on-the-fly.

    Stores only the list of (src, dst) pairs and their labels; subgraphs are
    extracted in ``__getitem__`` to avoid keeping all subgraphs in memory.
    """

    def __init__(
        self,
        root: str,
        pairs: List[Tuple[int, int]],
        labels: List[int],
        edge_index: torch.Tensor,
        node_features: torch.Tensor,
        num_hops: int = 2,
        transform=None,
        pre_transform=None,
    ):
        self.pairs = pairs
        self.labels = labels
        self.edge_index = edge_index
        self.node_features = node_features
        self.num_hops = num_hops

        # Fallback feature dimension for dummy graphs
        self.feature_dim = 100  # max_z default
        if node_features is not None:
            self.feature_dim += node_features.size(1)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Data:
        src, dst = self.pairs[idx]
        label = self.labels[idx]

        try:
            subgraph, _, _ = extract_enclosing_subgraph(
                src, dst, self.edge_index, self.node_features, num_hops=self.num_hops,
            )
            subgraph.y = torch.tensor([label], dtype=torch.float)
            return subgraph
        except Exception:
            # Return a minimal dummy graph to preserve batch alignment
            return Data(
                x=torch.zeros((1, self.feature_dim), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                z=torch.zeros((1,), dtype=torch.long),
                y=torch.tensor([label], dtype=torch.float),
                num_nodes=1,
            )
