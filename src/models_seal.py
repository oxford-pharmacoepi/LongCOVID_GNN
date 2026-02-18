"""
SEAL: Subgraph Extraction and Labelling for Link Prediction

Implementation of the SEAL method which extracts local enclosing subgraphs
around candidate links and uses structural node labelling (DRNL) to enable
GNNs to learn link prediction from local topological patterns.

Reference
---------
Zhang & Chen (2018) — "Link Prediction Based on Graph Neural Networks" (NeurIPS)
"""

import hashlib
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GINConv, global_max_pool, global_mean_pool
from torch_geometric.nn.aggr import SortAggregation
from torch_geometric.utils import k_hop_subgraph


# Default max DRNL label — shared across labelling and feature construction
MAX_Z = 50


# ═══════════════════════════════════════════════════════════════════════════
# Subgraph extraction utilities
# ═══════════════════════════════════════════════════════════════════════════

def drnl_node_labeling(
    num_nodes: int,
    src_node: int,
    dst_node: int,
    edge_index: torch.Tensor,
    max_z: int = MAX_Z,
) -> torch.Tensor:
    """Compute Double Radius Node Labelling (DRNL) for a local subgraph.

    Uses scipy sparse BFS for fast shortest-path computation.
    Each node is labelled based on its shortest-path distances to *src_node*
    and *dst_node* within the subgraph.
    """
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    data = np.ones(len(row), dtype=np.float32)
    adj = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

    # BFS-based shortest paths from src and dst (much faster than NetworkX)
    src_dist_arr = shortest_path(adj, method="D", directed=False, indices=src_node)
    dst_dist_arr = shortest_path(adj, method="D", directed=False, indices=dst_node)

    z = torch.zeros(num_nodes, dtype=torch.long)
    for node in range(num_nodes):
        d_s = src_dist_arr[node]
        d_t = dst_dist_arr[node]

        if np.isinf(d_s):
            d_s = max_z
        else:
            d_s = int(d_s)
        if np.isinf(d_t):
            d_t = max_z
        else:
            d_t = int(d_t)

        d = d_s + d_t

        if d_s == 0 and d_t == 0:
            label = 0
        elif d_s == 0 or d_t == 0:
            label = 1
        elif d >= max_z:
            label = 0
        else:
            label = 1 + min(d_s, d_t) + (d // 2) * ((d // 2) + (d % 2) - 1)
        z[node] = label

    return z


def extract_enclosing_subgraph(
    src: int,
    dst: int,
    edge_index: torch.Tensor,
    node_features: Optional[torch.Tensor] = None,
    num_hops: int = 1,
    max_nodes_per_hop: Optional[int] = 200,
    max_z: int = MAX_Z,
    adj_dict: Optional[dict] = None,
) -> Tuple[Data, int, int]:
    """Extract the *k*-hop enclosing subgraph around a candidate link.

    Returns ``(subgraph_data, rel_src, rel_dst)`` where ``rel_src`` and
    ``rel_dst`` are the *local* indices of the source and destination nodes.

    If *adj_dict* is provided (node → set of neighbours), it is used for fast
    neighbour lookup instead of scanning all edges.
    """
    # Remove the target link to ensure clean extraction
    mask = ~((edge_index[0] == src) & (edge_index[1] == dst))
    mask &= ~((edge_index[0] == dst) & (edge_index[1] == src))
    clean_edge_index = edge_index[:, mask]

    if max_nodes_per_hop is not None:
        # Hop-by-hop expansion with neighbour sampling
        visited = {src, dst}
        frontier = {src, dst}
        for _ in range(num_hops):
            new_nb = set()
            if adj_dict is not None:
                # O(degree) lookup
                for node in frontier:
                    new_nb.update(adj_dict.get(node, set()))
            else:
                # Vectorised: find all edges touching frontier nodes
                frontier_t = torch.tensor(sorted(frontier), dtype=torch.long)
                s, d = clean_edge_index
                src_mask = torch.isin(s, frontier_t)
                dst_mask = torch.isin(d, frontier_t)
                nb_from_src = d[src_mask].tolist()
                nb_from_dst = s[dst_mask].tolist()
                new_nb.update(nb_from_src)
                new_nb.update(nb_from_dst)
            new_nb -= visited
            if len(new_nb) > max_nodes_per_hop:
                new_nb = set(random.sample(sorted(new_nb), max_nodes_per_hop))
            visited |= new_nb
            frontier = new_nb

        # Build local subgraph
        node_list = sorted(visited)
        node_tensor = torch.tensor(node_list, dtype=torch.long)
        node_map = {n: i for i, n in enumerate(node_list)}

        # Vectorised edge filtering
        s, d = clean_edge_index
        edge_mask = torch.isin(s, node_tensor) & torch.isin(d, node_tensor)
        kept_s = s[edge_mask]
        kept_d = d[edge_mask]

        # Relabel nodes
        remap = torch.zeros(int(node_tensor.max()) + 1, dtype=torch.long)
        remap[node_tensor] = torch.arange(len(node_list))
        local_edge_index = torch.stack([remap[kept_s], remap[kept_d]]) if kept_s.numel() > 0 else torch.zeros((2, 0), dtype=torch.long)

        nodes = node_tensor
        rel_src = node_map[src]
        rel_dst = node_map[dst]
    else:
        # Standard PyG k-hop subgraph (no sampling)
        nodes, local_edge_index, mapping, _ = k_hop_subgraph(
            node_idx=[src, dst],
            num_hops=num_hops,
            edge_index=clean_edge_index,
            relabel_nodes=True,
        )
        rel_src = mapping[0].item()
        rel_dst = mapping[1].item()

    n = nodes.size(0)

    # DRNL labels (uses same max_z as feature construction)
    z = drnl_node_labeling(n, rel_src, rel_dst, local_edge_index, max_z=max_z)

    # Features are constructed on-the-fly in SEALDataset to save disk space
    subgraph_data = Data(edge_index=local_edge_index, z=z, num_nodes=n, nodes=nodes)
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
        conv_type: str = "sage",
        use_jk: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        self.k = k
        self.use_jk = use_jk

        # GNN layers — choose convolution type
        def make_conv(in_c, out_c):
            if conv_type == "gcn":
                return GCNConv(in_c, out_c)
            elif conv_type == "gin":
                mlp = nn.Sequential(nn.Linear(in_c, out_c), nn.ReLU(), nn.Linear(out_c, out_c))
                return GINConv(mlp)
            elif conv_type == "gat":
                return GATConv(in_c, out_c, heads=1, concat=False)
            else:  # default: sage
                return SAGEConv(in_c, out_c)

        self.convs = nn.ModuleList([make_conv(in_channels, hidden_channels)])
        for _ in range(num_layers - 1):
            self.convs.append(make_conv(hidden_channels, hidden_channels))

        # Batch normalisation
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)

        # Skip-connection projection (if input dim != hidden dim)
        if in_channels != hidden_channels:
            self.skip_proj = nn.Linear(in_channels, hidden_channels, bias=False)
        else:
            self.skip_proj = None

        # Readout layer
        if pooling == "sort":
            self.sort_aggr = SortAggregation(k=k)
        self.pooling_name = pooling

        # Effective hidden dim for MLP: if JK, concat all layers
        effective_hidden = hidden_channels * num_layers if use_jk else hidden_channels

        # MLP classifier
        if pooling == "sort":
            mlp_in = k * effective_hidden
        elif pooling == "mean+max":
            mlp_in = effective_hidden * 2
        else:
            mlp_in = effective_hidden

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

        layer_outputs = []
        for i in range(self.num_layers):
            x_in = x
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)
            # Skip / residual connection
            if i == 0 and self.skip_proj is not None:
                x = x + self.skip_proj(x_in)
            elif i > 0:
                x = x + x_in
            if self.use_jk:
                layer_outputs.append(x)

        # JKNet: concatenate all layer outputs
        if self.use_jk:
            x = torch.cat(layer_outputs, dim=-1)

        # Graph-level readout
        if self.pooling_name == "sort":
            x = self.sort_aggr(x, batch)
        elif self.pooling_name == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling_name == "max":
            x = global_max_pool(x, batch)
        elif self.pooling_name == "mean+max":
            x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling_name}")

        return self.mlp(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# On-the-fly dataset
# ═══════════════════════════════════════════════════════════════════════════

class SEALDataset(TorchDataset):
    """Memory-efficient dataset that extracts SEAL subgraphs on-the-fly.

    Stores only the list of (src, dst) pairs and their labels; subgraphs are
    extracted in ``__getitem__`` to avoid keeping all subgraphs in memory.
    
    Includes persistent caching to allow fast re-runs and fast epochs after the first.
    """

    def __init__(
        self,
        root: str,
        pairs: List[Tuple[int, int]],
        labels: List[int],
        edge_index: torch.Tensor,
        node_features: torch.Tensor,
        num_hops: int = 1,
        max_z: int = MAX_Z,
        max_nodes_per_hop: int = 200,
        adj_dict: Optional[dict] = None,
        use_cache: bool = True,
        save_cache: bool = True,
        transform=None,
        pre_transform=None,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pairs = pairs
        self.labels = labels
        self.edge_index = edge_index
        self.node_features = node_features
        self.num_hops = num_hops
        self.max_z = max_z
        self.max_nodes_per_hop = max_nodes_per_hop
        self.adj_dict = adj_dict
        self.use_cache = use_cache
        self.save_cache = save_cache

        # Fallback feature dimension for dummy graphs
        self.feature_dim = max_z
        if node_features is not None:
            self.feature_dim += node_features.size(1)

        # Cache hash uses shape + content signature to avoid stale hits
        edge_sig = f"{edge_index.shape}_{edge_index[:, :5].tolist()}_{edge_index[:, -5:].tolist()}"
        graph_hash = hashlib.md5(edge_sig.encode()).hexdigest()[:12]
        self.cache_dir = self.root / f"cache_{graph_hash}_hops{num_hops}_mnph{max_nodes_per_hop}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"  SEAL Cache: {self.cache_dir}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Data:
        src, dst = self.pairs[idx]
        label = self.labels[idx]
        max_z = self.max_z
        
        # Check cache
        cache_file = self.cache_dir / f"subgraph_{src}_{dst}.pt"
        subgraph = None
        
        if self.use_cache and cache_file.exists():
            try:
                subgraph = torch.load(cache_file, weights_only=False)
            except Exception:
                pass

        if subgraph is None:
            try:
                subgraph, _, _ = extract_enclosing_subgraph(
                    src, dst, self.edge_index, None,
                    num_hops=self.num_hops, max_z=max_z,
                    max_nodes_per_hop=self.max_nodes_per_hop,
                    adj_dict=self.adj_dict,
                )
                # Save minimal topo to cache
                if self.save_cache:
                    torch.save(subgraph, cache_file)
            except Exception:
                # Return a minimal dummy graph to preserve batch alignment
                return Data(
                    x=torch.zeros((1, self.feature_dim), dtype=torch.float),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    z=torch.zeros((1,), dtype=torch.long),
                    y=torch.tensor([label], dtype=torch.float),
                    num_nodes=1,
                )

        # Reconstruct features on-the-fly
        z = subgraph.z
        z_onehot = F.one_hot(z.clamp(max=max_z - 1), num_classes=max_z).float()
        if self.node_features is not None:
            nodes = subgraph.nodes
            subgraph.x = torch.cat([self.node_features[nodes].float(), z_onehot], dim=1)
        else:
            subgraph.x = z_onehot
            
        subgraph.y = torch.tensor([label], dtype=torch.float)

        # Remove helper attrs not needed for batching (avoids KeyError in collation)
        if hasattr(subgraph, 'nodes'):
            del subgraph.nodes

        return subgraph

    def precache_parallel(self, num_workers: int = 0):
        """Pre-extract and cache all subgraphs using parallel threads.

        Skips pairs already cached on disk.  Uses threads (not processes)
        to share the adj_dict without serialisation overhead.

        Parameters
        ----------
        num_workers : int
            Number of threads.  0 = auto (cpu_count - 1, capped at 8).
        """
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if num_workers <= 0:
            num_workers = min(os.cpu_count() - 1, 8)

        # Find indices that still need caching
        uncached = []
        for idx in range(len(self)):
            src, dst = self.pairs[idx]
            cache_file = self.cache_dir / f"subgraph_{src}_{dst}.pt"
            if not cache_file.exists():
                uncached.append(idx)

        if not uncached:
            print(f"  All {len(self)} subgraphs already cached.")
            return

        print(f"  Pre-caching {len(uncached)}/{len(self)} subgraphs "
              f"using {num_workers} threads...")

        done = 0
        errors = 0

        def _extract_one(idx):
            """Extract a single subgraph (triggers caching via __getitem__)."""
            try:
                _ = self[idx]
                return True
            except Exception:
                return False

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_extract_one, i): i for i in uncached}
            for future in as_completed(futures):
                if future.result():
                    done += 1
                else:
                    errors += 1
                if (done + errors) % 500 == 0:
                    print(f"    Cached {done + errors}/{len(uncached)} "
                          f"({errors} errors)")

        print(f"  Pre-caching complete: {done} OK, {errors} errors.")

