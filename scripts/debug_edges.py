
import torch
import torch_geometric
import os

def check_graph(path, name):
    print(f"Loading {name} from {path}...")
    if not os.path.exists(path):
        print(f"Error: {path} does not exist")
        return None
        
    graph = torch.load(path, weights_only=False)
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.num_edges}")
    
    print(f"  Type: {type(graph)}")
    edge_counts = {}
    
    # Node counts and ranges (Verified from logs)
    # Order: Drug, DrugType, Gene, Reactome, Disease, TherapeuticArea
    # Note: These must define the SEGMENTS of the node index space
    
    # We need exact counts to build ranges. 
    # Provided from logs:
    counts = {
        'Drug': 2471,
        'DrugType': 9,
        'Gene': 60605,
        'Reactome': 1985,  # Assuming this matches
        'Disease': 9339,
        'Therapeutic': 24  # Assuming this matches
    }
    
    # Calculate cumulative offsets
    offsets = {}
    current = 0
    ordered_types = ['Drug', 'DrugType', 'Gene', 'Reactome', 'Disease', 'Therapeutic']
    
    ranges = []
    print("  Node Ranges:")
    for ntype in ordered_types:
        count = counts[ntype]
        offsets[ntype] = (current, current + count)
        ranges.append((current, current + count, ntype))
        print(f"    {ntype}: [{current}, {current + count})")
        current += count
        
    print(f"    Total covered: {current} (Graph has {graph.num_nodes})")
    
    # Count edges by type
    print("  Counting edges by inferred type (this may take a moment)...")
    src_indices = graph.edge_index[0].tolist()
    dst_indices = graph.edge_index[1].tolist()
    
    from collections import defaultdict
    type_pair_counts = defaultdict(int)
    
    def get_type(idx):
        for start, end, ntype in ranges:
            if start <= idx < end:
                return ntype
        return "Unknown"

    for s, d in zip(src_indices, dst_indices):
        st = get_type(s)
        dt = get_type(d)
        if st > dt: st, dt = dt, st # Canonize order (undirected)
        type_pair_counts[f"{st}-{dt}"] += 1
        
    for pair, count in sorted(type_pair_counts.items()):
        edge_counts[pair] = count
        print(f"    {pair}: {count}")
        
    return edge_counts

print("="*50)
baseline_path = "results/graph_21.06_raw_20260128_121309.pt"
new_path = "results/graph_21.06_raw_20260128_130922.pt"

base_counts = check_graph(baseline_path, "Baseline")
print("-" * 30)
new_counts = check_graph(new_path, "New Graph")

print("="*50)
print("COMPARISON (Baseline vs New):")
if base_counts and new_counts:
    for etype in base_counts:
        b_count = base_counts.get(etype, 0)
        n_count = new_counts.get(etype, 0)
        diff = n_count - b_count
        status = "✅" if diff == 0 else "❌"
        print(f"{status} {etype}: {b_count} vs {n_count} (Diff: {diff})")
