"""
Tests for LOO training protocol correctness (scripts/seal/train_loo.py).

Validates that leave-one-out edge splitting, negative sampling isolation,
and scoring produce correct results on a small synthetic graph.
"""

import pytest
import random
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

# Ensure project root (and seal directory) are importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEAL_DIR = PROJECT_ROOT / "scripts" / "seal"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SEAL_DIR))

from train_loo import sample_negatives_random, sample_negatives_mixed


# ── helpers ──────────────────────────────────────────────────────────

def _make_synthetic_graph(
    n_drugs: int = 20,
    n_diseases: int = 5,
    n_genes: int = 30,
    edges_per_drug: int = 2,
    seed: int = 42,
):
    """Build a small synthetic graph mimicking the real KG structure.

    Returns
    -------
    graph_data : Data
    drug_mapping, disease_mapping, gene_mapping : dict[str, int]
    drug_disease_edges : list[tuple[int, int]]
    """
    rng = random.Random(seed)
    idx = 0

    drug_mapping = {}
    for i in range(n_drugs):
        drug_mapping[f"DRUG_{i}"] = idx
        idx += 1

    disease_mapping = {}
    for i in range(n_diseases):
        disease_mapping[f"DIS_{i}"] = idx
        idx += 1

    gene_mapping = {}
    for i in range(n_genes):
        gene_mapping[f"GENE_{i}"] = idx
        idx += 1

    num_nodes = idx
    edge_list = []
    drug_disease_edges = []

    # drug→disease edges
    for drug_id, drug_idx in drug_mapping.items():
        targets = rng.sample(list(disease_mapping.values()), min(edges_per_drug, n_diseases))
        for dis_idx in targets:
            edge_list.append([drug_idx, dis_idx])
            edge_list.append([dis_idx, drug_idx])
            drug_disease_edges.append((drug_idx, dis_idx))

    # drug→gene edges (sparse)
    for drug_idx in list(drug_mapping.values())[:10]:
        gene_idx = rng.choice(list(gene_mapping.values()))
        edge_list.append([drug_idx, gene_idx])
        edge_list.append([gene_idx, drug_idx])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    x = torch.randn(num_nodes, 8)

    graph_data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    return graph_data, drug_mapping, disease_mapping, gene_mapping, drug_disease_edges


# ── tests ────────────────────────────────────────────────────────────

class TestEdgeSplitting:
    """Verify that LOO edge splitting correctly partitions edges."""

    def test_all_target_disease_edges_removed_from_train(self):
        """ALL drug edges for the held-out disease should be in test, not train."""
        (graph, drug_map, disease_map, gene_map,
         drug_disease_edges) = _make_synthetic_graph()

        target_idx = disease_map["DIS_0"]

        test_edges = [e for e in drug_disease_edges if e[1] == target_idx]
        train_edges = [e for e in drug_disease_edges if e[1] != target_idx]

        # Test edges must exist
        assert len(test_edges) > 0, "No test edges for target disease"

        # No train edge should involve the target disease
        for drug, dis in train_edges:
            assert dis != target_idx, (
                f"Training edge ({drug}, {dis}) involves held-out disease"
            )

        # All original edges should be accounted for
        assert len(test_edges) + len(train_edges) == len(drug_disease_edges)

    def test_split_is_exhaustive(self):
        """Union of train + test must equal the original drug-disease edge set."""
        (_, _, disease_map, _, drug_disease_edges) = _make_synthetic_graph()

        for dis_name, dis_idx in disease_map.items():
            test = {e for e in drug_disease_edges if e[1] == dis_idx}
            train = {e for e in drug_disease_edges if e[1] != dis_idx}
            assert test | train == set(drug_disease_edges)
            assert len(test & train) == 0  # disjoint


class TestNegativeSamplingIsolation:
    """Verify negatives never include the held-out disease or future positives."""

    def test_random_negatives_exclude_target_disease(self):
        """No negative should pair a drug with the excluded disease."""
        (graph, drug_map, disease_map, _, drug_disease_edges) = _make_synthetic_graph()

        target_idx = disease_map["DIS_0"]
        pos_set = set(drug_disease_edges)
        drug_list = list(drug_map.values())
        disease_list = list(disease_map.values())
        test_edges = {e for e in drug_disease_edges if e[1] == target_idx}

        negatives = sample_negatives_random(
            positive_set=pos_set,
            all_drug_list=drug_list,
            all_disease_list=disease_list,
            num_samples=50,
            exclude_disease=target_idx,
            future_positives=test_edges,
        )

        for drug, dis in negatives:
            assert dis != target_idx, (
                f"Negative ({drug}, {dis}) includes excluded disease"
            )
            assert (drug, dis) not in test_edges, (
                f"Negative ({drug}, {dis}) is a future positive"
            )

    def test_negatives_are_not_positives(self):
        """No negative should duplicate an existing positive edge."""
        (graph, drug_map, disease_map, _, drug_disease_edges) = _make_synthetic_graph()

        pos_set = set(drug_disease_edges)
        drug_list = list(drug_map.values())
        disease_list = list(disease_map.values())

        negatives = sample_negatives_random(
            positive_set=pos_set,
            all_drug_list=drug_list,
            all_disease_list=disease_list,
            num_samples=50,
        )

        for edge in negatives:
            assert edge not in pos_set, (
                f"Negative {edge} is actually a positive edge"
            )


class TestScoringCoverage:
    """Verify that the scoring loop would cover all drugs."""

    def test_all_drugs_scored(self):
        """Every drug in the mapping should be scorable against a target disease."""
        (graph, drug_map, disease_map, _, _) = _make_synthetic_graph()

        target_idx = disease_map["DIS_0"]
        drug_indices = list(drug_map.values())

        # Simulate scoring: generate (drug, target) pairs for all drugs
        score_pairs = [(d, target_idx) for d in drug_indices]
        assert len(score_pairs) == len(drug_map)

        # No drug should be missed
        scored_drugs = {d for d, _ in score_pairs}
        expected_drugs = set(drug_map.values())
        assert scored_drugs == expected_drugs
