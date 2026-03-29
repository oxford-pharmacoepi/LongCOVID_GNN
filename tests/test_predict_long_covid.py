"""
Tests for Long COVID prediction pipeline (scripts/seal/predict_long_covid.py).

Validates gene file parsing, category filtering, and gene wiring logic.
"""

import pytest
import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEAL_DIR = PROJECT_ROOT / "scripts" / "seal"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SEAL_DIR))

from predict_long_covid import load_gwas_genes, ALL_CATEGORIES


# ── load_gwas_genes ──────────────────────────────────────────────────

class TestLoadGwasGenes:
    """Tests for the gene file parser."""

    def test_tabular_format(self, tmp_path):
        """Parse a tab-separated gene file with header."""
        gene_file = tmp_path / "genes.txt"
        gene_file.write_text(
            "# Test genes\n"
            "gene_id\tensmbl_id\tgroup\n"
            "FOXP4\tENSG00000137166\tcore\n"
            "HLA-DQA1\tENSG00000196735\tcore\n"
            "SLC6A20\tENSG00000163817\tbroader\n"
        )
        genes = load_gwas_genes(str(gene_file))

        assert len(genes) == 3
        # Each gene is (ensembl_id, category, symbol)
        assert genes[0] == ("ENSG00000137166", "core", "FOXP4")
        assert genes[1] == ("ENSG00000196735", "core", "HLA-DQA1")
        assert genes[2] == ("ENSG00000163817", "broader", "SLC6A20")

    def test_empty_file(self, tmp_path):
        """An empty file (header only) should return no genes."""
        gene_file = tmp_path / "empty.txt"
        gene_file.write_text(
            "# Empty control\n"
            "gene_id\tensmbl_id\tgroup\n"
        )
        genes = load_gwas_genes(str(gene_file))
        assert len(genes) == 0

    def test_comments_skipped(self, tmp_path):
        """Lines starting with # should be ignored."""
        gene_file = tmp_path / "comments.txt"
        gene_file.write_text(
            "# This is a comment\n"
            "# Another comment\n"
            "gene_id\tensmbl_id\tgroup\n"
            "# Inline comment\n"
            "FOXP4\tENSG00000137166\tcore\n"
        )
        genes = load_gwas_genes(str(gene_file))
        assert len(genes) == 1

    def test_legacy_format(self, tmp_path):
        """Parse the legacy whitespace-delimited format without headers."""
        gene_file = tmp_path / "legacy.txt"
        gene_file.write_text(
            "# === GWAS Lead Signals ===\n"
            "ENSG00000137166  # FOXP4 | respiratory\n"
            "ENSG00000196735  # HLA-DQA1 | immune\n"
        )
        genes = load_gwas_genes(str(gene_file))
        assert len(genes) == 2
        # Legacy format: (gene_id, category, symbol)
        assert genes[0][0] == "ENSG00000137166"
        assert genes[0][1] == "gwas"  # mapped from header
        assert genes[0][2] == "FOXP4"

    def test_real_gwas_file(self):
        """Parse the actual gwas_genes_long_covid.txt file."""
        gwas_path = PROJECT_ROOT / "gwas_genes_long_covid.txt"
        if not gwas_path.exists():
            pytest.skip("gwas_genes_long_covid.txt not found")

        genes = load_gwas_genes(str(gwas_path))
        assert len(genes) > 0, "No genes parsed from real file"

        # All should have valid ensembl IDs
        for ens_id, cat, symbol in genes:
            assert ens_id.startswith("ENSG"), (
                f"Invalid Ensembl ID: {ens_id}"
            )


class TestCategoryFiltering:
    """Tests for category-based gene filtering."""

    def test_filter_core_only(self, tmp_path):
        """Filtering to 'core' excludes 'broader' genes."""
        gene_file = tmp_path / "genes.txt"
        gene_file.write_text(
            "gene_id\tensmbl_id\tgroup\n"
            "FOXP4\tENSG00000137166\tcore\n"
            "HLA-DQA1\tENSG00000196735\tcore\n"
            "SLC6A20\tENSG00000163817\tbroader\n"
        )
        genes = load_gwas_genes(str(gene_file))

        allowed = {"core"}
        filtered = [(g, c, s) for g, c, s in genes if c in allowed]
        assert len(filtered) == 2
        assert all(c == "core" for _, c, _ in filtered)

    def test_all_categories_allows_everything(self, tmp_path):
        """Using ALL_CATEGORIES should keep all valid genes."""
        gene_file = tmp_path / "genes.txt"
        gene_file.write_text(
            "gene_id\tensmbl_id\tgroup\n"
            "FOXP4\tENSG00000137166\tcore\n"
            "SLC6A20\tENSG00000163817\tbroader\n"
        )
        genes = load_gwas_genes(str(gene_file))
        filtered = [(g, c, s) for g, c, s in genes if c in ALL_CATEGORIES]
        assert len(filtered) == 2


class TestGeneWiring:
    """Tests for the gene wiring logic."""

    def test_wiring_adds_bidirectional_edges(self):
        """Wiring a gene should add two edges (both directions)."""
        # Simulate wiring logic
        lc_idx = 100
        gene_idx = 50
        new_edges = [[lc_idx, gene_idx], [gene_idx, lc_idx]]

        edge_tensor = torch.tensor(new_edges, dtype=torch.long).t()
        assert edge_tensor.shape == (2, 2)
        assert edge_tensor[0, 0] == lc_idx
        assert edge_tensor[1, 0] == gene_idx
        assert edge_tensor[0, 1] == gene_idx
        assert edge_tensor[1, 1] == lc_idx

    def test_wiring_with_empty_genes_adds_no_edges(self):
        """An empty gene list should not add any edges."""
        genes = []
        gene_mapping = {"ENSG000001": 50, "ENSG000002": 51}
        new_edges = []

        for gene_id, category, symbol in genes:
            gene_idx = gene_mapping.get(gene_id)
            if gene_idx is not None:
                new_edges.append([100, gene_idx])
                new_edges.append([gene_idx, 100])

        assert len(new_edges) == 0

    def test_wiring_skips_unmapped_genes(self):
        """Genes not in the mapping should be silently skipped."""
        genes = [
            ("ENSG_MISSING", "core", "FAKE"),
            ("ENSG000001", "core", "REAL"),
        ]
        gene_mapping = {"ENSG000001": 50}
        lc_idx = 100
        new_edges = []

        for gene_id, category, symbol in genes:
            gene_idx = gene_mapping.get(gene_id)
            if gene_idx is not None:
                new_edges.append([lc_idx, gene_idx])
                new_edges.append([gene_idx, lc_idx])

        assert len(new_edges) == 2  # Only the mapped gene

    def test_hub_gene_exclusion(self):
        """Genes exceeding the drug-connection cap should be excluded."""
        from collections import defaultdict

        lc_idx = 100
        gene_idx = 50
        max_drug_per_gene = 3

        # Simulate adjacency: gene_idx connects to 5 drugs
        adj = defaultdict(set)
        drug_set = {0, 1, 2, 3, 4}
        for d in drug_set:
            adj[gene_idx].add(d)
            adj[d].add(gene_idx)

        drug_connections = len(adj.get(gene_idx, set()) & drug_set)
        assert drug_connections == 5
        assert drug_connections > max_drug_per_gene

        # This gene should be excluded
        new_edges = []
        if drug_connections <= max_drug_per_gene:
            new_edges.append([lc_idx, gene_idx])
        assert len(new_edges) == 0  # Excluded
