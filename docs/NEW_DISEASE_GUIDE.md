# Adapting the Pipeline for a New Disease

This guide explains how to use the SEAL drug repurposing pipeline for a disease of your choice. There are two main scenarios:

1. **Your disease already has drug edges** in the Open Targets graph (e.g., psoriasis, type 2 diabetes) → Use **Leave-One-Out (LOO) validation**.
2. **Your disease has no drug edges** (e.g., Long COVID) → Use the **cold-start prediction** workflow.

---

## Prerequisites

```bash
# Clone the repository and install dependencies
git clone https://github.com/spet5356/LongCOVID_GNN.git
cd LongCOVID_GNN
uv sync

# Ensure the graph file exists (built by the pipeline)
ls results/graph_*.pt
```

You need a pre-built graph file (e.g., `results/graph_21.06.pt`). If you don't have one, run:

```bash
uv run python run_pipeline.py --step graph
```

---

## Step 1: Find Your Disease ID

Every disease in Open Targets has an **EFO** or **MONDO** identifier. To find yours:

1. Go to [Open Targets Platform](https://platform.opentargets.org/)
2. Search for your disease (e.g., "psoriasis")
3. Click on the disease page
4. Note the disease ID from the URL — e.g., `EFO_0000676` for Psoriasis

### Check your disease is in the graph

```python
import torch
data = torch.load("results/graph_21.06.pt")

# List all disease IDs in the graph
disease_ids = [data.node_id_map_inv.get(i, "?")
               for i in range(data.num_nodes)
               if data.node_type[i] == 2]  # type 2 = disease

# Check if your disease is present
target = "EFO_0000676"  # Psoriasis
print(f"{target} in graph: {target in disease_ids}")
```

---

## Scenario A: LOO Validation (Disease Has Drug Edges)

This is the main evaluation mode. It trains SEAL on all drug–disease edges *except* those for your target disease, then ranks all 2,471 drugs for that disease.

### Basic run

```bash
uv run python scripts/seal/train_loo.py \
    --target-disease EFO_0000676 \
    --seed 42 \
    --use-jk
```

### Quick smoke test (1 epoch, ~2 min)

```bash
uv run python scripts/seal/train_loo.py \
    --target-disease EFO_0000676 \
    --epochs 1 \
    --patience 0 \
    --no-mlflow \
    --use-jk
```

### Full run with best settings

```bash
uv run python scripts/seal/train_loo.py \
    --target-disease EFO_0000676 \
    --epochs 50 \
    --hidden 32 \
    --layers 3 \
    --hops 2 \
    --conv sage \
    --use-jk \
    --neg-strategy mixed \
    --neg-ratio 3 \
    --hard-ratio 0.5 \
    --seed 42
```

### Multi-seed run (recommended)

```bash
for SEED in 42 123 7; do
    uv run python scripts/seal/train_loo.py \
        --target-disease EFO_0000676 \
        --seed $SEED \
        --use-jk
done
```

### Key CLI arguments

| Argument | Default | Description |
|:---|:---:|:---|
| `--target-disease` | `EFO_0003854` | EFO/MONDO ID of the disease to evaluate |
| `--epochs` | `50` | Training epochs |
| `--hidden` | `32` | Hidden dimension (32 > 64 in our ablation) |
| `--hops` | `2` | Subgraph extraction hops (2 is the sweet spot) |
| `--conv` | `sage` | Convolution type: `sage`, `gat`, `gin` |
| `--use-jk` | off | Enable Jumping Knowledge (recommended with sage) |
| `--neg-strategy` | `mixed` | Negative sampling: `random`, `hard`, `mixed` |
| `--patience` | `10` | Early stopping patience |
| `--no-mlflow` | off | Disable MLflow logging |
| `--seed` | `42` | Random seed |

### Understanding results

Results are saved to `results/seal_results/` as a JSON file. Key metrics:

- **Hits@K** (K = 10, 20, 50, 100): How many true drugs appear in the top K predictions. Higher is better.
- **Median Rank**: Median position of true drugs in the full ranking (out of 2,471). Lower is better.
- **MRR**: Mean Reciprocal Rank. Higher is better.
- **Val AUC**: Validation AUC (note: can be misleading — see paper §3.10).

**Rule of thumb**: Median Rank < 100 is excellent, 100–300 is good, > 500 is challenging (likely a broad disease).

---

## Scenario B: Cold-Start Prediction (No Drug Edges)

If your disease has no approved drug indications in the Open Targets graph, you need to provide **gene targets** as structural bridges between the disease and the drug space. This is what we did for Long COVID.

### Step 1: Curate a gene list

Create a file listing genes associated with your disease. Each line should contain: gene symbol, Ensembl ID, and source category. Example:

```
# my_disease_genes.txt
GENE_SYMBOL    ENSEMBL_ID          CATEGORY
FOXP4          ENSG00000174207     GWAS
HLA-DQA1       ENSG00000196735     GWAS
ACE2           ENSG00000130234     mechanism
```

**Tips for gene curation:**
- Start with **GWAS lead signals** — highest confidence, least noisy
- Consider adding causal inference genes and mechanism-of-action genes
- **Exclude mega-hub genes** (degree > 300 in the Open Targets graph, e.g., TP53, EP300) — these dominate SEAL subgraphs and dilute signal
- Fewer, more specific genes tends to outperform larger gene sets (see paper §3.11)

### Step 2: Run the cold-start prediction

Adapt `scripts/seal/predict_long_covid.py` for your disease. The key change is the MONDO/EFO ID and gene file:

```bash
uv run python scripts/seal/predict_long_covid.py \
    --gene-file my_disease_genes.txt \
    --seed 42
```

For a multi-seed consensus:

```bash
for SEED in 7 42 123 456 789; do
    uv run python scripts/seal/predict_long_covid.py \
        --gene-file my_disease_genes.txt \
        --seed $SEED
done

# Aggregate results
uv run python scripts/seal/aggregate_lc_results.py
```

### Step 3: Interpret results

The consensus script produces a ranked drug list. Drugs appearing in ≥3/5 seeds are high-confidence candidates. Cross-reference against ClinicalTrials.gov for independent validation.

---

## Common Issues

| Problem | Solution |
|:---|:---|
| Disease ID not found | Check the exact EFO/MONDO ID on Open Targets. Some diseases have subtypes. |
| 0 true drugs found | Your disease may have no drug edges → use Scenario B (cold-start). |
| Very poor performance (median rank > 1000) | Likely a broadly-treated disease with many drugs. Try GAT instead of SEAL. |
| Out of memory | Reduce `--max-nodes-per-hop` (default 200) or use `--hops 1`. |
| Slow subgraph extraction | First run caches subgraphs to disk. Subsequent runs are faster. |

---

## Example: Running for Psoriasis

```bash
# 1. Quick smoke test
uv run python scripts/seal/train_loo.py \
    --target-disease EFO_0000676 \
    --epochs 1 \
    --patience 0 \
    --no-mlflow \
    --use-jk

# 2. Full 3-seed evaluation
for SEED in 42 123 7; do
    uv run python scripts/seal/train_loo.py \
        --target-disease EFO_0000676 \
        --seed $SEED \
        --use-jk \
        --epochs 50
done
```

Expected output: a JSON results file containing Hits@K, median rank, MRR, and the top-ranked drugs.
