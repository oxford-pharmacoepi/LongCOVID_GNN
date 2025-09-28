# KG-Bench: Benchmarking Graph Neural Networks for Drug Repurposing

This repository accompanies the project "KG-Bench: Benchmarking Graph Neural Network Algorithms for Drug Repurposing". It introduces KG-Bench, a FAIR-compliant benchmarking framework for evaluating Graph Neural Network (GNN) architectures in the context of drug repurposing, using the Open Targets dataset.

Drug repurposing—finding new therapeutic uses for existing drugs—is a promising strategy to accelerate drug development. KG-Bench presents a systematic approach to evaluate GNN models on drug–disease association prediction tasks using knowledge graphs (KGs) constructed from Open Targets data. The framework addresses key challenges such as:

+ Lack of standardized benchmarks
+ Data leakage between training and test sets
+ Imbalanced learning scenarios due to sparse negative samples

The framework supports retrospective validation using time-stamped versions of the Open Targets dataset, enabling realistic evaluation of model generalization to newly reported drug–disease associations.

## This GitHub repository provides:

+ Scripts to construct biomedical knowledge graphs from Open Targets data
+ Preprocessed datasets for training, validation, and testing
+ Implementations of GNN models: GCNConv, GraphSAGE, and TransformerConv
+ Benchmarking pipeline with ablation studies and negative sampling strategies
+ Evaluation metrics including AUC, precision-recall curves, and more

## Project Structure

```
drug_disease_prediction/
├── src/                           # Shared modules
│   ├── __init__.py               # Package initialization
│   ├── models.py                 # GNN model definitions (GCN, GraphSAGE, Transformer)
│   ├── utils.py                  # Utility functions and evaluation metrics
│   ├── config.py                 # Configuration management
│   └── data_processing.py        # Data loading & preprocessing
│
├── scripts/                      # Main pipeline scripts
│   ├── 1_create_graph.py         # Knowledge graph construction
│   ├── 2_train_models.py         # Model training and validation
│   ├── 3_test_evaluate.py        # Model testing and evaluation
│   └── 4_explain_predictions.py  # GNN explanation analysis
│
├── processed_data/               # Pre-processed data files
├── run_pipeline.py               # Main pipeline orchestrator
├── requirements.txt              # Python dependencies
├── config.json                   # Configuration file
└── README.md                     # This file
```

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd drug_disease_prediction

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Setup

**👥 Most users should choose Option 1** for the quickest setup. Only choose Option 2 if you need to work with raw OpenTargets data or want to understand the full data processing pipeline.

---

### 🚀 Option 1: Use Pre-processed Data (Recommended - Quick Start)

**Best for:** Getting started quickly, running experiments, most research use cases

The repository includes pre-processed data files ready for immediate use. No additional downloads required!

**What you get:**
- Filtered and cleaned drug, disease, and gene datasets
- Pre-built knowledge graph edges
- Ready-to-use training/validation/test splits
- Mapping files for all entities

**Expected directory structure:**
```
processed_data/
├── tables/
│   ├── processed_molecules.csv     # Filtered drug molecules
│   ├── processed_indications.csv   # Drug-disease indications
│   ├── processed_diseases.csv      # Filtered diseases
│   ├── processed_genes.csv         # Target genes
│   └── processed_associations.csv  # Gene-disease associations
├── mappings/
│   ├── drug_key_mapping.json       # Drug ID to node index
│   ├── drug_type_key_mapping.json  # Drug type mappings
│   ├── gene_key_mapping.json       # Gene ID mappings
│   ├── reactome_key_mapping.json   # Pathway mappings
│   ├── disease_key_mapping.json    # Disease ID mappings
│   ├── therapeutic_area_key_mapping.json # Therapeutic area mappings
│   └── mapping_summary.json        # Node count summary
└── edges/
    ├── 1_molecule_drugType_edges.pt   # Drug-DrugType edges
    ├── 2_molecule_disease_edges.pt    # Drug-Disease edges  
    ├── 3_molecule_gene_edges.pt       # Drug-Gene edges
    ├── 4_gene_reactome_edges.pt       # Gene-Pathway edges
    ├── 5_disease_therapeutic_edges.pt # Disease-TherapeuticArea edges
    ├── 6_disease_gene_edges.pt        # Disease-Gene edges
    ├── edge_statistics.json           # Edge count summary
    └── training_drug_disease_pairs.csv # Training pairs with names
```

**✅ You're ready to go!** Skip to the [Usage](#usage) section.

---

### 🔧 Option 2: Download Raw OpenTargets Data (Advanced)

**Best for:** Custom data processing, understanding the full pipeline, working with different OpenTargets versions

This option requires downloading large datasets from OpenTargets and involves more setup time.

**Requirements:**
- ~50GB+ free disk space
- Stable internet connection for large downloads
- FTP client or command line tools

#### Access the Data

Visit the OpenTargets downloads page: https://platform.opentargets.org/downloads/

#### Method A: Using FileZilla (Recommended)
1. **Host**: `ftp.ebi.ac.uk`
2. **Remote site**: `/pub/databases/opentargets/platform/`
3. **Navigate** to the version folders: `21.06`, `23.06`, or `24.06`
4. **Go to**: `output/etl/parquet/` within each version
5. **Download** the required datasets from each version

#### Method B: Command Line Download
```bash
# Create directory structure
mkdir -p raw_data/{21.06,23.06,24.06}

# Download using wget (example for 21.06)
cd raw_data/21.06
wget -r -np -nH --cut-dirs=7 https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/21.06/output/etl/parquet/indication/
wget -r -np -nH --cut-dirs=7 https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/21.06/output/etl/parquet/molecule/
wget -r -np -nH --cut-dirs=7 https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/21.06/output/etl/parquet/disease/
wget -r -np -nH --cut-dirs=7 https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/21.06/output/etl/parquet/target/
wget -r -np -nH --cut-dirs=7 https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/21.06/output/etl/parquet/associationByOverallDirect/

# Repeat for versions 23.06 and 24.06 (only indication needed for these)
```

#### Required Data by Version

**Training Version (21.06):**
From `/pub/databases/opentargets/platform/21.06/output/etl/parquet/`:
- `indication/`
- `molecule/`
- `disease/` → rename to `diseases/`
- `target/` → rename to `targets/`
- `associationByOverallDirect/`

**Validation Version (23.06):**
From `/pub/databases/opentargets/platform/23.06/output/etl/parquet/`:
- `indication/`

**Test Version (24.06):**
From `/pub/databases/opentargets/platform/24.06/output/etl/parquet/`:
- `indication/`

#### Final Directory Structure:
```
raw_data/
├── 21.06/
│   ├── indication/           
│   ├── molecule/            
│   ├── diseases/            # renamed from disease
│   ├── targets/             # renamed from target
│   └── associationByOverallDirect/
├── 23.06/
│   └── indication/          
└── 24.06/
    └── indication/          
```

**Important Notes:**
- All files are in PARQUET format
- The actual FTP path includes `/output/etl/parquet/` before the dataset names
- Rename `disease` to `diseases` and `target` to `targets` after download
- Large datasets may require significant download time and storage space
- Check OpenTargets license terms before using the data

After downloading, you'll need to update your `config.json` to point to the raw data directory and run the full processing pipeline.

---

## Usage

### Complete Pipeline
```bash
python run_pipeline.py
```

### Individual Steps
```bash
# Step 1: Create graph from processed data
python scripts/1_create_graph.py

# Step 2: Train models
python scripts/2_train_models.py

# Step 3: Evaluate models
python scripts/3_test_evaluate.py

# Step 4: Explain predictions
python scripts/4_explain_predictions.py
```

## Configuration

Create a `config.json` file:

```json
{
  "training_version": 21.06,
  "validation_version": 23.06,
  "test_version": 24.06,
  "as_dataset": "associationByOverallDirect",
  "negative_sampling_approach": "random",
  "processed_path": "processed_data/"
}
```

## Models

- **GCN**: Graph Convolutional Network
- **GraphSAGE**: Sample and Aggregate
- **Graph Transformer**: Attention-based GNN

## Output Files

The pipeline generates various output files during execution:

- `*_graph.pt` - Graph objects
- `*_best_model.pt` - Trained models  
- `test_results_summary.csv` - Performance metrics
- `test_evaluation_report.txt` - Detailed results

## License

MIT License
