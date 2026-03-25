#!/usr/bin/env bash
set -euo pipefail

# Overnight runner for Long COVID method/input sensitivity analyses.
# Runs:
#   1) Another-seed gene-config comparison: NARROW / BROAD / FULL
#   2) Hub-threshold sensitivity on NARROW: max-drug-per-gene = 3, 5, 10, none
#
# Default seed is 123 because it gives a second fair comparison against the
# existing seed=42 table without changing any model hyperparameters.
#
# Usage:
#   bash scripts/seal/run_long_covid_methods_sensitivity.sh
#   bash scripts/seal/run_long_covid_methods_sensitivity.sh 123

if [[ "${LONGCOVID_SKIP_CAFFEINATE:-0}" != "1" ]] && command -v caffeinate >/dev/null 2>&1; then
    export LONGCOVID_SKIP_CAFFEINATE=1
    exec caffeinate -dimsu bash "$0" "$@"
fi

SEED="${1:-123}"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="results/long_covid/sensitivity_seed${SEED}_${STAMP}"
LOG_DIR="$RUN_DIR/logs"
MANIFEST="$RUN_DIR/manifest.tsv"
SUMMARY_MD="$RUN_DIR/summary.md"

mkdir -p "$LOG_DIR"

BASE_CMD=(uv run python -u scripts/seal/predict_long_covid.py
    --conv sage
    --use-jk
    --hidden 32
    --epochs 50
    --hops 2
    --neg-ratio 3
    --neg-strategy mixed
    --hard-ratio 0.5
    --top-k 100
    --seed "$SEED")

echo -e "label\tseed\tgene_categories\tmax_drug_per_gene\tjson_path\tlog_path" > "$MANIFEST"

run_case() {
    local label="$1"
    local gene_categories="$2"
    local max_drug_per_gene="$3"

    local log_file="$LOG_DIR/${label}.log"
    local before_file
    before_file="$(ls -1t results/long_covid/seal_long_covid_*.json 2>/dev/null | head -1 || true)"

    echo
    echo "============================================================"
    echo "[$(date)] Running ${label}"
    echo "  seed=${SEED}"
    echo "  gene_categories=${gene_categories}"
    echo "  max_drug_per_gene=${max_drug_per_gene}"
    echo "============================================================"

    if [[ "$max_drug_per_gene" == "none" ]]; then
        "${BASE_CMD[@]}" --gene-categories "$gene_categories" 2>&1 | tee "$log_file"
    else
        "${BASE_CMD[@]}" --gene-categories "$gene_categories" --max-drug-per-gene "$max_drug_per_gene" 2>&1 | tee "$log_file"
    fi

    local after_file
    after_file="$(ls -1t results/long_covid/seal_long_covid_*.json | head -1)"
    if [[ -z "$after_file" || "$after_file" == "$before_file" ]]; then
        echo "No new result JSON detected for ${label}" >&2
        exit 1
    fi

    echo -e "${label}\t${SEED}\t${gene_categories}\t${max_drug_per_gene}\t${after_file}\t${log_file}" >> "$MANIFEST"
    echo "Saved ${label} -> ${after_file}"
}

echo "============================================================"
echo " Long COVID method sensitivity overnight run"
echo " Started: $(date)"
echo " Seed: ${SEED}"
echo " Output dir: ${RUN_DIR}"
echo "============================================================"
echo
echo "Planned unique runs: 6"
echo "  Gene configs: NARROW, BROAD, FULL"
echo "  Hub thresholds: 3, 5, 10, none"
echo "  Note: NARROW = hub threshold 5 baseline, but kept in both summaries"
echo

# Gene-config comparison
run_case "gene_narrow" "gwas" "5"
run_case "gene_broad" "gwas,own,cat3,causal,combinatorial" "5"
run_case "gene_full" "all" "5"

# Hub-threshold sensitivity on NARROW
run_case "hub_cap3" "gwas" "3"
echo -e "hub_cap5\t${SEED}\tgwas\t5\t$(awk -F '\t' '$1=="gene_narrow" {print $5}' "$MANIFEST")\t$(awk -F '\t' '$1=="gene_narrow" {print $6}' "$MANIFEST")" >> "$MANIFEST"
run_case "hub_cap10" "gwas" "10"
run_case "hub_none" "gwas" "none"

uv run python scripts/seal/summarise_lc_sensitivity.py \
    --manifest "$MANIFEST" \
    --output "$SUMMARY_MD"

echo
echo "============================================================"
echo " Overnight run complete: $(date)"
echo " Manifest: $MANIFEST"
echo " Summary:  $SUMMARY_MD"
echo "============================================================"
