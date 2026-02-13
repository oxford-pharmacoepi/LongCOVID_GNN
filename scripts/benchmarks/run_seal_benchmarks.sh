#!/bin/bash
# Multi-disease benchmarking for SEAL

diseases=("EFO_0003854" "EFO_0003929" "MONDO_0002009" "HP_0000726")
names=("Osteoporosis_Postmenopausal" "Multiple_Sclerosis_RR" "Major_Depressive_Disorder" "Dementia")

mkdir -p results/seal_benchmarks

for i in "${!diseases[@]}"; do
    disease=${diseases[$i]}
    name=${names[$i]}
    echo "=========================================================="
    echo "RUNNING SEAL BENCHMARK FOR: $name ($disease)"
    echo "=========================================================="
    
    uv run scripts/seal/train_loo.py --target-disease $disease --epochs 50 --hops 2 > "results/seal_benchmarks/${name}_fixed.log" 2>&1
    
    echo "DONE. Check results/seal_benchmarks/${name}_fixed.log"
done
