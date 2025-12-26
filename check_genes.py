import json


# Load gene mapping
with open('processed_data/mappings/gene_key_mapping.json', 'r') as f:
    gene_mapping = json.load(f)

# Load GWAS genes
gwas_genes = []
with open('gwas_genes_long_covid.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            gene_id = line.split('#')[0].strip()
            if gene_id:
                gwas_genes.append(gene_id)

print(f'Total GWAS genes: {len(gwas_genes)}')
print(f'Total genes in mapping: {len(gene_mapping)}')
print()

# Check which genes are found
found = []
missing = []

for gene in gwas_genes:
    if gene in gene_mapping:
        found.append((gene, gene_mapping[gene]))
    else:
        missing.append(gene)

print(f'Found in mapping: {len(found)}/{len(gwas_genes)} genes ({len(found)/len(gwas_genes)*100:.1f}%)')
print()

if found:
    print('Found genes (showing first 10 with their indices):')
    for gene, idx in found[:10]:
        print(f'  {gene}: {idx}')
    if len(found) > 10:
        print(f'  ... and {len(found)-10} more')

print()

if missing:
    print(f'Missing from mapping: {len(missing)} genes')
    for gene in missing:
        print(f'  {gene}')
