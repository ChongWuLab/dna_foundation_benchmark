import os
import pandas as pd
import numpy as np
import argparse
import glob
from multiprocessing import Pool, cpu_count

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_name', type=str, help='Model name to use in the file paths')
args = parser.parse_args()
model_name = args.model_name


source_path = f"Path/to/generated/embeddings/by/scripts/in/job_scripts"
dest_path = f"Path/to/store/organized/embeddings/files"

os.makedirs(dest_path, exist_ok=True)

# 1: Read all CSV files from source directory
print("Scanning for CSV files...")
csv_files = glob.glob(os.path.join(source_path, "*.csv"))
print(f"Found {len(csv_files)} CSV files.")

# 2: Combine all files into a single DataFrame
print("Reading and combining CSV files...")
dfs = []
for file in csv_files:
    print(f"Reading {os.path.basename(file)}...")
    df = pd.read_csv(file)
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
print(f"Combined DataFrame shape: {combined_df.shape}")

# 3: Perform sanity checks
total_genes = combined_df['gene_id'].nunique()
print(f"Total unique genes: {total_genes}")

# Count subjects per gene
gene_subject_counts = combined_df.groupby('gene_id')['subject_id'].nunique()
print("Subject counts per gene:")
print(f"Min: {gene_subject_counts.min()}")
print(f"Max: {gene_subject_counts.max()}")
print(f"Mean: {gene_subject_counts.mean():.2f}")
print(f"Median: {gene_subject_counts.median()}")

# 4: Group by gene_ids and split into smaller files
unique_genes = np.array(combined_df['gene_id'].unique())

# Calculate number of complete files with genes_per_file genes each
genes_per_file = 5
num_complete_files = total_genes // genes_per_file
remainder_genes = total_genes % genes_per_file

print(f"Will create {num_complete_files} files with 5 genes each, plus 1 file with {remainder_genes} genes")

# Create list of gene chunks (genes_per_file genes per chunk except the last one)
gene_chunks = []
for i in range(num_complete_files):
    start_idx = i * genes_per_file
    gene_chunks.append(unique_genes[start_idx:start_idx + genes_per_file])

# Add the remainder chunk if there are any genes left
if remainder_genes > 0:
    gene_chunks.append(unique_genes[num_complete_files * genes_per_file:])

# Function to process a single chunk
def process_chunk(args):
    i, gene_chunk = args
    
    mask = np.isin(gene_id_array, gene_chunk)
    chunk_df = combined_df.loc[mask]
    
    output_file = os.path.join(dest_path, f"embed_genes_group_{model_name}_{i}.csv")
    print(f"Saving chunk {i+1} with {len(gene_chunk)} genes to {output_file}")
    chunk_df.to_csv(output_file, index=False)
    return i

gene_id_array = np.array(combined_df['gene_id'])
print("Processing chunks in parallel...")
num_cores = min(cpu_count(), len(gene_chunks))
print(f"Using {num_cores} CPU cores")

with Pool(num_cores) as pool:
    args = [(i, gene_chunks[i]) for i in range(len(gene_chunks))]
    results = pool.map(process_chunk, args)

print("Processing complete!")