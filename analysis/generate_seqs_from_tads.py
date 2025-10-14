import pandas as pd
import subprocess

# Load selected TADs
project_dir = ".."
tads = pd.read_csv(f'{project_dir}/data_processed/TAD/boundaries_selected.tsv', sep='\t', header=0)

# Load chromosome sizes into a dictionary
chrom_sizes = {}
with open(f'{project_dir}/data_processed/TAD/chrom_sizes.txt') as f:
    for line in f:
        chrom, size = line.strip().split()
        chrom_sizes[chrom] = int(size)

output_lines = []

for _, row in tads.iterrows():
    chrom = row['chrom']
    start = int(row['start'])
    end = int(row['end'])
    
    center = (start + end) // 2  # integer midpoint
    seq_start = center - 3000
    seq_end = center + 3000
    
    # Construct BED line: chrom, start, end
    output_lines.append(f"{chrom}\t{seq_start}\t{seq_end}\n")

# Write intervals to BED file
with open(f'{project_dir}/data_processed/TAD/tad_6kb_intervals.bed', 'w') as f:
    f.writelines(output_lines)

# Finally, extract sequences from the reference genome
subprocess.run(["bedtools", "getfasta", "-fi", f"{project_dir}/data_processed/TAD/hg38.ml.fa", 
                "-bed", f"{project_dir}/data_processed/TAD/tad_6kb_intervals.bed", 
                "-fo", f"{project_dir}/data_processed/TAD/tad_6kb_sequences.fa"])