import random
import pandas as pd
import subprocess

# Load chromosome sizes
project_dir = ".."
chrom_sizes = pd.read_csv(f'{project_dir}/data_processed/TAD/chrom_sizes.txt', sep='\t', header=None, names=['chrom','size'])

# Exclude chrX
chrom_sizes = chrom_sizes[chrom_sizes['chrom'] != 'chrX'].reset_index(drop=True)

desired_count = 1500
interval_length = 6000
intervals = []
temp_bed = f'{project_dir}/data_processed/TAD/temp_interval.bed'
temp_fasta = f'{project_dir}/data_processed/TAD/temp_sequence.fa'

def check_sequence_valid(chrom, start, end):
    # Write single interval to temporary bed file
    with open(temp_bed, 'w') as f:
        f.write(f"{chrom}\t{start}\t{end}\n")
    
    # Extract sequence
    subprocess.run(["bedtools", "getfasta", "-fi", f"{project_dir}/data_processed/TAD/hg38.ml.fa",
                   "-bed", temp_bed, "-fo", temp_fasta], check=True)
    
    # Check if sequence contains N
    with open(temp_fasta, 'r') as f:
        next(f)  # Skip header line
        sequence = f.readline().strip().upper()
        return 'N' not in sequence

def pick_random_interval():
    # Pick a random base index from the entire genome length
    r = random.randint(0, total_size-1)
    # Determine which chromosome this index falls into
    chrom_index = chrom_sizes['cum_size'].searchsorted(r, side='right')
    chrom = chrom_sizes.iloc[chrom_index]['chrom']
    
    # The chromosome must be large enough for a 6kb window
    chrom_length = chrom_sizes.iloc[chrom_index]['size']
    start_limit = chrom_length - interval_length
    if start_limit < 1:
        return None
    
    # Random start coordinate
    start = random.randint(0, start_limit)
    end = start + interval_length
    
    # Check if the sequence contains N
    if check_sequence_valid(chrom, start, end):
        return chrom, start, end
    return None

# Create a cumulative distribution for weighted chromosome selection
chrom_sizes['cum_size'] = chrom_sizes['size'].cumsum()
total_size = chrom_sizes['cum_size'].iloc[-1]

# Generate intervals
attempts = 0
max_attempts = desired_count * 10

while len(intervals) < desired_count and attempts < max_attempts:
    picked = pick_random_interval()
    if picked is not None:
        intervals.append(picked)
    attempts += 1

if len(intervals) < desired_count:
    print(f"Warning: Only found {len(intervals)} valid intervals after {attempts} attempts")

# Write final intervals to a BED file
with open(f'{project_dir}/data_processed/TAD/background_6kb_intervals.bed', 'w') as f:
    for c, s, e in intervals:
        f.write(f"{c}\t{s}\t{e}\n")


subprocess.run(["rm", temp_bed, temp_fasta])

# Finally, extract sequences from the reference genome
subprocess.run(["bedtools", "getfasta", "-fi", f"{project_dir}/data_processed/TAD/hg38.ml.fa", 
                "-bed", f"{project_dir}/data_processed/TAD/background_6kb_intervals.bed", 
                "-fo", f"{project_dir}/data_processed/TAD/background_6kb_sequences.fa"])