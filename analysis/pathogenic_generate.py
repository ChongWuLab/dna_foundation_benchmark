import csv
import os
from pyfaidx import Fasta

def generate_sequences(input_csv, reference_fasta, output_csv, half_window):
    """
    Generates reference and alternative sequences.
    """
    print(f"Processing with window size {half_window*2} bp...")
    
    ref_genome = Fasta(reference_fasta, rebuild=False)

    with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)

        writer.writerow([
            'ref_seq', 'alt_seq', 'class', 'chromosome', 'ref', 'pos', 'alt', 'MAF', 'split', 'label'
        ])

        for row in reader:
            ref_allele = row['REF']
            alt_allele = row['ALT']

            if len(ref_allele) != 1 or len(alt_allele) != 1:
                continue

            chrom = row['CHROM']
            if chrom not in ref_genome or chrom == "chrY":
                continue

            pos = int(row['POS'])
            
            genome_ref_base = ref_genome[chrom][pos-1].seq.upper()
            
            # Check if the REF allele in file matches the actual reference genome
            if ref_allele.upper() != genome_ref_base:
                print(f"Mismatch at {chrom}:{pos}. CSV REF is '{ref_allele}', but Genome is '{genome_ref_base}'. Skipping.")
                continue

            chr_length = len(ref_genome[chrom])

            start = pos - half_window
            end = pos + half_window - 1

            if start < 1 or end > chr_length:
                continue

            ref_seq = ref_genome[chrom][start:end].seq

            variant_index = half_window -1
            
            alt_seq_list = list(ref_seq)
            alt_seq_list[variant_index] = alt_allele
            alt_seq = "".join(alt_seq_list)

            writer.writerow([
                ref_seq,
                alt_seq,
                row['INT_LABEL'],
                chrom,
                ref_allele,
                pos,
                alt_allele,
                row.get('MAF', ''),
                row.get('split', ''),
                row.get('LABEL', '')
            ])


project_dir = ".."
input_csv = f'{project_dir}/data_processed/pathogenic/vep_pathogenic_coding.csv'
reference_fasta = f"{project_dir}/data_processed/TAD/hg38.ml.fa"
output_dir = f'{project_dir}/data_processed/pathogenic'

os.makedirs(output_dir, exist_ok=True)

window_configs = {
    "short": 1024,
    "medium": 3000,
    "long": 98304
}

# Loop through each configuration and generate the corresponding sequence file
for size_label, half_window_size in window_configs.items():
    output_csv_path = f'{output_dir}/seqs_pathogenic_{size_label}.csv'
    
    print(f"Generating file: {output_csv_path}")

    generate_sequences(
        input_csv=input_csv,
        reference_fasta=reference_fasta,
        output_csv=output_csv_path,
        half_window=half_window_size
    )
