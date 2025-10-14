import csv
from pyfaidx import Fasta

# Input files
project_dir = ".."
input_csv = f'{project_dir}/data_processed/pathogenic/vep_pathogenic_coding.csv'
reference_fasta = f"{project_dir}/data_processed/TAD/hg38.ml.fa"

output_csv = f'{project_dir}/data_processed/pathogenic/sequences_pathogenic.csv'

# Load the reference genome
ref_genome = Fasta(reference_fasta)

window = 3000

with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)

    # Write headers
    writer.writerow([
        'ref_seq', 'alt_seq', 'class', 'chromosome', 'ref', 'pos', 'alt', 'MAF', 'split', 'label'
    ])

    for row in reader:
        chrom = row['CHROM']
        if chrom == "chrY":  # Skip chrY rows
            continue
        pos = int(row['POS'])
        ref_allele = row['REF']
        alt_allele = row['ALT']
        int_label = row['INT_LABEL']
        maf = row.get('MAF', '')  # Handle empty MAF values
        split_value = row.get('split', '')
        label = row.get('LABEL', '')

        # Determine chromosome length
        chr_length = len(ref_genome[chrom])

        start = pos - window
        end = pos + window

        # Skip if we cannot extract a full ±5kb window
        if start < 1 or end > chr_length:
            continue

        # Extract reference sequence
        ref_seq = ref_genome[chrom][start-1:end].seq

        # Position of the variant within the extracted sequence
        variant_index = (pos - start)

        # Construct the alt sequence
        alt_seq = ref_seq[:variant_index] + alt_allele + ref_seq[variant_index+1:]

        # Write to the output file
        writer.writerow([
            ref_seq, alt_seq, int_label, chrom, ref_allele, pos, alt_allele, maf, split_value, label
        ])
