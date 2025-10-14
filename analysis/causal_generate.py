import csv
import argparse
from pyfaidx import Fasta

def parse_vcf_line(line):
    """Parse a VCF line and return relevant fields"""
    if line.startswith('#'):
        return None
    
    fields = line.strip().split('\t')
    if len(fields) < 5:
        return None
    
    chrom = fields[0]
    pos = int(fields[1])
    col3_allele = fields[3]
    col4_allele = fields[4]
    
    return {
        'CHROM': chrom,
        'POS': pos,
        'COL3': col3_allele,
        'COL4': col4_allele
    }

def generate_sequences(input_vcf, reference_fasta, output_csv, window_size, qtl_type):
    """Generate reference and alternative sequences for given window size"""
    
    ref_genome = Fasta(reference_fasta)
    
    with open(input_vcf, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow([
            'ref_seq', 'alt_seq', 'chromosome', 'ref', 'pos', 'alt'
        ])
        
        for line in infile:
            variant = parse_vcf_line(line)
            if variant is None:
                continue
                
            chrom = variant['CHROM']
            if chrom == "chrY":
                continue
                
            pos = variant['POS']
            col3_allele = variant['COL3']
            col4_allele = variant['COL4']
            
            if chrom not in ref_genome:
                continue
                
            # For eqtl, perform sanity check to determine correct REF/ALT
            if qtl_type == 'eqtl':
                # Get reference nucleotide from genome
                genome_ref = ref_genome[chrom][pos-1].seq.upper()
                
                # Determine which column matches the reference genome
                if col3_allele.upper() == genome_ref:
                    ref_allele = col3_allele
                    alt_allele = col4_allele
                elif col4_allele.upper() == genome_ref:
                    ref_allele = col4_allele
                    alt_allele = col3_allele
                else:
                    # Neither matches: skip this variant
                    print("Something's wrong with mapping variants to reference genome. Skipping this variant.")
                    continue
            else:
                # For ipaqtl, paqtl, sqtl: column 3 is REF, column 4 is ALT
                ref_allele = col3_allele
                alt_allele = col4_allele
            
            # Determine chromosome length
            chr_length = len(ref_genome[chrom])
            
            start = pos - window_size
            end = pos + window_size
            
            # Skip if we cannot extract a full window
            if start < 1 or end > chr_length:
                continue
            
            ref_seq = ref_genome[chrom][start-1:end].seq
            variant_index = (pos - start)
            alt_seq = ref_seq[:variant_index] + alt_allele + ref_seq[variant_index+1:]
            
            writer.writerow([
                ref_seq, alt_seq, chrom, ref_allele, pos, alt_allele
            ])


parser = argparse.ArgumentParser(description='Generate reference and alternative sequences from VCF files')
parser.add_argument('--qtl_type', choices=['eqtl', 'ipaqtl', 'paqtl', 'sqtl'], 
                   required=True, help='QTL type to determine input directory')

args = parser.parse_args()

# Define window sizes and their labels
window_configs = [
    (1024, 'short'),
    (3000, 'medium'),
    (98304, 'long')
]

# Define input and output file patterns
project_dir = ".."

for window_size, size_label in window_configs:
    for causal in ["positive", "negative"]:
        if causal == 'negative':
            input_file = f'{project_dir}/data_processed/causal/{args.qtl_type}/blood_neg.vcf'
            output_file = f'{project_dir}/data_processed/causal/{args.qtl_type}/neg_seqs_{size_label}.csv'
        else:
            input_file = f'{project_dir}/data_processed/causal/{args.qtl_type}/blood_pos.vcf'
            output_file = f'{project_dir}/data_processed/causal/{args.qtl_type}/pos_seqs_{size_label}.csv'
        
        print(f"Processing {input_file} with window size {window_size*2} bp -> {output_file}")
        
        generate_sequences(
            input_vcf=input_file,
            reference_fasta=f"{project_dir}/data_processed/TAD/hg38.ml.fa",
            output_csv=output_file,
            window_size=window_size,
            qtl_type=args.qtl_type
        )

print("Sequence generation completed!")