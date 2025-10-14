import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

project_dir = ".."

# Model name mapping
file_model_mapping = {
    'grover.csv': 'GROVER',
    'hyena.csv': 'HyenaDNA-160K',
    'cadph.csv': 'Caduceus-Ph-131K',
    'ntv2.csv': 'NT-v2',
    'dnabert2.csv': 'DNABERT-2',
    'hyena_long.csv': 'HyenaDNA-1M'
}

custom_colors = {
    'GROVER': '#3498db',
    'HyenaDNA-160K': '#e74c3c',
    'Caduceus-Ph-131K': '#2ecc71',
    'NT-v2': '#9b59b6',
    'DNABERT-2': '#f39c12',
    'HyenaDNA-1M': '#1abc9c'
}

# Get all CSV files in the directory
csv_files = [f"{project_dir}/runtimes/{filename}" for filename in file_model_mapping]

plt.figure(figsize=(12, 8))

max_lengths = {}

# Process each CSV file (representing a model)
for csv_file in csv_files:
    filename = os.path.basename(csv_file)
    model_name = file_model_mapping[filename]
    
    df = pd.read_csv(csv_file)
    
    # Store max sequence length
    max_lengths[model_name] = df['sequence_length'].max()
    
    # Plot sequence_length vs total_runtime_seconds - without dots
    plt.plot(df['sequence_length'][1:], df['total_runtime_seconds'][1:], 
             linestyle='-', linewidth=3.0, 
             color=custom_colors.get(model_name, '#333333'),
             label=f"{model_name}")

# Collect all unique sequence lengths to use as x-ticks
all_seq_lengths = set()
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    all_seq_lengths.update(df['sequence_length'].tolist())

# Sort sequence lengths and set as x-ticks
all_seq_lengths = sorted(list(all_seq_lengths))

plt.title("Runtime Comparison on Different DNA Sequence Lengths", fontsize=18, fontweight='bold')
plt.xlabel("Number of Nucleotides", fontsize=16)
plt.ylabel("Total Runtime for 1000 Replications (seconds)", fontsize=16)
plt.xscale('log', base=2)  # Keep log scale

tick_values = all_seq_lengths[1:]
plt.xticks(tick_values, tick_values, rotation=45)

for model, max_length in max_lengths.items():
    plt.axvline(x=max_length, color=custom_colors[model], linestyle='--', alpha=0.5, 
                linewidth=1.5)

plt.grid(True, which="both", ls="-", alpha=0.2)
plt.gca().set_facecolor('#f8f9fa')

plt.legend(title="Models", fontsize=12, 
           frameon=True, facecolor='white', edgecolor='#e6e6e6',
           loc='lower right')

# Add minor gridlines for better readability of the log scale
plt.grid(True, which='minor', linestyle=':', alpha=0.2)
plt.tight_layout()
plt.savefig(f'{project_dir}/runtimes/runtime_comparison.png', dpi=300, bbox_inches='tight')