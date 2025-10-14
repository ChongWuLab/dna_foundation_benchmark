import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_style("whitegrid")

# List of datasets to skip (these are multi-class problems for which AUC may not be rigorous)
skip_datasets = [
    'regulatory_region_type',
    'splice_site_type_DNABERT',
    'splice_site_type_NT',
    'covid_variants',
    'enhancer_strength'
]

def process_file(file_path, foundation_name):
    df = pd.read_csv(file_path)
    
    # Check if a dataset column exists; if not, use the index as dataset names.
    if 'dataset' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'dataset'})
    
    # Filter out datasets that should be skipped.
    df = df[~df['dataset'].isin(skip_datasets)]
    
    # Reshape dataframe into long format: columns 'dataset', 'Pooling', and 'AUC'
    df_long = df.melt(id_vars=['dataset'], var_name='Pooling', value_name='AUC')
    
    df_long['AUC'] = pd.to_numeric(df_long['AUC'], errors='coerce')
    df_long = df_long.dropna(subset=['AUC'])
    
    df_long['Foundation'] = foundation_name
    return df_long

# Process each CSV file and label with its foundation model name.
df_cadph    = process_file('../results_final/cadph_across_pooling.csv',    'CADPH')
df_dnabert2 = process_file('../results_final/dnabert2_across_pooling.csv', 'DNABERT2')
df_grover   = process_file('../results_final/grover_across_pooling.csv',   'GROVER')
df_hyena    = process_file('../results_final/hyena_across_pooling.csv',    'HYENA')
df_ntv2     = process_file('../results_final/ntv2_across_pooling.csv',     'NTv2')

# Combine the processed data from all models.
combined_df = pd.concat([df_cadph, df_dnabert2, df_grover, df_hyena, df_ntv2], ignore_index=True)

foundation_order = ['CADPH', 'DNABERT2', 'GROVER', 'HYENA', 'NTv2']

foundation_mapping = {
    'CADPH': 'Caduceus-Ph',
    'DNABERT2': 'DNABERT-2',
    'HYENA': 'HyenaDNA',
    'NTv2': 'NT-v2'
}
custom_labels = [foundation_mapping.get(f, f) for f in foundation_order]

# Determine the pooling method order.
pooling_order = combined_df[combined_df['Foundation'] == 'CADPH']['Pooling'].unique()

data_to_plot = []
positions = []

width = 0.2  
offsets = np.linspace(-width, width, num=len(pooling_order))

# Loop over each foundation model and pooling method.
for i, foundation in enumerate(foundation_order):
    for j, pooling in enumerate(pooling_order):
        auc_values = combined_df[
            (combined_df['Foundation'] == foundation) & 
            (combined_df['Pooling'] == pooling)
        ]['AUC'].dropna().values
        data_to_plot.append(auc_values)
        positions.append(i + offsets[j])


fig, ax = plt.subplots(figsize=(10, 6))

boxprops     = dict(linewidth=1.5, color='dimgray')
whiskerprops = dict(linewidth=1.5, color='dimgray')
capprops     = dict(linewidth=1.5, color='dimgray')
medianprops  = dict(linewidth=2.5, color='firebrick')
flierprops   = dict(marker='o', markerfacecolor='dimgray', markersize=5, linestyle='none')

bp = ax.boxplot(data_to_plot, positions=positions, widths=0.15, patch_artist=True,
                boxprops=boxprops, whiskerprops=whiskerprops,
                capprops=capprops, medianprops=medianprops, flierprops=flierprops)

ax.set_xticks(range(len(foundation_order)))
ax.set_xticklabels(custom_labels, fontsize=12)
ax.set_xlabel('Foundation Model', fontsize=14)
ax.set_ylabel('AUC', fontsize=14)
ax.set_title('AUC Distributions by Foundation Model and Pooling Method', fontsize=16, fontweight='bold')

# Use a custom color palette for pooling methods.
colors = ['#4c72b0', '#dd8452', '#55a868'][:len(pooling_order)]
for idx, box in enumerate(bp['boxes']):
    pooling_idx = idx % len(pooling_order)
    box.set_facecolor(colors[pooling_idx])

# Create a legend for the pooling methods.
legend_handles = [mpatches.Patch(color=color, label=pooling)
                  for color, pooling in zip(colors, pooling_order)]
ax.legend(handles=legend_handles, title='Pooling Method', fontsize=12, title_fontsize=12)

ax.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.savefig("./boxplot_pooling.png", dpi=300)

