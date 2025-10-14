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
    
    # Check if a dataset column exists; if not, use the index as dataset names
    if 'dataset' not in df.columns and 'dataset name' in df.columns:
        df = df.rename(columns={'dataset name': 'dataset'})
    elif 'dataset' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'dataset'})
    
    # Filter out datasets that should be skipped
    df = df[~df['dataset'].isin(skip_datasets)]
    
    # Reshape dataframe into long format
    df_long = df.melt(id_vars=['dataset'], var_name='Classifier', value_name='AUC')
    df_long['AUC'] = pd.to_numeric(df_long['AUC'], errors='coerce')
    
    df_long = df_long.dropna(subset=['AUC'])

    df_long['Foundation'] = foundation_name
    return df_long

def create_boxplot(combined_df, pooling_type, ax):

    foundation_order = ['CADPH', 'DNABERT2', 'GROVER', 'HYENA', 'NTv2']
    
    foundation_mapping = {
        'CADPH': 'Caduceus-Ph',
        'DNABERT2': 'DNABERT-2',
        'HYENA': 'HyenaDNA',
        'NTv2': 'NT-v2'
    }
    custom_labels = [foundation_mapping.get(f, f) for f in foundation_order]
    
    classifier_order = ['rf', 'elastic', 'nb']
    classifier_labels = {'rf': 'Random Forest', 'elastic': 'Elastic Net', 'nb': 'Naive Bayes'}
    
    data_to_plot = []
    positions = []
    
    width = 0.2
    offsets = np.linspace(-width, width, num=len(classifier_order))
    
    # Loop over each foundation model and classifier method
    for i, foundation in enumerate(foundation_order):
        for j, classifier in enumerate(classifier_order):
            auc_values = combined_df[
                (combined_df['Foundation'] == foundation) & 
                (combined_df['Classifier'] == classifier)
            ]['AUC'].dropna().values
            data_to_plot.append(auc_values)
            positions.append(i + offsets[j])
    
    # Define custom properties for boxplot aesthetics
    boxprops = dict(linewidth=1.5, color='dimgray')
    whiskerprops = dict(linewidth=1.5, color='dimgray')
    capprops = dict(linewidth=1.5, color='dimgray')
    medianprops = dict(linewidth=2.5, color='firebrick')
    flierprops = dict(marker='o', markerfacecolor='dimgray', markersize=5, linestyle='none')
    
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.15, patch_artist=True,
                    boxprops=boxprops, whiskerprops=whiskerprops,
                    capprops=capprops, medianprops=medianprops, flierprops=flierprops)
    
    ax.set_xticks(range(len(foundation_order)))
    ax.set_xticklabels(custom_labels, fontsize=12)
    
    if ax.get_subplotspec().is_last_row():
        ax.set_xlabel('Foundation Model', fontsize=14)
    
    ax.set_ylabel('AUC', fontsize=14)
    ax.set_title(f'AUC Distributions by Foundation Models and Downstream Classifier, {pooling_type} Pooling', 
                 fontsize=14, fontweight='bold')
    
    # Use a custom color palette for classifier methods
    colors = ['#4c72b0', '#dd8452', '#55a868'][:len(classifier_order)]
    for idx, box in enumerate(bp['boxes']):
        classifier_idx = idx % len(classifier_order)
        box.set_facecolor(colors[classifier_idx])
    
    if ax.get_subplotspec().is_first_row():
        legend_handles = [mpatches.Patch(color=color, label=classifier_labels.get(classifier, classifier))
                          for color, classifier in zip(colors, classifier_order)]
        ax.legend(handles=legend_handles, title='Classifier Method', fontsize=12, title_fontsize=12)
    
    ax.tick_params(axis='both', labelsize=12)
    
    return ax


if __name__ == "__main__":
    pooling_types = {
        'summpool': 'Summary-Token',
        'meanpool': 'Mean',
        'maxpool': 'Max'
    }
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    
    # Process each pooling type
    for idx, (pooling_key, pooling_name) in enumerate(pooling_types.items()):
        df_cadph = process_file(f'../results_final/cadph_{pooling_key}_across_classifiers.csv', 'CADPH')
        df_dnabert2 = process_file(f'../results_final/dnabert2_{pooling_key}_across_classifiers.csv', 'DNABERT2')
        df_grover = process_file(f'../results_final/grover_{pooling_key}_across_classifiers.csv', 'GROVER')
        df_hyena = process_file(f'../results_final/hyena_{pooling_key}_across_classifiers.csv', 'HYENA')
        df_ntv2 = process_file(f'../results_final/ntv2_{pooling_key}_across_classifiers.csv', 'NTv2')
        
        # Combine the processed data from all models
        combined_df = pd.concat([df_cadph, df_dnabert2, df_grover, df_hyena, df_ntv2], ignore_index=True)
        
        create_boxplot(combined_df, pooling_name, axes[idx])
    
    plt.tight_layout()
    plt.savefig("./stacked_boxplots_classifiers.png", dpi=300)