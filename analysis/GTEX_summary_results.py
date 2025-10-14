import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import seaborn as sns
import itertools


def log_p(p):
    """Helper function to return -log10(p), handling zero or underflow p-values."""
    return -np.log10(max(p, np.finfo(float).tiny))

# Define paths and model names
base_path = "/rsrch4/home/biostatistics/hfeng3/review_datasets/results_final/gtex_results/blood_cov_out"
output_dir = f"{base_path}/analysis_results"
os.makedirs(output_dir, exist_ok=True)

# Define model groups
short_sequence_models = ['dnabert2', 'ntv2', 'hyena', 'cadph', 'grover']
long_sequence_models = ['cadph_long', 'hyena_long', 'enformer']
model_names = short_sequence_models + long_sequence_models

# Functions for data processing and analysis
def load_model_data():
    """Load and clean all model data."""
    model_data = {}
    
    for model in model_names:
        file_path = os.path.join(base_path, f"{model}.csv")
        try:
            df = pd.read_csv(file_path)
            if len(df.columns) > 5:
                df = df.iloc[:, :5]
                
            # Handle non-finite values
            df = df.replace([np.inf, -np.inf], np.nan)
            df_before = len(df)
            df = df.dropna(subset=['rf_corr'])
            
            if len(df) > 0:
                model_data[model] = df
                print(f"Loaded {model} with {len(df)} rows (removed {df_before - len(df)} rows with non-finite values)")
            else:
                print(f"Warning: {model} has no valid data after removing non-finite values")
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return model_data

def compare_rf_vs_xgb(model_data):
    """Compare RF vs XGB performance for each model."""
    results = []
    
    for model, df in model_data.items():
        valid_mask = (~df['rf_corr'].isna()) & (~df['xgb_corr'].isna()) & (~df['rf_mse'].isna()) & (~df['xgb_mse'].isna())
        valid_df = df[valid_mask]
        
        if len(valid_df) > 0:
            # Correlation comparison
            corr_wilcoxon = wilcoxon(valid_df['rf_corr'], valid_df['xgb_corr'])
            corr_better = "RF" if valid_df['rf_corr'].mean() > valid_df['xgb_corr'].mean() else "XGB"
            
            # MSE comparison
            mse_wilcoxon = wilcoxon(valid_df['rf_mse'], valid_df['xgb_mse'])
            mse_better = "RF" if valid_df['rf_mse'].mean() < valid_df['xgb_mse'].mean() else "XGB"
            
            results.append({
                'model': model,
                'rf_corr_mean': valid_df['rf_corr'].mean(),
                'xgb_corr_mean': valid_df['xgb_corr'].mean(),
                'corr_p_value': corr_wilcoxon.pvalue,
                'corr_log_p': log_p(corr_wilcoxon.pvalue),
                'corr_better': corr_better,
                'corr_significant': corr_wilcoxon.pvalue < 0.01,
                'rf_mse_mean': valid_df['rf_mse'].mean(),
                'xgb_mse_mean': valid_df['xgb_mse'].mean(),
                'mse_p_value': mse_wilcoxon.pvalue,
                'mse_log_p': log_p(mse_wilcoxon.pvalue),
                'mse_better': mse_better,
                'mse_significant': mse_wilcoxon.pvalue < 0.01,
                'n_valid_pairs': len(valid_df)
            })
            
    return results

def compare_model_pair(model1, model2, model_data, metric='rf_corr'):
    """Compare two models using paired Wilcoxon test."""
    if model1 not in model_data or model2 not in model_data:
        return None
        
    df1 = model_data[model1]
    df2 = model_data[model2]
    
    # Create dictionaries for easier lookup
    model1_dict = df1.set_index('gene_id')[metric].to_dict()
    model2_dict = df2.set_index('gene_id')[metric].to_dict()
    
    # Find common genes and extract values
    common_genes = set(model1_dict.keys()).intersection(set(model2_dict.keys()))
    model1_values = []
    model2_values = []
    
    for gene_id in common_genes:
        val1 = model1_dict.get(gene_id)
        val2 = model2_dict.get(gene_id)
        
        if val1 is not None and not np.isnan(val1) and val2 is not None and not np.isnan(val2):
            model1_values.append(val1)
            model2_values.append(val2)
    
    if not model1_values:
        return None
    
    wilcoxon_result = wilcoxon(model1_values, model2_values)
    
    mean1 = np.mean(model1_values)
    mean2 = np.mean(model2_values)
    
    if metric == 'rf_mse':
        better = model1 if mean1 < mean2 else model2
    else:
        better = model1 if mean1 > mean2 else model2
    
    return {
        'model_pair': f"{model1} vs {model2}",
        'metric': metric,
        'model1': model1,
        'model2': model2,
        'model1_mean': mean1,
        'model2_mean': mean2,
        'p_value': wilcoxon_result.pvalue,
        'log_p_value': log_p(wilcoxon_result.pvalue),
        'better': better,
        'significant': wilcoxon_result.pvalue < 0.01,
        'n_valid_pairs': len(model1_values)
    }

def compare_model_groups(model_group, model_data):
    """Compare all pairs of models within a group."""
    results = []
    
    for model1, model2 in itertools.combinations(model_group, 2):
        corr_comparison = compare_model_pair(model1, model2, model_data, 'rf_corr')
        mse_comparison = compare_model_pair(model1, model2, model_data, 'rf_mse')
        
        if corr_comparison:
            results.append(corr_comparison)
        if mse_comparison:
            results.append(mse_comparison)
    
    return results

def compare_long_vs_short(model_data):
    """Compare long sequence vs short sequence versions of the same model."""
    pairs = [('hyena', 'hyena_long'), ('cadph', 'cadph_long')]
    results = []
    
    for short_model, long_model in pairs:
        corr_comparison = compare_model_pair(short_model, long_model, model_data, 'rf_corr')
        mse_comparison = compare_model_pair(short_model, long_model, model_data, 'rf_mse')
        
        if corr_comparison:
            # Reformat for consistency with output format
            corr_comparison['short_mean'] = corr_comparison['model1_mean']
            corr_comparison['long_mean'] = corr_comparison['model2_mean']
            results.append(corr_comparison)
            
        if mse_comparison:
            # Reformat for consistency with output format
            mse_comparison['short_mean'] = mse_comparison['model1_mean']
            mse_comparison['long_mean'] = mse_comparison['model2_mean']
            results.append(mse_comparison)
    
    return results

def create_histogram(model, df, metric='rf_corr', output_dir=None):
    """Create a publication-quality histogram with black-framed bars,
    inspired by scientific journals like Nature."""

    # Set a professional and clean plot style, optimized for clarity.
    sns.set_style("whitegrid", {
        "axes.edgecolor": ".15",  # Darker edge color for axes
        "axes.linewidth": 1,
        "grid.color": ".8",
        "grid.linestyle": ":"  # Dotted grid for less visual clutter
    })
    plt.figure(figsize=(10, 8), dpi=300)

    model_showcase_dict = {
        "hyena": "HyenaDNA",
        "hyena_long": "HyenaDNA-450K",
        "cadph": "Caduceus-Ph",
        "grover": "GROVER",
        "cadph_long": "Caduceus-Ph Long Sequence Input",
        "enformer": "Enformer",
        "dnabert2": "DNABERT-2",
        "ntv2": "NT-v2"
    }

    valid_data = df[metric].dropna()


    ax = sns.histplot(
        valid_data,
        kde=False,
        bins=30,
        color='#75aadb',
        alpha=0.9,
        edgecolor='black',
        linewidth=1.5
    )

    # Calculate statistics
    mean = np.mean(valid_data)
    variance = np.var(valid_data)
    n_samples = len(valid_data)

    plt.axvline(mean, color='#d55e00', linestyle='--', linewidth=2.5, label=f'Mean: {mean:.4f}')

    metric_name = "Correlation" if metric == 'rf_corr' else "Mean Squared Error"
    plt.title(f'Distribution of {metric_name} for {model_showcase_dict.get(model, model)}',
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel(metric_name, fontsize=14, labelpad=15)
    plt.ylabel('Frequency', fontsize=14, labelpad=15)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    stats_text = f'n = {n_samples}\nMean = {mean:.4f}\nVariance = {variance:.4f}'
    plt.text(0.04, 0.96, stats_text, transform=ax.transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='none'))

    legend = plt.legend(loc='upper right', fontsize=12, frameon=True, facecolor='white', framealpha=0.8, edgecolor='none')

    sns.despine(trim=True)

    plt.tight_layout()

    if output_dir:
        file_path = os.path.join(output_dir, f"{model}_{metric}_histogram.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()



def extract_top_genes(model_data, output_dir):
    """Extract top 10 genes by correlation and MSE for each model."""
    with open(os.path.join(output_dir, "top_genes_by_metrics.txt"), "w") as f:
        f.write("Top 10 Genes by RF Correlation and MSE for Each Model\n")
        f.write("==================================================\n\n")
        
        for model, df in model_data.items():
            f.write(f"Model: {model}\n")
            f.write("-" * len(f"Model: {model}") + "\n\n")
            
            # Top genes by correlation (higher is better)
            f.write("Top 10 Genes by RF Correlation (Higher is Better):\n")
            f.write("Rank | Gene ID | RF Correlation\n")
            f.write("-" * 40 + "\n")
            
            top10_corr = df.sort_values('rf_corr', ascending=False).head(10)
            for i, (_, row) in enumerate(top10_corr.iterrows(), 1):
                f.write(f"{i:4d} | {row['gene_id']:15s} | {row['rf_corr']:.5e}\n")
            
            f.write("\n")
            
            # Top genes by MSE (lower is better)
            f.write("Top 10 Genes by RF MSE (Lower is Better):\n")
            f.write("Rank | Gene ID | RF MSE\n")
            f.write("-" * 40 + "\n")
            
            top10_mse = df.sort_values('rf_mse', ascending=True).head(10)
            for i, (_, row) in enumerate(top10_mse.iterrows(), 1):
                f.write(f"{i:4d} | {row['gene_id']:15s} | {row['rf_mse']:.5e}\n")
            
            f.write("\n\n")

def write_comparison_results(results, filename, title):
    """Write comparison results to a text file."""
    with open(filename, "w") as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")
        
        # Group by model_pair
        model_pairs = {}
        for result in results:
            if result['model_pair'] not in model_pairs:
                model_pairs[result['model_pair']] = []
            model_pairs[result['model_pair']].append(result)
        
        for model_pair, pair_results in model_pairs.items():
            f.write(f"Model Pair: {model_pair}\n")
            f.write("-" * len(f"Model Pair: {model_pair}") + "\n\n")
            
            for result in pair_results:
                if result['metric'] == 'rf_corr':
                    f.write(f"RF Correlation Comparison:\n")
                    f.write(f"  Number of Valid Gene Pairs: {result['n_valid_pairs']}\n")
                    if 'short_mean' in result:  # For long vs short comparisons
                        f.write(f"  Short Model Mean RF Correlation: {result['short_mean']:.4f}\n")
                        f.write(f"  Long Model Mean RF Correlation: {result['long_mean']:.4f}\n")
                    else:
                        models = model_pair.split(" vs ")
                        f.write(f"  {result['model1']} Mean RF Correlation: {result['model1_mean']:.4f}\n")
                        f.write(f"  {result['model2']} Mean RF Correlation: {result['model2_mean']:.4f}\n")
                    f.write(f"  P-value (Wilcoxon Signed-Rank Test): {result['p_value']:.6e}\n")
                    f.write(f"  -log10(P-value): {result['log_p_value']:.4f}\n")
                    f.write(f"  Better Model: {result['better']}")
                    f.write(f" (Statistically Significant)" if result['significant'] else " (Not Significant)")
                    f.write("\n\n")
                elif result['metric'] == 'rf_mse':
                    f.write(f"RF MSE Comparison:\n")
                    f.write(f"  Number of Valid Gene Pairs: {result['n_valid_pairs']}\n")
                    if 'short_mean' in result:  # For long vs short comparisons
                        f.write(f"  Short Model Mean RF MSE: {result['short_mean']:.4f}\n")
                        f.write(f"  Long Model Mean RF MSE: {result['long_mean']:.4f}\n")
                    else:
                        models = model_pair.split(" vs ")
                        f.write(f"  {result['model1']} Mean RF MSE: {result['model1_mean']:.4f}\n")
                        f.write(f"  {result['model2']} Mean RF MSE: {result['model2_mean']:.4f}\n")
                    f.write(f"  P-value (Wilcoxon Signed-Rank Test): {result['p_value']:.6e}\n")
                    f.write(f"  -log10(P-value): {result['log_p_value']:.4f}\n")
                    f.write(f"  Better Model: {result['better']}")
                    f.write(f" (Statistically Significant)" if result['significant'] else " (Not Significant)")
                    f.write("\n\n")
            
            f.write("-" * 50 + "\n\n")



# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# Load and clean data
model_data = load_model_data()

# 1. Compare RF vs XGB performance
rf_vs_xgb_results = compare_rf_vs_xgb(model_data)

# Write results and create summary CSV
with open(os.path.join(output_dir, "rf_vs_xgb_comparison.txt"), "w") as f:
    f.write("Statistical Comparison of RF vs XGB Models (Paired Tests)\n")
    f.write("===================================================\n\n")
    
    for result in rf_vs_xgb_results:
        f.write(f"Model: {result['model']}\n")
        f.write(f"Number of Valid Gene Pairs: {result['n_valid_pairs']}\n")
        f.write(f"Correlation Comparison:\n")
        f.write(f"  RF Mean Correlation: {result['rf_corr_mean']:.4f}\n")
        f.write(f"  XGB Mean Correlation: {result['xgb_corr_mean']:.4f}\n")
        f.write(f"  P-value (Wilcoxon Signed-Rank Test): {result['corr_p_value']:.6e}\n")
        f.write(f"  -log10(P-value): {result['corr_log_p']:.4f}\n")
        f.write(f"  Better Model: {result['corr_better']}")
        f.write(f" (Statistically Significant)" if result['corr_significant'] else " (Not Significant)")
        f.write("\n\n")
        
        f.write(f"MSE Comparison:\n")
        f.write(f"  RF Mean MSE: {result['rf_mse_mean']:.4f}\n")
        f.write(f"  XGB Mean MSE: {result['xgb_mse_mean']:.4f}\n")
        f.write(f"  P-value (Wilcoxon Signed-Rank Test): {result['mse_p_value']:.6e}\n")
        f.write(f"  -log10(P-value): {result['mse_log_p']:.4f}\n")
        f.write(f"  Better Model: {result['mse_better']}")
        f.write(f" (Statistically Significant)" if result['mse_significant'] else " (Not Significant)")
        f.write("\n\n")
        f.write("-" * 50 + "\n\n")

# Create summary CSV
model_summary = pd.DataFrame([{
    'model': result['model'],
    'rf_corr_mean': result['rf_corr_mean'],
    'xgb_corr_mean': result['xgb_corr_mean'],
    'rf_mse_mean': result['rf_mse_mean'],
    'xgb_mse_mean': result['xgb_mse_mean'],
    'corr_better': result['corr_better'],
    'corr_significant': result['corr_significant'],
    'mse_better': result['mse_better'],
    'mse_significant': result['mse_significant'],
    'n_valid_pairs': result['n_valid_pairs']
} for result in rf_vs_xgb_results])
model_summary.to_csv(os.path.join(output_dir, "model_performance_summary.csv"), index=False)

# 2. Compare short sequence models
short_model_comparisons = compare_model_groups(short_sequence_models, model_data)
write_comparison_results(
    short_model_comparisons, 
    os.path.join(output_dir, "short_sequence_model_comparison.txt"),
    "Comparison of Short Sequence Models - Paired Tests"
)

# 3. Compare long sequence models
long_model_comparisons = compare_model_groups(long_sequence_models, model_data)
write_comparison_results(
    long_model_comparisons, 
    os.path.join(output_dir, "long_sequence_model_comparison.txt"),
    "Comparison of Long Sequence Models - Paired Tests"
)

# 4. Compare long vs short versions of same models
long_vs_short_results = compare_long_vs_short(model_data)
write_comparison_results(
    long_vs_short_results, 
    os.path.join(output_dir, "long_vs_short_model_comparison.txt"),
    "Comparison of Long vs Short Models - Paired Tests"
)

# 5. Create histograms for all models
for model, df in model_data.items():
    create_histogram(model, df, 'rf_corr', output_dir)
    create_histogram(model, df, 'rf_mse', output_dir)

# 6. Extract top genes for each model
extract_top_genes(model_data, output_dir)

print(f"Analysis complete. Results saved to {output_dir} directory.")
