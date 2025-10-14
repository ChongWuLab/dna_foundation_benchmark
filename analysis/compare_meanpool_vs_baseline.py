import os
import pandas as pd
import argparse
import glob
from pathlib import Path

# Define model name mapping
MODEL_NAME_MAP = {
    "baseline": "baseline CNN",
    "dnabert2_meanpool": "DNABERT-2",
    "ntv2_meanpool": "NT-v2",
    "hyena_meanpool": "HyenaDNA",
    "grover_meanpool": "GROVER",
    "cadph_meanpool": "Caduceus-Ph"
}

# Define datasets that should use Accuracy instead of AUC
ACCURACY_DATASETS = [
    "enhancer_strength", 
    "splice_site_type_NT", 
    "regulatory_region_type", 
    "splice_site_type_DNABERT", 
    "covid_variants"
]

def extract_dataset_name(filename, model_folder):
    """Extract dataset name based on the model folder"""
    stem = Path(filename).stem
    
    # For baseline folder, use the filename as is
    if model_folder == "baseline":
        return stem
    # For other folders, remove the _rf suffix
    else:
        if stem.endswith("_rf"):
            return stem[:-3]  # Remove _rf
        return stem

def format_value(value):
    """Format numeric values to 3 decimal places"""
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return value

def main():
    parent_folder = "../results_final"
    
    # Initialize a dictionary to store results
    results = {}
    all_datasets = set()
    
    # Process each model folder
    for model_folder in sorted(os.listdir(parent_folder)):
        model_path = os.path.join(parent_folder, model_folder)
        
        if not os.path.isdir(model_path) or model_folder not in MODEL_NAME_MAP:
            continue
            
        model_name = MODEL_NAME_MAP[model_folder]
        results[model_name] = {}
        
        # Find all CSV files for this model
        csv_files = glob.glob(os.path.join(model_path, "*.csv"))
        
        for csv_file in csv_files:
            dataset_name = extract_dataset_name(csv_file, model_folder)
            
            # Skip files that don't match the expected pattern for non-baseline folders
            if model_folder != "baseline" and not Path(csv_file).stem.endswith("_rf"):
                continue
                
            all_datasets.add(dataset_name)
            
            df = pd.read_csv(csv_file)
            
            # Determine which metric to use
            metric = "Accuracy" if dataset_name in ACCURACY_DATASETS else "AUC"
            
            # Extract the value for the chosen metric
            try:
                value = df.loc[df["Metric"] == metric, "Value"].values[0]
                results[model_name][dataset_name] = format_value(value)
            except (IndexError, KeyError):
                results[model_name][dataset_name] = "NA"
    
    # Convert to DataFrame for easy output
    result_df = pd.DataFrame.from_dict(results)
    
    # Sort datasets
    all_datasets = sorted(list(all_datasets))
    result_df = result_df.reindex(all_datasets)
    
    print(result_df.to_string())
    
    output_file = os.path.join(parent_folder, "compare_with_baseline.csv")
    result_df.to_csv(output_file)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()