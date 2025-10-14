import os
import glob
import pandas as pd
from pathlib import Path


input_base_path = "Path/to/stored/regression/results"
output_base_path = "../results_final/gtex_results/blood_cov_out"

model_names = ['dnabert2', 'hyena', 'ntv2', 'cadph', 'enformer', 'hyena_long', 'cadph_long']

os.makedirs(output_base_path, exist_ok=True)

for model_name in model_names:
    print(f"Processing {model_name}...")
    
    # Path to the model's directory containing CSV files
    model_dir = os.path.join(input_base_path, model_name)
    
    # Check if the directory exists
    if not os.path.exists(model_dir):
        print(f"Directory not found: {model_dir}")
        continue
    
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(model_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {model_dir}")
        continue
    
    print(f"Found {len(csv_files)} csv files.")
    # Read and combine all CSV files
    combined_df = pd.DataFrame()
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Append to the combined dataframe
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    # Make sure gene_id is unique
    if combined_df['gene_id'].duplicated().any():
        print(f"Warning: Duplicate gene_ids found for {model_name}")
    

    output_file = os.path.join(output_base_path, f"{model_name}.csv")
    combined_df.to_csv(output_file, index=False)
    print(f"Saved combined CSV for {model_name} with {len(combined_df)} rows to {output_file}")

