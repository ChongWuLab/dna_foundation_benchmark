import os
import pandas as pd

def create_pretrain_vs_1k_summary(base_path: str):
    """
    Creates a comparison summary between HyenaDNA re-pretrained and HyenaDNA 1k models.
    
    Args:
        base_path (str): Path to the parent folder containing model results.
    """
    # Define the folders and model names
    pretrain_folder = os.path.join(base_path, "hyena_self_pretrain_meanpool")
    baseline_folder = os.path.join(base_path, "hyena_1k_meanpool")
    
    # Column names for the output CSV
    model_names = ["HyenaDNA Re-Pretrained", "HyenaDNA"]
    
    # Datasets that should use Accuracy instead of AUC
    accuracy_datasets = [
        "enhancer_strength", 
        "splice_site_type_NT", 
        "regulatory_region_type", 
        "splice_site_type_DNABERT", 
        "covid_variants"
    ]
    
    # Dictionary to hold metric values
    all_data = {}
    
    # Check if folders exist
    if not os.path.isdir(pretrain_folder):
        print(f"[Error] Folder '{pretrain_folder}' does not exist.")
        return
    
    if not os.path.isdir(baseline_folder):
        print(f"[Error] Folder '{baseline_folder}' does not exist.")
        return
    
    # Process both folders
    for folder_idx, folder_path in enumerate([pretrain_folder, baseline_folder]):
        model_name = model_names[folder_idx]
        
        for file_name in os.listdir(folder_path):
            if file_name.endswith("_rf.csv"):
                dataset_name = file_name.replace("_rf.csv", "")
                
                # Initialize the dataset entry if it doesn't exist
                if dataset_name not in all_data:
                    all_data[dataset_name] = {}
                
                csv_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(csv_path)
                
                # Determine which metric to use
                metric = "Accuracy" if dataset_name in accuracy_datasets else "AUC"
                
                # Find the appropriate metric row
                metric_row = df.loc[df["Metric"] == metric, "Value"]
                if not metric_row.empty:
                    metric_value = metric_row.values[0]
                    all_data[dataset_name][model_name] = metric_value
                else:
                    print(f"[Warning] No {metric} row in {csv_path}")
    
    # Build the final DataFrame
    columns = ["dataset name"] + model_names
    output_rows = []
    
    for dataset_name, model_dict in all_data.items():
        row = {"dataset name": dataset_name}
        for model_name in model_names:
            row[model_name] = model_dict.get(model_name, "")
        output_rows.append(row)
    
    df_out = pd.DataFrame(output_rows, columns=columns)
    df_out.sort_values(by="dataset name", inplace=True)
    
    output_file = os.path.join(base_path, "pretrain_vs_1k.csv")
    df_out.to_csv(output_file, index=False, float_format="%.4f")
    print(f"Created {output_file}")

def main():
    results_path = "../results_final"
    
    create_pretrain_vs_1k_summary(results_path)

if __name__ == "__main__":
    main()