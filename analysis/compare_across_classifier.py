import os
import pandas as pd

def combine_across_classifiers():
    base_models = ["cadph", "dnabert2", "grover", "hyena", "ntv2"]
    
    # Pooling methods
    pooling_methods = {
        "summary pooling": "_summpool",           # No suffix for summary-token pooling
        "meanpool": "_meanpool",    # Suffix for mean pooling
        "maxpool": "_maxpool"       # Suffix for max pooling
    }
    
    # The classifier types we want to compare
    classifier_types = {
        "rf": "_rf.csv",          # Random Forest
        "nb": "_nb.csv",          # Naive Bayes
        "elastic": "_elastic.csv" # Elastic Net
    }
    
    # Base path to results
    base_path = "../results_final"
    
    # Loop through each model
    for model in base_models:
        # Loop through each pooling method
        for pooling_name, pooling_suffix in pooling_methods.items():
            # Create the full folder name for this model+pooling combination
            folder_name = f"{model}{pooling_suffix}"
            folder_path = os.path.join(base_path, folder_name)
            
            if not os.path.isdir(folder_path):
                print(f"[Warning] Folder '{folder_path}' does not exist. Skipping.")
                continue
                
            # Dictionary to store all AUC values
            all_data = {}
            
            # Loop through files in the directory
            for file_name in os.listdir(folder_path):
                # Find which classifier this file corresponds to
                clf_match = None
                for clf_name, clf_suffix in classifier_types.items():
                    if file_name.endswith(clf_suffix):
                        clf_match = clf_name
                        # Get dataset name by removing the classifier suffix
                        dataset_name = file_name.replace(clf_suffix, "")
                        break
                        
                if clf_match is None:
                    continue
                    
                csv_path = os.path.join(folder_path, file_name)
                
                df = pd.read_csv(csv_path)
                
                # Find the row where Metric == "AUC"
                auc_row = df.loc[df["Metric"] == "AUC", "Value"]
                if auc_row.empty:
                    print(f"[Warning] No AUC row found in {csv_path}")
                    continue
                    
                auc_value = auc_row.values[0]
                
                if dataset_name not in all_data:
                    all_data[dataset_name] = {}
                all_data[dataset_name][clf_match] = auc_value
                

            if all_data:
                columns = ["dataset name", "rf", "nb", "elastic"]
                output_rows = []
                
                # Create rows for each dataset
                for dataset_name, clf_dict in all_data.items():
                    row = {
                        "dataset name": dataset_name,
                        "rf": clf_dict.get("rf", ""),
                        "nb": clf_dict.get("nb", ""),
                        "elastic": clf_dict.get("elastic", "")
                    }
                    output_rows.append(row)
                    
                # Create DataFrame and sort by dataset name
                df_out = pd.DataFrame(output_rows, columns=columns)
                df_out.sort_values(by="dataset name", inplace=True)
                
                out_csv_name = os.path.join(base_path, f"{folder_name}_across_classifiers.csv")

                df_out.to_csv(out_csv_name, index=False, float_format="%.4f")
                
                print(f"Created {out_csv_name}")

if __name__ == "__main__":
    combine_across_classifiers()