import os
import pandas as pd

def combine_across_pooling():
    # The five base models you need
    base_models = ["cadph", "dnabert2", "grover", "hyena", "ntv2"]

    base_path = "../results_final"

    for model in base_models:
        # We'll create a dictionary that maps *pooling label* -> *folder name*.
        folder_map = {
            "summary token": f"{model}",            # no pooling
            "mean":          f"{model}_meanpool",   # meanpool
            "max":           f"{model}_maxpool",    # maxpool
        }

        all_data = {}

        # Loop over each pooling variant for this model
        for pooling_label, folder_name in folder_map.items():
            folder_path = os.path.join(base_path, folder_name)
            if not os.path.isdir(folder_path):
                print(f"[Warning] Folder '{folder_path}' does not exist. Skipping.")
                continue

            # Read all CSV files in this folder
            for file_name in os.listdir(folder_path):
                # We only care about files ending in "_rf.csv"
                if file_name.endswith("_rf.csv"):
                    dataset_name = file_name.replace("_rf.csv", "")
                    csv_path = os.path.join(folder_path, file_name)

                    df = pd.read_csv(csv_path)

                    # Find the row where Metric == "AUC"
                    auc_row = df.loc[df["Metric"] == "AUC", "Value"]
                    if auc_row.empty:
                        print(f"[Warning] No AUC row found in {csv_path}")
                        continue

                    auc_value = auc_row.values[0]

                    # Store AUC in the dictionary
                    if dataset_name not in all_data:
                        all_data[dataset_name] = {}
                    all_data[dataset_name][pooling_label] = auc_value

        columns = ["dataset name", "summary token", "mean", "max"]
        output_rows = []

        for dataset_name, pooling_dict in all_data.items():
            row = {
                "dataset name": dataset_name,
                "summary token": pooling_dict.get("summary token", ""),
                "mean":          pooling_dict.get("mean", ""),
                "max":           pooling_dict.get("max", ""),
            }
            output_rows.append(row)

        df_out = pd.DataFrame(output_rows, columns=columns)
        df_out.sort_values(by="dataset name", inplace=True)

        # Create the CSV name for this model, e.g. "cadph_across_pooling.csv"
        out_csv_name = os.path.join(base_path, f"{model}_across_pooling.csv")

        df_out.to_csv(out_csv_name, index=False, float_format="%.4f")

        print(f"Created {out_csv_name}")

if __name__ == "__main__":
    combine_across_pooling()
