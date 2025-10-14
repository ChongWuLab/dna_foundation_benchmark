import os
import pandas as pd

def combine_across_models(pooling: str, base_path: str, base_models: list):
    """
    Combines results across multiple models (base_models) for a single pooling type.
    Produces a CSV:
      - auc_summary.csv            (if pooling == "")
      - auc_summary_{pooling}.csv  (otherwise)
    with columns:
      dataset name, cadph, dnabert2, grover, hyena, ntv2

    Args:
        pooling (str): "", "meanpool", or "maxpool".
        base_path (str): Path to folders containing model results.
        base_models (list): Names of the models (e.g. ["cadph", "dnabert2", ...]).
    """
    # Decide the subfolders and the output CSV filename
    if pooling:
        folders = [os.path.join(base_path, f"{m}_{pooling}") for m in base_models]
        out_filename = os.path.join(base_path, f"auc_summary_{pooling}.csv")
    else:
        folders = [os.path.join(base_path, m) for m in base_models]
        out_filename = os.path.join(base_path, "auc_summary.csv")

    all_data = {}

    # Traverse each folder, pairing it with its model name
    for model_name, folder_path in zip(base_models, folders):
        if not os.path.isdir(folder_path):
            print(f"[Warning] Folder '{folder_path}' does not exist. Skipping.")
            continue

        # Read all files ending with "_rf.csv"
        for file_name in os.listdir(folder_path):
            if file_name.endswith("_rf.csv"):
                dataset_name = file_name.replace("_rf.csv", "")
                csv_path = os.path.join(folder_path, file_name)

                df = pd.read_csv(csv_path)
                # Find the AUC row
                auc_row = df.loc[df["Metric"] == "AUC", "Value"]
                if auc_row.empty:
                    print(f"[Warning] No AUC row in {csv_path}")
                    continue

                auc_value = auc_row.values[0]

                # Store in dictionary
                if dataset_name not in all_data:
                    all_data[dataset_name] = {}
                all_data[dataset_name][model_name] = auc_value

    # Build the final DataFrame
    columns = ["dataset name"] + base_models
    output_rows = []
    for dataset_name, model_dict in all_data.items():
        row = {"dataset name": dataset_name}
        for model_name in base_models:
            row[model_name] = model_dict.get(model_name, "")
        output_rows.append(row)

    df_out = pd.DataFrame(output_rows, columns=columns)
    df_out.sort_values(by="dataset name", inplace=True)

    # Write out the CSV with four decimal places
    df_out.to_csv(out_filename, index=False, float_format="%.4f")
    print(f"Created {out_filename}")


def main():
    base_path = "../results_final"
    # The five model names (used both for folder naming and final CSV columns)
    base_models = ["dnabert2", "ntv2", "hyena", "cadph", "grover"]

    pooling_methods = ["", "meanpool", "maxpool"]
    for pooling in pooling_methods:
        combine_across_models(pooling=pooling, base_path=base_path, base_models=base_models)


if __name__ == "__main__":
    main()
