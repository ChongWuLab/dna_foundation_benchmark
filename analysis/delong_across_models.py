import os
import pandas as pd
import numpy as np
from scipy.stats import norm


# DeLong's test helper methods
def get_XY(true, preds):
    """
    Splits 'preds' into two arrays:
      X = predictions for the positive class
      Y = predictions for the negative class
    """
    X = preds[true.astype(bool)]
    Y = preds[~true.astype(bool)]
    return X, Y

def mann_whitney(X, Y):
    """
    Computes the AUC by a rank-based method:
      AUC = 1/(len(X)*len(Y)) * sum(for each y in Y: # of X > y)
    with tie-handling (0.5).
    """
    return 1/(len(X)*len(Y)) * sum([np.sum(np.where(X == y, 0.5, y < X)) for y in Y])

def S_01_calculate(V01_A, V01_B, theta_A, theta_B, n):
    return (1 / (n - 1)) * np.sum((V01_A - theta_A) * (V01_B - theta_B))

def S_10_calculate(V10_A, V10_B, theta_A, theta_B, m):
    return (1 / (m - 1)) * np.sum((V10_A - theta_A) * (V10_B - theta_B))

def delong_test(true, preds_A, preds_B, return_auc=False):
    """
    Performs the two-sided DeLong test comparing AUC(preds_A) vs AUC(preds_B).
    
    Returns:
      (z, p_value) or (z, p_value, aucA, aucB) if return_auc=True.
    """
    X_A, Y_A = get_XY(true, preds_A)
    X_B, Y_B = get_XY(true, preds_B)
    m = len(X_A)
    n = len(Y_A)

    V01_A = np.array([np.mean(np.where(X_A == y_a, 0.5, y_a < X_A)) for y_a in Y_A])
    V10_A = np.array([np.mean(np.where(Y_A == x_a, 0.5, x_a > Y_A)) for x_a in X_A])
    V01_B = np.array([np.mean(np.where(X_B == y_b, 0.5, y_b < X_B)) for y_b in Y_B])
    V10_B = np.array([np.mean(np.where(Y_B == x_b, 0.5, x_b > Y_B)) for x_b in X_B])

    theta_A = mann_whitney(X_A, Y_A)
    theta_B = mann_whitney(X_B, Y_B)

    var_A = (S_10_calculate(V10_A, V10_A, theta_A, theta_B, m) / m
             + S_01_calculate(V01_A, V01_A, theta_A, theta_B, n) / n)
    var_B = (S_10_calculate(V10_B, V10_B, theta_A, theta_B, m) / m
             + S_01_calculate(V01_B, V01_B, theta_A, theta_B, n) / n)
    cov_AB = (S_10_calculate(V10_A, V10_B, theta_A, theta_B, m) / m
              + S_01_calculate(V01_A, V01_B, theta_A, theta_B, n) / n)

    z = (theta_A - theta_B) / np.sqrt(var_A + var_B - 2 * cov_AB)
    p_value = 2*(1 - norm.cdf(abs(z)))

    if return_auc:
        return z, p_value, theta_A, theta_B
    else:
        return z, p_value



# 2. Reading predictions
def read_preds(model_folder: str, dataset_name: str, preds_base: str):
    """
    Reads the CSV:  preds_base/model_folder/dataset_name_rf.csv
    Returns (true_array, pred_array).
    Raises FileNotFoundError if the file doesn't exist.
    """
    csv_path = os.path.join(preds_base, model_folder, f"{dataset_name}_rf.csv")

    df = pd.read_csv(csv_path)
    true_vals = df["True"].values
    pred_vals = df["Pred"].values
    return true_vals, pred_vals


# 3. Processing function
def process_significance(auc_summary_csv, preds_folder, pooling, winners_txt):
    """
    - Reads 'auc_summary_csv'
    - For each dataset, sorts models by descending AUC
    - Does pairwise DeLong test only when AUC1 > AUC2
    - If a model is significantly better (p<0.05) than at least TWO others,
      we consider it a "winner" for that dataset.
    - Writes one line per dataset in 'winners_txt'.

    Args:
      auc_summary_csv: Path to e.g. 'auc_summary.csv'
      preds_folder:    Base folder containing subfolders with predictions
      pooling:         One of ["", "meanpool", "maxpool"]
      winners_txt:     Output file name
    """

    # The DNA model names in the CSV columns
    dna_models = ["cadph", "dnabert2", "grover", "hyena", "ntv2"]

    # We'll map each base model to the actual folder name for preds
    # e.g., if pooling == "meanpool", "cadph" -> "cadph_meanpool"
    if pooling:
        model_folders = {m: f"{m}_{pooling}" for m in dna_models}
    else:
        model_folders = {m: m for m in dna_models}

    # Read the AUC summary
    df_summary = pd.read_csv(auc_summary_csv)

    winners_by_dataset = {}

    for _, row in df_summary.iterrows():
        dataset_name = str(row["dataset name"])

        # Collect (model, AUC) ignoring NaN
        model_aucs = []
        for m in dna_models:
            auc_val = row.get(m, float('nan'))
            if pd.isna(auc_val):
                continue
            model_aucs.append((m, auc_val))

        # Sort by descending AUC
        model_aucs.sort(key=lambda x: x[1], reverse=True)

        dataset_winners = []
        num_models = len(model_aucs)

        for i in range(num_models):
            m1, auc1 = model_aucs[i]
            count_signif_better = 0

            for j in range(i+1, num_models):
                m2, auc2 = model_aucs[j]

                if auc1 > auc2:
                    # Attempt to read both predictions
                    try:
                        true_vals, preds1 = read_preds(model_folders[m1], dataset_name, preds_folder)
                        _, preds2 = read_preds(model_folders[m2], dataset_name, preds_folder)
                    except FileNotFoundError:
                        # If file is missing for either model, skip
                        continue

                    # DeLong test
                    z, p = delong_test(true_vals, preds1, preds2, return_auc=False)
                    if p < 0.01:
                        count_signif_better += 1

            # If significantly better than >= 2 others, it's a winner
            if count_signif_better >= 2:
                dataset_winners.append(m1)

        winners_by_dataset[dataset_name] = dataset_winners

    # Write results to text file: one line per dataset
    # Even if no winners, we still write "dataset_name,"
    all_dataset_names = df_summary["dataset name"].astype(str).unique()
    all_dataset_names = sorted(all_dataset_names)

    with open(winners_txt, "w") as f:
        for ds in all_dataset_names:
            model_list = winners_by_dataset.get(ds, [])
            if model_list:
                line = f"{ds},{','.join(model_list)}\n"
            else:
                line = f"{ds},\n"
            f.write(line)

    print(f"[{pooling or 'no-pooling'}] Done! Results written to {winners_txt}")


# 4. Main code: run all pooling types
def main():
    # Adjust these base paths as needed
    project_dir = ".."
    preds_folder = f"{project_dir}/preds"

    # We will run the same logic for no pooling, meanpool, and maxpool
    # (If you only need certain ones, just remove any you don't want.)
    pool_configs = [
        {
            "pooling": "meanpool",
            "summary_csv": f"{project_dir}/results_final/auc_summary_meanpool.csv",
            "output_txt": f"{project_dir}/results_final/significant_winners_meanpool.txt"
        },
        {
            "pooling": "maxpool",
            "summary_csv": f"{project_dir}/results_final/auc_summary_maxpool.csv",
            "output_txt": f"{project_dir}/results_final/significant_winners_maxpool.txt"
        },
        {
            "pooling": "",
            "summary_csv": f"{project_dir}/results_final/auc_summary.csv",
            "output_txt": f"{project_dir}/results_final/significant_winners.txt"
        }
    ]

    for cfg in pool_configs:
        pooling = cfg["pooling"]
        summary_csv = cfg["summary_csv"]
        winners_txt = cfg["output_txt"]

        # Check if the CSV exists; if not, skip
        if not os.path.exists(summary_csv):
            print(f"[Warning] CSV '{summary_csv}' not found. Skipping {pooling or 'no-pooling'} scenario.")
            continue

        process_significance(
            auc_summary_csv=summary_csv,
            preds_folder=preds_folder,
            pooling=pooling,
            winners_txt=winners_txt
        )

if __name__ == "__main__":
    main()
