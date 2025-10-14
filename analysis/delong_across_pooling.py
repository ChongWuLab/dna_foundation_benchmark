import os
import pandas as pd
import numpy as np
from scipy.stats import norm


# 1. DeLong's test helper methods
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
def read_preds(model_name, pooling_method, dataset_name, preds_base):
    """
    Reads predictions for a given model, pooling method, and dataset.
    Returns (true_array, pred_array).
    Raises FileNotFoundError if the file doesn't exist.
    """
    # Construct the filename based on the pooling method
    if pooling_method == "summary token":
        folder = model_name  # No suffix for summary token
    else:
        folder = f"{model_name}_{pooling_method}pool"  # meanpool or maxpool
    
    csv_path = os.path.join(preds_base, folder, f"{dataset_name}_rf.csv")
    
    df = pd.read_csv(csv_path)
    true_vals = df["True"].values
    pred_vals = df["Pred"].values
    return true_vals, pred_vals


# 3. Processing function
def process_pooling_significance(model_name, pooling_summary_csv, preds_folder, output_txt):
    """
    Analyzes which pooling method is significantly better for each dataset for a specific model.
    
    Args:
        model_name: Name of the model (e.g., "cadph", "ntv2")
        pooling_summary_csv: Path to CSV with AUC values across pooling methods
        preds_folder: Base folder containing prediction files
        output_txt: Output text file path
    """
    # The pooling methods in the CSV columns
    pooling_methods = ["summary token", "mean", "max"]
    
    # Read the pooling summary CSV
    df_summary = pd.read_csv(pooling_summary_csv)
    
    winners_by_dataset = {}
    
    for _, row in df_summary.iterrows():
        dataset_name = str(row["dataset name"])
        
        # Collect (pooling_method, AUC) ignoring NaN
        pooling_aucs = []
        for method in pooling_methods:
            auc_val = row.get(method, float('nan'))
            if pd.isna(auc_val):
                continue
            pooling_aucs.append((method, auc_val))
        
        # Sort by descending AUC
        pooling_aucs.sort(key=lambda x: x[1], reverse=True)
        
        dataset_winner = None
        num_methods = len(pooling_aucs)
        
        # Only consider the top method as a potential winner
        if num_methods >= 3:  # Need at least 3 methods to find one that's better than 2 others
            top_method, top_auc = pooling_aucs[0]
            count_signif_better = 0
            
            for j in range(1, num_methods):
                other_method, other_auc = pooling_aucs[j]
                
                if top_auc > other_auc:
                    # Attempt to read both predictions
                    try:
                        true_vals, preds1 = read_preds(model_name, top_method, dataset_name, preds_folder)
                        _, preds2 = read_preds(model_name, other_method, dataset_name, preds_folder)
                    except FileNotFoundError:
                        # If file is missing for either pooling method, skip
                        continue
                    
                    # DeLong test
                    z, p = delong_test(true_vals, preds1, preds2, return_auc=False)
                    if p < 0.01:
                        count_signif_better += 1
            
            # If significantly better than >= 2 others, it's a winner
            if count_signif_better >= 2:
                dataset_winner = top_method
        
        winners_by_dataset[dataset_name] = dataset_winner
    
    # Write results to text file: one line per dataset
    all_dataset_names = df_summary["dataset name"].astype(str).unique()
    all_dataset_names = sorted(all_dataset_names)
    
    with open(output_txt, "w") as f:
        for ds in all_dataset_names:
            winner = winners_by_dataset.get(ds)
            if winner:
                line = f"{ds},{winner}\n"
            else:
                line = f"{ds},\n"
            f.write(line)
    
    print(f"[{model_name}] Done! Results written to {output_txt}")


# 4. Run for all models
def main():
    # Use fixed paths as provided
    project_dir = ".."
    preds_folder = f"{project_dir}/preds"
    results_folder = f"{project_dir}/results_final"
    
    models = ["cadph", "dnabert2", "grover", "hyena", "ntv2"]
    
    for model in models:
        summary_csv = f"{results_folder}/{model}_across_pooling.csv"
        output_txt = f"{results_folder}/{model}_pooling_winners.txt"
        
        # Check if the CSV exists; if not, skip
        if not os.path.exists(summary_csv):
            print(f"[Warning] CSV '{summary_csv}' not found. Skipping {model}.")
            continue
        
        process_pooling_significance(
            model_name=model,
            pooling_summary_csv=summary_csv,
            preds_folder=preds_folder,
            output_txt=output_txt
        )

if __name__ == "__main__":
    main()