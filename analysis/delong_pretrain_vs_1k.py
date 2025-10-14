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


# 2. Processing function
def process_pretrain_vs_1k_significance(preds_folder, summary_csv, output_txt):
    """
    Analyzes whether the re-pretrained model is significantly better than the 1k model.
    
    Args:
        preds_folder: Base folder containing prediction files
        summary_csv: Path to CSV with metrics comparing the two models
        output_txt: Output text file path
    """
    # The two model folders and their display names
    pretrain_folder = "hyena_self_pretrain_meanpool"
    baseline_folder = "hyena_1k_meanpool"
    model_display_names = {
        pretrain_folder: "HyenaDNA Re-Pretrained",
        baseline_folder: "HyenaDNA"
    }
    

    df_summary = pd.read_csv(summary_csv)
    
    results_by_dataset = {}
    
    for _, row in df_summary.iterrows():
        dataset_name = str(row["dataset name"])
        
        # Path to the prediction files
        pretrain_file = os.path.join(preds_folder, pretrain_folder, f"{dataset_name}_rf.csv")
        baseline_file = os.path.join(preds_folder, baseline_folder, f"{dataset_name}_rf.csv")
        
        # Skip if either file is missing
        if not os.path.exists(pretrain_file) or not os.path.exists(baseline_file):
            print(f"[Warning] Skipping {dataset_name}: one or both prediction files missing")
            results_by_dataset[dataset_name] = ""
            continue
        
        # Read the prediction files
        pretrain_df = pd.read_csv(pretrain_file)
        baseline_df = pd.read_csv(baseline_file)
        
        # Get the true and prediction values
        true_vals = pretrain_df["True"].values
        pretrain_preds = pretrain_df["Pred"].values
        baseline_preds = baseline_df["Pred"].values
        
        # Perform DeLong test
        z, p_value, pretrain_auc, baseline_auc = delong_test(
            true_vals, pretrain_preds, baseline_preds, return_auc=True
        )
        
        # Determine which model is better
        if p_value < 0.01:  # Significance threshold of 0.01
            if pretrain_auc > baseline_auc:
                winner = model_display_names[pretrain_folder]
            else:
                winner = model_display_names[baseline_folder]
        else:
            winner = ""
        
        results_by_dataset[dataset_name] = (winner)
    
    # Write results to text file
    with open(output_txt, "w") as f:
        f.write("dataset,winner,p_value,AUCs(pretrain_vs_1k)\n")
        
        for dataset in sorted(results_by_dataset.keys()):
            winner = results_by_dataset[dataset]
                
            f.write(f"{dataset},{winner}\n")
    
    print(f"Done! Results written to {output_txt}")


# 3. Main
def main():
    project_dir = ".."
    preds_folder = f"{project_dir}/preds"
    results_folder = f"{project_dir}/results_final"
    
    summary_csv = os.path.join(results_folder, "pretrain_vs_1k.csv")
    output_txt = os.path.join(results_folder, "pretrain_vs_1k_significance.txt")
    
    if not os.path.exists(summary_csv):
        print(f"[Error] Summary CSV '{summary_csv}' not found.")
        return
    
    # Perform the significance tests
    process_pretrain_vs_1k_significance(
        preds_folder=preds_folder,
        summary_csv=summary_csv,
        output_txt=output_txt
    )

if __name__ == "__main__":
    main()