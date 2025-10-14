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
    Computes the AUC using a rank-based method with tie handling (0.5).
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
      (z, p_value) or (z, p_value, auc_A, auc_B) if return_auc=True.
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

    var_A = (S_10_calculate(V10_A, V10_A, theta_A, theta_B, m) / m +
             S_01_calculate(V01_A, V01_A, theta_A, theta_B, n) / n)
    var_B = (S_10_calculate(V10_B, V10_B, theta_A, theta_B, m) / m +
             S_01_calculate(V01_B, V01_B, theta_A, theta_B, n) / n)
    cov_AB = (S_10_calculate(V10_A, V10_B, theta_A, theta_B, m) / m +
              S_01_calculate(V01_A, V01_B, theta_A, theta_B, n) / n)

    z = (theta_A - theta_B) / np.sqrt(var_A + var_B - 2 * cov_AB)
    p_value = 2 * (1 - norm.cdf(abs(z)))

    if return_auc:
        return z, p_value, theta_A, theta_B
    else:
        return z, p_value


# 2. Reading predictions
def read_preds(model_folder, dataset_name, preds_folder):
    """
    Reads predictions from:
      preds_base/model_folder/dataset_name_rf.csv
    Expects the CSV to have columns "True" and "Pred".
    """
    csv_path = os.path.join(preds_folder, model_folder, f"{dataset_name}_rf.csv")
    df = pd.read_csv(csv_path)
    true_vals = df["True"].values
    pred_vals = df["Pred"].values
    return true_vals, pred_vals


# 3. Processing baseline comparisons
def process_baseline_comparisons(auc_summary_csv, preds_folder, output_baseline_better, output_win_over_baseline):
    """
    For each dataset (from the AUC summary CSV), compares each DNA model's AUC (from its meanpool predictions)
    to the baseline model's AUC (from the 'baseline' folder) using the DeLong test.
    
    Records:
      - DNA models that are significantly beaten by baseline (p < 0.01 and baseline AUC > DNA AUC)
      - DNA models that are significantly better than baseline (p < 0.01 and DNA AUC > baseline AUC)
    
    Writes results to:
      - 'baseline_better.txt': one line per dataset: dataset, DNA model(s) beaten by baseline
      - 'win_over_baseline.txt': one line per dataset: dataset, DNA model(s) that beat baseline
    """
    dna_models = ["cadph", "dnabert2", "grover", "hyena", "ntv2"]
    # Use meanpool predictions for DNA models
    model_folders = {m: f"{m}_meanpool" for m in dna_models}
    
    df_summary = pd.read_csv(auc_summary_csv)
    baseline_better = {}
    win_over_baseline = {}
    
    for _, row in df_summary.iterrows():
        dataset_name = str(row["dataset name"])
        baseline_beats = []
        dna_beats_baseline = []
        for m in dna_models:
            try:
                true_dna, preds_dna = read_preds(model_folders[m], dataset_name, preds_folder)
                true_baseline, preds_baseline = read_preds("baseline", dataset_name, preds_folder)
            except FileNotFoundError:
                continue

            # Compare baseline vs. DNA model
            z, p, auc_baseline, auc_dna = delong_test(true_baseline, preds_baseline, preds_dna, return_auc=True)
            if p < 0.01:
                if auc_baseline > auc_dna:
                    baseline_beats.append(m)
                elif auc_dna > auc_baseline:
                    dna_beats_baseline.append(m)
        baseline_better[dataset_name] = baseline_beats
        win_over_baseline[dataset_name] = dna_beats_baseline

    datasets = sorted(df_summary["dataset name"].astype(str).unique())
    
    with open(output_baseline_better, "w") as f:
        for ds in datasets:
            models = baseline_better.get(ds, [])
            f.write(f"{ds}," + ",".join(models) + "\n")
    
    with open(output_win_over_baseline, "w") as f:
        for ds in datasets:
            models = win_over_baseline.get(ds, [])
            f.write(f"{ds}," + ",".join(models) + "\n")
    
    print(f"Baseline comparisons complete! Results written to '{output_baseline_better}' and '{output_win_over_baseline}'.")


# 4. Main
def main():
    # Adjust project directory as needed
    project_dir = ".."
    preds_folder = os.path.join(project_dir, "preds")
    
    # Use the meanpool AUC summary CSV to obtain dataset names.
    auc_summary_csv = os.path.join(project_dir, "results_final", "auc_summary_meanpool.csv")
    output_baseline_better = os.path.join(project_dir, "results_final", "baseline_better.txt")
    output_win_over_baseline = os.path.join(project_dir, "results_final", "win_over_baseline.txt")
    
    if not os.path.exists(auc_summary_csv):
        print(f"[Warning] CSV '{auc_summary_csv}' not found. Exiting.")
        return
    
    process_baseline_comparisons(
        auc_summary_csv=auc_summary_csv,
        preds_folder=preds_folder,
        output_baseline_better=output_baseline_better,
        output_win_over_baseline=output_win_over_baseline
    )

if __name__ == "__main__":
    main()
