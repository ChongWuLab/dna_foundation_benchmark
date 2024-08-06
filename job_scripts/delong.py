import pandas as pd
import numpy as np
from scipy.stats import norm
import os

def get_XY(true, preds):
    X = preds[true.astype(bool)]
    Y = preds[~true.astype(bool)]
    return X, Y


def mann_whitney(X, Y):
    return 1/(len(X)*len(Y)) * sum([np.sum(np.where(X == y, 0.5, y < X)) for y in Y])


def S_01_calculate(V01_A, V01_B, theta_A, theta_B, n):
    return (1 / (n - 1)) * np.sum((V01_A - theta_A) * (V01_B - theta_B))


def S_10_calculate(V10_A, V10_B, theta_A, theta_B, m):
    return (1 / (m - 1)) * np.sum((V10_A - theta_A) * (V10_B - theta_B))


def delong_test(true, preds_A, preds_B, return_auc=False):
    """
    Performs DeLong Test, takes true labels and predicted probabilities of two models
    """
    X_A, Y_A = get_XY(true, preds_A)
    X_B, Y_B = get_XY(true, preds_B)
    m = len(X_A)
    n = len(Y_A)

    # Structural components
    # For every preds in Y, calculate the average phi results with all X preds
    V01_A = np.array([np.mean(np.where(X_A == y_a, 0.5, y_a < X_A)) for y_a in Y_A])
    # For every preds in X, calculate the average phi results with all Y preds
    V10_A = np.array([np.mean(np.where(Y_A == x_a, 0.5, x_a > Y_A)) for x_a in X_A])
    # Same as above for model B
    V01_B = np.array([np.mean(np.where(X_B == y_b, 0.5, y_b < X_B)) for y_b in Y_B])
    V10_B = np.array([np.mean(np.where(Y_B == x_b, 0.5, x_b > Y_B)) for x_b in X_B])

    theta_A = mann_whitney(X_A, Y_A)
    theta_B = mann_whitney(X_B, Y_B)

    # To get variances, we only need three entries in the final S matrix:
    # The (0,0) entry, corresponding to var(theta_A). This is S_10(0,0)/m + S_01(0,0)/n
    var_A = S_10_calculate(V10_A, V10_A, theta_A, theta_B, m)/m + S_01_calculate(V01_A, V01_A, theta_A, theta_B, n)/n
    # The (1,1) entry, corresponding to var(theta_B). This is S_10(1,1)/m + S_01(1,1)/n
    var_B = S_10_calculate(V10_B, V10_B, theta_A, theta_B, m)/m + S_01_calculate(V01_B, V01_B, theta_A, theta_B, n)/n
    # The (0,1) and (1,0) entry, corresponding to cov(theta_A, theta_B). This is S_10(0,1)/m + S_01(0,1)/n
    cov_AB = S_10_calculate(V10_A, V10_B, theta_A, theta_B, m)/m + S_01_calculate(V01_A, V01_B, theta_A, theta_B, n)/n

    z = (theta_A - theta_B)/((var_A + var_B - 2 * cov_AB)**0.5)
    p_value = 2*(1 - norm.cdf(abs(z)))

    if return_auc == False:
        return z,p_value
    else:
        return z,p_value,theta_A,theta_B
    


### Perform DeLong Test ###

dnabert2_path = '../preds/dnabert2/'
ntv2_path = '../preds/ntv2/'
hyena_path = '../preds/hyena/'
output_path = "../preds/compare_result"
datasets = os.listdir(dnabert2_path)
datasets = [file[:-4] for file in datasets if file.endswith('.csv')]
pvalue_005 = []
pvalue_001 = []


for data in datasets:
    dnabert2 = pd.read_csv(os.path.join(dnabert2_path, data + '.csv'))
    ntv2 = pd.read_csv(os.path.join(ntv2_path, data + '.csv'))
    hyena = pd.read_csv(os.path.join(hyena_path, data + '.csv'))

    true = dnabert2["True"].values
    dnabert2_preds = dnabert2["Pred"].values
    ntv2_preds = ntv2["Pred"].values
    hyena_preds = hyena["Pred"].values

    d_vs_n = delong_test(true, dnabert2_preds, ntv2_preds)
    n_vs_h = delong_test(true, ntv2_preds, hyena_preds)
    d_vs_h = delong_test(true, dnabert2_preds, hyena_preds)

    pvalue_005.append([data, d_vs_n[1]<0.05, n_vs_h[1]<0.05, d_vs_h[1]<0.05])
    pvalue_001.append([data, d_vs_n[1]<0.01, n_vs_h[1]<0.01, d_vs_h[1]<0.01])

df = pd.DataFrame(pvalue_005, columns=['data', 'dnabert2_vs_ntv2', 'ntv2_vs_hyena', 'dnabert2_vs_hyena'])
df.to_csv(f'../preds/{output_path}_005.csv', index=False)

df = pd.DataFrame(pvalue_001, columns=['data', 'dnabert2_vs_ntv2', 'ntv2_vs_hyena', 'dnabert2_vs_hyena'])
df.to_csv(f'../preds/{output_path}_001.csv', index=False)