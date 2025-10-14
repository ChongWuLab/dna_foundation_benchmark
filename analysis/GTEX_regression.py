import os
import pandas as pd
import numpy as np
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import argparse

# Get the job index to determine which CSV file to process
try:
    job_index = int(os.environ.get('LSB_JOBINDEX', 0))
    if job_index == 0:
        job_index = 1
except:
    job_index = 1

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_name', type=str, default="dnabert2", help="Model name to specify target directory")
parser.add_argument('--seed', type=int, default=42, help="Random seed for train-test split")
args = parser.parse_args()

# Define paths
embedding_dir = f'Path/to/stored/organized/embeddings/files'
expression_path = '../data_processed/GTEX/Whole_Blood.v8.expression_regressed.tsv'
output_dir = f'Path/to/store/regression/results/for/blood/with/covariates/regressedout'

os.makedirs(output_dir, exist_ok=True)

# List all CSV files in the directory
csv_files = sorted(glob.glob(os.path.join(embedding_dir, '*.csv')))
if not csv_files:
    print(f"No CSV files found in {embedding_dir}")
    exit(0)

# Get the CSV file to process based on job index (1-based indexing)
if job_index > len(csv_files):
    print(f"Job index {job_index} exceeds the number of available CSV files ({len(csv_files)})")
    exit(0)

csv_file = csv_files[job_index - 1]
print(f"Processing file {job_index}/{len(csv_files)}: {os.path.basename(csv_file)}")

print(f"Loading embedding data from {csv_file}")
embedding_df = pd.read_csv(csv_file)
print(f"Loaded embedding data with shape {embedding_df.shape}")

print(f"Loading expression data from {expression_path}")
expression_df = pd.read_csv(expression_path, sep='\t')
print(f"Loaded expression data with shape {expression_df.shape}")

# Get embedding feature columns
embedding_cols = [col for col in embedding_df.columns if col.startswith('embedding_')]
print(f"Found {len(embedding_cols)} embedding features")

# Get subject columns from expression data (columns after 4th column)
subject_cols = expression_df.columns[4:]

# Create a consistent train-test split for all subjects
np.random.seed(args.seed)
train_subjects = np.random.choice(subject_cols, size=int(len(subject_cols)*0.75), replace=False)
test_subjects = np.array([s for s in subject_cols if s not in train_subjects])

print(f"Created consistent train-test split with seed {args.seed}")
print(f"Train set: {len(train_subjects)} subjects, Test set: {len(test_subjects)} subjects")


results = []

# Process each gene in the embedding file
unique_genes = embedding_df['gene_id'].unique()
print(f"Found {len(unique_genes)} unique genes in the embedding file")

for gene_id in unique_genes:
    print("----------------------------")
    print(f"Processing gene: {gene_id}")
    
    # Skip if gene not in expression data
    if gene_id not in expression_df['gene_id'].values:
        print(f"Gene {gene_id} not found in expression data, skipping")
        continue
    
    # Get data for this gene
    gene_embedding = embedding_df[embedding_df['gene_id'] == gene_id]
    gene_expr = expression_df[expression_df['gene_id'] == gene_id]
    
    # Prepare data for training
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    
    for subj in train_subjects:
        if subj in gene_embedding['subject_id'].values:
            # Get embedding for this subject
            row = gene_embedding[gene_embedding['subject_id'] == subj]
            X_train.append(row[embedding_cols].values[0])
            # Get expression value
            y_train.append(gene_expr[subj].values[0])
    
    for subj in test_subjects:
        if subj in gene_embedding['subject_id'].values:
            row = gene_embedding[gene_embedding['subject_id'] == subj]
            X_test.append(row[embedding_cols].values[0])
            y_test.append(gene_expr[subj].values[0])
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    if len(X_train) < 5 or len(X_test) < 5:
        print(f"Not enough data for gene {gene_id}, skipping")
        continue
        
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train and evaluate models
    # Random Forest
    print(f"Training Random Forest for gene {gene_id}")
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 5, 10],
    }
    rf_grid = GridSearchCV(
        estimator=rf_model,
        param_grid=rf_param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=0
    )
    rf_grid.fit(X_train, y_train)
    
    # XGBoost
    print(f"Training XGBoost for gene {gene_id}")
    xgb_model = XGBRegressor(random_state=42, n_jobs=-1)
    xgb_param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 5, 7],
    }
    xgb_grid = GridSearchCV(
        estimator=xgb_model,
        param_grid=xgb_param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=0
    )
    xgb_grid.fit(X_train, y_train)
    
    best_rf = rf_grid.best_estimator_
    best_xgb = xgb_grid.best_estimator_
    rf_pred = best_rf.predict(X_test)
    xgb_pred = best_xgb.predict(X_test)
    
    # Calculate metrics
    rf_mse = mean_squared_error(y_test, rf_pred)
    xgb_mse = mean_squared_error(y_test, xgb_pred)
    
    rf_corr, _ = pearsonr(y_test, rf_pred)
    xgb_corr, _ = pearsonr(y_test, xgb_pred)
    
    print(f"RF: MSE={rf_mse:.4f}, Corr={rf_corr:.4f}")
    print(f"XGB: MSE={xgb_mse:.4f}, Corr={xgb_corr:.4f}")
    
    results.append({
        'gene_id': gene_id,
        'rf_corr': rf_corr,
        'xgb_corr': xgb_corr,
        'rf_mse': rf_mse,
        'xgb_mse': xgb_mse,
        'rf_params': str(rf_grid.best_params_),
        'xgb_params': str(xgb_grid.best_params_)
    })

results_df = pd.DataFrame(results)

output_path = f"{output_dir}/results_file_{job_index}.csv"
results_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")
