import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import roc_auc_score

def get_hyperparameter_grid(model_name: str) -> dict:
    LOW_DIM_MODELS = ['hyena', 'cadph', 'hyena_long', 'cadph_long', 'dnabert2', 'grover', 'ntv2']
    HIGH_DIM_MODELS = ['enformer_hidden', 'enformer_output', 'alpha', 'sei_hidden', 'sei_output']
    
    if model_name in HIGH_DIM_MODELS:
        return {
            'n_estimators': [100, 200, 400],
            'max_features': ['log2', 32],
            'max_depth': [10, None],
            'min_samples_leaf': [5, 10]
        }
    else:
        return {
            'n_estimators': [100, 200, 400],
            'max_features': ['sqrt', 'log2', 32],
            'max_depth': [10, None],
            'min_samples_leaf': [5, 10]
        }

def load_and_prepare_data(model_name: str, folder_name: str) -> tuple:
    print(f"\nLoading data for model '{model_name}' from folder '{folder_name}'...")
    pos_file = f"/rsrch4/home/biostatistics/hfeng3/review_datasets/data_processed/causal/{folder_name}/{model_name}_differences_pos.csv"
    neg_file = f"/rsrch4/home/biostatistics/hfeng3/review_datasets/data_processed/causal/{folder_name}/{model_name}_differences_neg.csv"
    df_pos = pd.read_csv(pos_file)
    df_neg = pd.read_csv(neg_file)
    df_pos['label'] = 1
    df_neg['label'] = 0
    df_combined = pd.concat([df_pos, df_neg], ignore_index=True)
    X = df_combined.iloc[:, 4:-1].values
    y = df_combined['label'].values
    chromosomes = df_combined['chromosome']
    print(f"Data loaded successfully. Found {X.shape[0]} total samples.")
    return X, y, chromosomes

def calculate_cohens_d(scores: np.ndarray, labels: np.ndarray) -> float:
    scores_pos, scores_neg = scores[labels == 1], scores[labels == 0]
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    if n_pos < 2 or n_neg < 2: return np.nan
    mean_pos, mean_neg = np.mean(scores_pos), np.mean(scores_neg)
    std_pos, std_neg = np.std(scores_pos, ddof=1), np.std(scores_neg, ddof=1)
    pooled_std = np.sqrt(((n_pos - 1) * std_pos**2 + (n_neg - 1) * std_neg**2) / (n_pos + n_neg - 2))
    return (mean_pos - mean_neg) / pooled_std if pooled_std > 1e-6 else 0.0



parser = argparse.ArgumentParser(description="Benchmark RF classifiers using rotating chromosome test sets with 4-fold CV tuning.")
parser.add_argument("--model", type=str, required=True, help="Name of the model.")
parser.add_argument("--folder", type=str, required=True, choices=['eqtl', 'ipaqtl', 'sqtl', 'paqtl'], help="The data type folder.")
args = parser.parse_args()

TEST_CHROMS_GROUP1 = [f'chr{c}' for c in [3, 6, 9, 12, 16, 18, 19, 21]]
TEST_CHROMS_GROUP2 = [f'chr{c}' for c in [2, 5, 11, 14, 17, 20, 22]] + ['chrX']
TEST_CHROMS_GROUP3 = [f'chr{c}' for c in [1, 4, 7, 8, 10, 13, 15]]

test_scenarios = [
    {
        'name': 'Group1_as_test',
        'test_chroms': TEST_CHROMS_GROUP1,
        'description': 'Test on Group1 chromosomes'
    },
    {
        'name': 'Group2_as_test', 
        'test_chroms': TEST_CHROMS_GROUP2,
        'description': 'Test on Group2 chromosomes'
    },
    {
        'name': 'Group3_as_test',
        'test_chroms': TEST_CHROMS_GROUP3,
        'description': 'Test on Group3 chromosomes'
    }
]

ALL_CHROMS_LIST = [f'chr{c}' for c in range(1, 23)] + ['chrX']

X_all, y_all, chromosomes_all = load_and_prepare_data(args.model, args.folder)

all_results = []

for scenario_idx, scenario in enumerate(test_scenarios):
    print(f"\nSCENARIO {scenario_idx + 1}: {scenario['description']}")
    print(f"Test chromosomes: {scenario['test_chroms']}")
    
    current_test_chroms = scenario['test_chroms']
    train_val_chroms = [c for c in ALL_CHROMS_LIST if c not in current_test_chroms]
    
    print(f"Train/Validation chromosomes: {train_val_chroms}")
    
    is_test = np.isin(chromosomes_all, current_test_chroms)
    X_train_val, y_train_val = X_all[~is_test], y_all[~is_test]
    chromosomes_train_val = chromosomes_all[~is_test]
    X_test, y_test = X_all[is_test], y_all[is_test]

    print(f"Train/Validation samples: {len(X_train_val)}")
    print(f"Test samples: {len(X_test)}")
    
    n_splits = 4
    
    group_kfold_tuner = GroupKFold(n_splits=n_splits)
    param_grid = get_hyperparameter_grid(args.model)
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    print(f"Starting {n_splits}-fold cross-validation for hyperparameter tuning...")
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=group_kfold_tuner,
        verbose=1
    )
    
    try:
        grid_search.fit(X_train_val, y_train_val, groups=chromosomes_train_val)
        
        best_params = grid_search.best_params_
        cv_score = grid_search.best_score_
        print(f"Hyperparameter tuning complete.")
        print(f"Best parameters: {best_params}")
        print(f"Mean cross-validated AUROC: {cv_score:.4f}")
        
    except Exception as e:
        print(f"ERROR in hyperparameter tuning for scenario {scenario['name']}: {e}")
        continue

    final_model = RandomForestClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_train_val, y_train_val)
    
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    
    final_auroc = roc_auc_score(y_test, y_pred_proba)
    final_cohens_d = calculate_cohens_d(y_pred_proba, y_test)
    
    print(f"Test AUROC: {final_auroc:.4f}")
    print(f"Test Cohen's d: {final_cohens_d:.4f}")
    
    result_row = {
        'model': args.model,
        'data_type': args.folder,
        'test_scenario': scenario['name'],
        'test_chromosomes': ','.join(current_test_chroms),
        'train_val_chromosomes': ','.join(train_val_chroms),
        'cv_auroc': cv_score,
        'test_auroc': final_auroc,
        'cohens_d': final_cohens_d,
        'best_params': str(best_params)
    }
    all_results.append(result_row)


results_df = pd.DataFrame(all_results)
output_file = f"../results_final/causal_results/{args.model}_{args.folder}.csv"
results_df.to_csv(output_file, index=False)

print("FINAL RESULTS SUMMARY")
print(results_df[['test_scenario', 'cv_auroc', 'test_auroc', 'cohens_d']].to_string(index=False))
print(f"Mean test AUROC across scenarios: {results_df['test_auroc'].mean():.4f}")
print(f"Std test AUROC across scenarios: {results_df['test_auroc'].std():.4f}")
print(f"Results saved to '{output_file}'")