import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, accuracy_score
import argparse

# Load command line arguments
parser = argparse.ArgumentParser(description='Process some files.')
parser.add_argument('--data_path', 
                    type=str, 
                    required=True,
                    help='The path of the dataset, specifically, the directory that the train.csv and test.csv lies in')
parser.add_argument('--data_name', 
                    type=str, 
                    required=True,
                    help='The name of the dataset, meaning just the name excluding absolute path')
parser.add_argument('--pooling',
                    type=str,
                    choices=["cls", "mean"],
                    default="cls",
                    help='The pooling method used for the output sequence of token embeddings')
parser.add_argument('--multiclass',
                    type=str,
                    choices=["yes", "no"],
                    default="no",
                    help='The number of classes')
args = parser.parse_args()

# Load the data
if args.pooling == "cls":
    train_data = pd.read_csv(f"{args.data_path}/results/train_embed_dnabert.csv")
    test_data = pd.read_csv(f"{args.data_path}/results/test_embed_dnabert.csv")
else:
    train_data = pd.read_csv(f"{args.data_path}/results/train_embed_dnabert_meanpool.csv")
    test_data = pd.read_csv(f"{args.data_path}/results/test_embed_dnabert_meanpool.csv")

# Separate features and labels
X_train = train_data.drop('target', axis=1)
y_train = train_data['target']
X_test = test_data.drop('target', axis=1)
y_test = test_data['target']

# Hyperparameter grid
param_grid = {
    'n_estimators': [1000, 500, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [20, None],
    'min_samples_split': [2, 5]
}

# Initialize the RandomForestClassifier
rf = RandomForestClassifier()

if args.multiclass == "no":
    # Initialize GridSearchCV with 4-fold cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=4, n_jobs=-1, scoring='roc_auc')

    # Perform grid search on the training data
    grid_search.fit(X_train, y_train)

    # Retrieve the best estimator
    best_rf = grid_search.best_estimator_

    # Make predictions on test data
    y_pred = best_rf.predict(X_test)
    y_pred_proba = best_rf.predict_proba(X_test)[:, 1]  # Probability estimates for AUC

    # Calculate metrics
    mcc = matthews_corrcoef(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

else:
    # Initialize GridSearchCV with 4-fold cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=4, n_jobs=-1, scoring='accuracy')

    # Perform grid search on the training data
    grid_search.fit(X_train, y_train)

    # Retrieve the best estimator
    best_rf = grid_search.best_estimator_

    # Make predictions on test data
    y_pred = best_rf.predict(X_test)

    # Calculate metrics
    mcc = matthews_corrcoef(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, best_rf.predict_proba(X_test), multi_class='ovr')



# Record the metrics into a CSV file
results_df = pd.DataFrame({
    'Metric': ['MCC', 'AUC', 'F1-Score', 'Accuracy'],
    'Value': [mcc, auc, f1, accuracy]
})



if args.pooling == "cls":
    results_df.to_csv(f"/rsrch4/home/biostatistics/hfeng3/review_datasets/results_final/dnabert2/{args.data_name}.csv",
                      index=False)
    if args.multiclass == "no":
        preds_table = pd.DataFrame({
        'True': y_test,
        'Pred': y_pred_proba
        })
        preds_table.to_csv(f"/rsrch4/home/biostatistics/hfeng3/review_datasets/preds/dnabert2/{args.data_name}.csv",
                       index=False)
        
else:
    results_df.to_csv(f"/rsrch4/home/biostatistics/hfeng3/review_datasets/results_final/dnabert2_meanpool/{args.data_name}.csv",
                      index=False)
    if args.multiclass == "no":
        preds_table = pd.DataFrame({
        'True': y_test,
        'Pred': y_pred_proba
        })
        preds_table.to_csv(f"/rsrch4/home/biostatistics/hfeng3/review_datasets/preds/dnabert2_meanpool/{args.data_name}.csv",
                       index=False)

print(f"Metrics recorded in {args.data_name}.")