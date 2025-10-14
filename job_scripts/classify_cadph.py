import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, accuracy_score
import argparse
import os

project_dir = ".."
parser = argparse.ArgumentParser(description='Process some files.')
parser.add_argument('--data_name', 
                    type=str, 
                    required=True,
                    help='The name of the dataset, meaning just the name excluding absolute path')
parser.add_argument('--pooling',
                    type=str,
                    choices=["sep", "mean", "max"],
                    default="sep",
                    help='The pooling method used for the output sequence of token embeddings')
parser.add_argument('--multiclass',
                    type=str,
                    choices=["yes", "no"],
                    default="no",
                    help='The number of classes')
parser.add_argument('--classifier',
                    type=str,
                    choices=["rf", "nb", "elastic"],
                    default="rf",
                    help='The classifier used')
args = parser.parse_args()

# Load the data
if args.pooling == "sep":
    train_data = pd.read_csv(f"{project_dir}/embeddings/{args.data_name}/train_embed_cadph.csv")
    test_data = pd.read_csv(f"{project_dir}/embeddings/{args.data_name}/test_embed_cadph.csv")
elif args.pooling == "mean":
    train_data = pd.read_csv(f"{project_dir}/embeddings/{args.data_name}/train_embed_cadph_meanpool.csv")
    test_data = pd.read_csv(f"{project_dir}/embeddings/{args.data_name}/test_embed_cadph_meanpool.csv")
else:
    train_data = pd.read_csv(f"{project_dir}/embeddings/{args.data_name}/train_embed_cadph_maxpool.csv")
    test_data = pd.read_csv(f"{project_dir}/embeddings/{args.data_name}/test_embed_cadph_maxpool.csv")

# Separate features and labels
X_train = train_data.drop('target', axis=1)
y_train = train_data['target']
X_test = test_data.drop('target', axis=1)
y_test = test_data['target']

if args.classifier == "rf":

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [1000, 500, 200],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [20, None],
        'min_samples_split': [2, 5]
    }

    rf = RandomForestClassifier(random_state=42)

    if args.multiclass == "no":
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=4, n_jobs=-1, scoring='roc_auc')
        grid_search.fit(X_train, y_train)

        best_rf = grid_search.best_estimator_

        y_pred = best_rf.predict(X_test)
        y_pred_proba = best_rf.predict_proba(X_test)[:, 1]  # Probability estimates for AUC

        mcc = matthews_corrcoef(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
    else:
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=4, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        best_rf = grid_search.best_estimator_

        y_pred = best_rf.predict(X_test)

        mcc = matthews_corrcoef(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, best_rf.predict_proba(X_test), multi_class='ovr')


    # Record the metrics into a CSV file
    results_df = pd.DataFrame({
        'Metric': ['MCC', 'AUC', 'F1-Score', 'Accuracy'],
        'Value': [mcc, auc, f1, accuracy]
    })

elif args.classifier == "nb":

    param_grid = {
        'var_smoothing': [1e-10, 1e-9, 1e-8]
    }

    nb = GaussianNB()

    if args.multiclass == "no":
        grid_search = GridSearchCV(estimator=nb, param_grid=param_grid, cv=4, n_jobs=-1, scoring='roc_auc')
        grid_search.fit(X_train, y_train)

        best_nb = grid_search.best_estimator_

        y_pred = best_nb.predict(X_test)
        y_pred_proba = best_nb.predict_proba(X_test)[:, 1]

        mcc = matthews_corrcoef(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
    else:
        grid_search = GridSearchCV(estimator=nb, param_grid=param_grid, cv=4, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        best_nb = grid_search.best_estimator_

        y_pred = best_nb.predict(X_test)
        
        mcc = matthews_corrcoef(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, best_nb.predict_proba(X_test), multi_class='ovr')

    # Record the metrics into a CSV file
    results_df = pd.DataFrame({
        'Metric': ['MCC', 'AUC', 'F1-Score', 'Accuracy'],
        'Value': [mcc, auc, f1, accuracy]
    })

elif args.classifier == "elastic":
    # Hyperparameter grid for Elastic Net Logistic Regression
    param_grid = {
        'C': [0.1, 1, 10],
        'l1_ratio': [0.1, 0.5, 0.9],  # mix of L1 and L2
        'class_weight': ['balanced', None],
        'max_iter': [1000],
        'tol': [1e-4]
    }

    # For elastic net, we use saga solver as it supports both l1 and l2 penalties
    lr = LogisticRegression(penalty='elasticnet', solver='saga', random_state=42)

    if args.multiclass == "no":
        grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=4, n_jobs=-1, scoring='roc_auc')
        grid_search.fit(X_train, y_train)

        best_lr = grid_search.best_estimator_

        y_pred = best_lr.predict(X_test)
        y_pred_proba = best_lr.predict_proba(X_test)[:, 1]

        mcc = matthews_corrcoef(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
    else:
        grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=4, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        best_lr = grid_search.best_estimator_

        y_pred = best_lr.predict(X_test)
        
        mcc = matthews_corrcoef(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, best_lr.predict_proba(X_test), multi_class='ovr')

    # Record the metrics into a CSV file
    results_df = pd.DataFrame({
        'Metric': ['MCC', 'AUC', 'F1-Score', 'Accuracy'],
        'Value': [mcc, auc, f1, accuracy]
    })



if args.pooling == "sep":
    os.makedirs(f"{project_dir}/results_final/cadph", exist_ok=True)
    os.makedirs(f"{project_dir}/preds/cadph", exist_ok=True)
    results_df.to_csv(f"{project_dir}/results_final/cadph/{args.data_name}_{args.classifier}.csv",
                      index=False)
    if args.multiclass == "no":
        preds_table = pd.DataFrame({
        'True': y_test,
        'Pred': y_pred_proba
        })
        preds_table.to_csv(f"{project_dir}/preds/cadph/{args.data_name}_{args.classifier}.csv",
                       index=False)

elif args.pooling == "mean":
    os.makedirs(f"{project_dir}/results_final/cadph_meanpool", exist_ok=True)
    os.makedirs(f"{project_dir}/preds/cadph_meanpool", exist_ok=True)
    results_df.to_csv(f"{project_dir}/results_final/cadph_meanpool/{args.data_name}_{args.classifier}.csv",
                      index=False)
    if args.multiclass == "no":
        preds_table = pd.DataFrame({
        'True': y_test,
        'Pred': y_pred_proba
        })
        preds_table.to_csv(f"{project_dir}/preds/cadph_meanpool/{args.data_name}_{args.classifier}.csv",
                       index=False)

else:
    os.makedirs(f"{project_dir}/results_final/cadph_maxpool", exist_ok=True)
    os.makedirs(f"{project_dir}/preds/cadph_maxpool", exist_ok=True)
    results_df.to_csv(f"{project_dir}/results_final/cadph_maxpool/{args.data_name}_{args.classifier}.csv",
                      index=False)
    if args.multiclass == "no":
        preds_table = pd.DataFrame({
        'True': y_test,
        'Pred': y_pred_proba
        })
        preds_table.to_csv(f"{project_dir}/preds/cadph_maxpool/{args.data_name}_{args.classifier}.csv",
                       index=False)

print(f"Metrics recorded in {args.data_name}.")