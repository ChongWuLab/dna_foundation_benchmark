import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


expression_path = '../data_processed/GTEX/Whole_Blood.v8.normalized_expression.bed'
covariates_path = '../data_processed/GTEX/Whole_Blood.v8.covariates.txt'
output_path = f'../data_processed/GTEX/Whole_Blood.v8.expression_regressed.tsv'

expression_df = pd.read_csv(expression_path, sep='\t')
covariates_df = pd.read_csv(covariates_path, sep='\t', index_col=0)

# Convert all values to numeric
for col in covariates_df.columns:
    covariates_df[col] = pd.to_numeric(covariates_df[col], errors='coerce')

# Make the last covariate one-hot (sex)
covariates_df.loc[covariates_df.index[-1]] = covariates_df.loc[covariates_df.index[-1]] - 1

# Transpose so subjects are rows and covariates are columns
covariates_df = covariates_df.transpose()


meta_columns = expression_df.iloc[:, :4].copy()
subject_cols = expression_df.columns[4:]

covariate_matrix = covariates_df.values
expression_matrix = expression_df.iloc[:, 4:].values

# Create a matrix to store residuals
residuals_matrix = np.zeros_like(expression_matrix)

# Process each gene
print(f"Regressing covariates for {len(expression_df)} genes")
for i in range(len(expression_df)):
    if i % 1000 == 0:
        print(f"Processing gene {i}/{len(expression_df)}")
    
    # Get expression values for this gene
    y = expression_matrix[i]
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(covariate_matrix, y)
    
    # Calculate residuals
    y_pred = model.predict(covariate_matrix)
    residuals_matrix[i] = y - y_pred


residuals_df = pd.DataFrame(residuals_matrix, columns=subject_cols)
result_df = pd.concat([meta_columns, residuals_df], axis=1)
    
print(f"Saving to {output_path}")
result_df.to_csv(output_path, sep='\t', index=False)
print("Done")