import pandas as pd

project_dir = ".."
df = pd.read_csv(f'{project_dir}/data_processed/TAD/insulation_IMR90_insulation_2048bp_window10.tsv', sep='\t')

df = df.dropna(subset=['boundary_strength_20480'])
threshold = df['boundary_strength_20480'].quantile(0.95)
print(f"90th percentile threshold for boundary strength is: {threshold}")

# Select only rows above this threshold
high_strength_df = df[df['boundary_strength_20480'] > threshold]
print(f"Number of rows above the top 5% threshold: {len(high_strength_df)}")

sample_size = min(1500, len(high_strength_df))
sampled_df = high_strength_df.sample(n=sample_size, random_state=42)
sampled_df.to_csv(f'{project_dir}/data_processed/TAD/boundaries_selected.tsv', sep='\t', index=False)