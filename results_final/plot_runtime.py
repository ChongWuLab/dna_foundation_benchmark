import pandas as pd
import matplotlib.pyplot as plt
import seaborn

# Load the CSV file to check its content
file_path = 'runtime_table.csv'
data = pd.read_csv(file_path)

# Define the desired order and the corresponding x-axis custom values
desired_order = ['4mC_E.coli', 'prom_core_notata', 'binary', 'enhancer_ensembl', 'regulatory', 'H3K36me3', 'covid', "Hela-S3", 'HUVEC', "GM12878"]
x_axis_values = [41, 70, 199, 269, 401, 500, 999, 1113, 1267, 1622]

# Reordering the dataframe according to the specified order
filtered_data_ordered = data.set_index('Data').loc[desired_order].reset_index()

# Evenly spaced x-axis indices for plotting
even_x_indices = range(len(x_axis_values))

# Plotting with evenly spaced x-axis labels
plt.style.use('seaborn-v0_8-poster')
plt.figure(figsize=(9, 5))
plt.plot(even_x_indices, filtered_data_ordered['DNABERT2 Mean Pool'], marker='o', linestyle='-', color='b', label='DNABERT-2', markersize=9)
plt.plot(even_x_indices, filtered_data_ordered['NT Mean Pool'], marker='o', linestyle='-', color='r', label='NT-v2', markersize=9)
plt.plot(even_x_indices, filtered_data_ordered['HyenaDNA Mean Pool'], marker='o', linestyle='-', color='g', label='HyenaDNA', markersize=9)
plt.xlabel('Median Number of Nucleotides', fontsize=12, labelpad=10)
plt.ylabel('Average Runtime in Seconds', fontsize=12)
plt.xticks(even_x_indices, x_axis_values, fontsize=10, weight="bold")
plt.yticks(fontsize=10, weight="bold")
plt.legend(fontsize='large', title_fontsize='13')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.savefig("./plots/runtime.png", dpi=200)


