import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.read_csv(file_path)


# Read the data
data1 = load_data('dnabert2_ordered.csv')
data2 = load_data('dnabert2_meanpool_ordered.csv')
data3 = load_data('ntv2_ordered.csv')
data4 = load_data('ntv2_meanpool_ordered.csv')
data5 = load_data('hyena_ordered.csv')
data6 = load_data('hyena_meanpool_ordered.csv')

# Prepare the data for plotting by creating a unified DataFrame with an identifier column
data1['Source'] = 'DNABERT2'
data2['Source'] = 'DNABERT2 Meanpool'
data3['Source'] = 'NTV2'
data4['Source'] = 'NTV2 Meanpool'
data5['Source'] = 'Hyena'
data6['Source'] = 'Hyena Meanpool'


# Combine all data into one DataFrame
all_data_combined = pd.concat([
    data1[['AUC', 'Source']], data2[['AUC', 'Source']],
    data3[['AUC', 'Source']], data4[['AUC', 'Source']],
    data5[['AUC', 'Source']], data6[['AUC', 'Source']]
])


sns.set_theme(style="whitegrid", context="talk")
fig, axes = plt.subplots(1, 3, figsize=(12, 7), sharey=True)
palette = ["Set2", "Set1", "Pastel1"]

# Datasets by pairs
pair_names = [("DNABERT2 Meanpool", "DNABERT2"), ("NTV2 Meanpool", "NTV2"), ("Hyena Meanpool", "Hyena")]


# Plot each pair in a subplot
for i, (pair, ax) in enumerate(zip(pair_names, axes)):
    subset = all_data_combined[all_data_combined['Source'].isin(pair)]
    sns.boxplot(x='Source', y='AUC', data=subset, ax=ax, palette=palette[i], linewidth=2.5, width=0.6)
    if pair[1] == "Hyena":
        ax.set_title("HyenaDNA", fontsize=18, fontweight='bold')
    elif pair[1] == "DNABERT2":
        ax.set_title("DNABERT-2", fontsize=18, fontweight='bold')
    else:
        ax.set_title("NT-v2", fontsize=18, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('AUC Score' if i == 0 else '', weight="bold", labelpad=10)
    ax.grid(True, linestyle='-.')

    ax.set_xticks(range(2))
    if pair[1] == "Hyena":
        ax.set_xticklabels(["EOS Pooling", "Mean Pooling"], fontsize=16)
    else:
        ax.set_xticklabels(["CLS Pooling", "Mean Pooling"], fontsize=16)


plt.tight_layout()
plt.savefig("./plots/box.png", dpi=200)
