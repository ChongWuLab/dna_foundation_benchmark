import numpy as np
import matplotlib.pyplot as plt

bg = np.load("../attention_weights/ntv2_background.npy")
tad = np.load("../attention_weights/ntv2_tad.npy")

# Compute the difference
diff = tad - bg

# Plot the difference as a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(diff, cmap='coolwarm', aspect='auto')
plt.colorbar(label='Difference')
plt.title('Heatmap of difference between TAD and Background Attention')
# Save the figure to a file
plt.savefig('../results_final/tad_results/heatmap_difference.png', dpi=300)