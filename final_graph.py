
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Define Sample Sizes and Target Variables
# Adjusted to match your final real-world dataset size
sample_sizes = [500, 1000, 2000, 2884]
pollutants = ['Nitrate', 'Kjeldahl nitrogen', 'Total Phosphorus']

# 2. Cleaned Real Data from your PyTorch Loop
rmse_data = {
    'Nitrate': [8.9082, 9.205, 12.834, 11.8822], 
    'Kjeldahl nitrogen': [9.018, 12.975, 8.9802, 14.5764], 
    'Total Phosphorus': [1.4603, 1.4444, 1.6906, 1.4996]
}

r2_data = {
    # The massive negative outlier for Nitrate at 500 is visually capped at -8 
    # to match the scale of Total Phosphorus so the graph remains readable.
    'Nitrate': [-8.0, -1.4499, -0.5946, -0.4653], 
    'Kjeldahl nitrogen': [0.9624, 0.8493, 0.8740, 0.9410], 
    'Total Phosphorus': [-7.1699, -6.4539, -5.5531, -4.6998]
}

# 3. Automatically Calculate Predictive Status Based on R2 Scores
status_data = {}
for pol in pollutants:
    status_data[pol] = []
    for r2 in r2_data[pol]:
        if r2 < 0.30:
            status_data[pol].append(1) # Poor (Captures all negative scores)
        elif r2 < 0.50:
            status_data[pol].append(2) # Low-Moderate
        elif r2 < 0.70:
            status_data[pol].append(3) # Moderate-Strong
        else:
            status_data[pol].append(4) # Strong

status_labels = ['Poor', 'Low-Moderate', 'Moderate-Strong', 'Strong']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
plt.style.use('default')

# --- GRAPH 1: RMSE vs Sample Size ---
plt.figure(figsize=(8, 5))
for idx, pol in enumerate(pollutants):
    plt.plot(sample_sizes, rmse_data[pol], marker='o', linewidth=2, label=pol, color=colors[idx])
plt.title('Impact of Sample Size on Model Error (RMSE)', fontsize=14, fontweight='bold')
plt.xlabel('Sample Size (Number of Records)', fontsize=12)
plt.ylabel('RMSE (Normalized)', fontsize=12)
plt.xticks(sample_sizes)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig('actual_rmse_vs_samplesize.png', dpi=300, bbox_inches='tight')

# --- GRAPH 2: R-squared vs Sample Size ---
plt.figure(figsize=(8, 5))
for idx, pol in enumerate(pollutants):
    plt.plot(sample_sizes, r2_data[pol], marker='s', linewidth=2, label=pol, color=colors[idx])
plt.title('Impact of Sample Size on Variance Explained (R-squared)', fontsize=14, fontweight='bold')
plt.xlabel('Sample Size (Number of Records)', fontsize=12)
plt.ylabel('R-squared Score', fontsize=12)
plt.xticks(sample_sizes)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')
plt.savefig('actual_r2_vs_samplesize.png', dpi=300, bbox_inches='tight')

# --- GRAPH 3: Predictive Status vs Sample Size (Heatmap) ---
df_status = pd.DataFrame(status_data, index=sample_sizes).T
plt.figure(figsize=(8, 4))

cax = plt.matshow(df_status, cmap='Blues', vmin=1, vmax=4, fignum=False, aspect='auto')
plt.title('Evolution of Predictive Status by Sample Size\n', fontsize=14, fontweight='bold')
plt.xlabel('Sample Size (Number of Records)', fontsize=12)

plt.xticks(range(len(sample_sizes)), sample_sizes)
plt.yticks(range(len(pollutants)), pollutants)

# Custom colorbar for categorical labels
cbar = plt.colorbar(cax, ticks=[1.375, 2.125, 2.875, 3.625])
cbar.ax.set_yticklabels(status_labels)

plt.savefig('actual_status_vs_samplesize.png', dpi=300, bbox_inches='tight')
print("All final, real-world graphs successfully generated and saved!")