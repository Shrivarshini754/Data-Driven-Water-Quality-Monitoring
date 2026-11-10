import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Define Sample Sizes and Target Variables
sample_sizes = [500, 1000, 2000, 3000]
pollutants = ['Kjeldahl Nitrogen', 'Total Phosphorus', 'Nitrate']

# Simulating RMSE (Error drops as sample size increases)
rmse_data = {
    'Kjeldahl Nitrogen': [0.45, 0.35, 0.28, 0.24],
    'Total Phosphorus': [0.38, 0.29, 0.22, 0.18],
    'Nitrate': [0.65, 0.55, 0.48, 0.42]
}

# Simulating R2 (Correlation improves as sample size increases)
r2_data = {
    'Kjeldahl Nitrogen': [0.40, 0.48, 0.55, 0.60],
    'Total Phosphorus': [0.35, 0.45, 0.52, 0.56],
    'Nitrate': [0.15, 0.20, 0.25, 0.30]
}

# Simulating Status mapping (1: Poor, 2: Low-Moderate, 3: Moderate-Strong, 4: Strong)
status_data = {
    'Kjeldahl Nitrogen': [2, 2, 3, 3],
    'Total Phosphorus': [2, 3, 3, 3],
    'Nitrate': [1, 1, 2, 2]
}
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
plt.tight_layout()
plt.savefig('rmse_vs_samplesize.png', dpi=300)

# --- GRAPH 2: R-squared vs Sample Size ---
plt.figure(figsize=(8, 5))
for idx, pol in enumerate(pollutants):
    plt.plot(sample_sizes, r2_data[pol], marker='s', linewidth=2, label=pol, color=colors[idx])
plt.title('Impact of Sample Size on Variance Explained ($R^2$)', fontsize=14, fontweight='bold')
plt.xlabel('Sample Size (Number of Records)', fontsize=12)
plt.ylabel('$R^2$ Score', fontsize=12)
plt.xticks(sample_sizes)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('r2_vs_samplesize.png', dpi=300)

# --- GRAPH 3: Predictive Status vs Sample Size (Heatmap) ---
# --- GRAPH 3: Predictive Status vs Sample Size (Heatmap) ---
df_status = pd.DataFrame(status_data, index=sample_sizes).T
plt.figure(figsize=(8, 4))

# FIX: Changed fignum=1 to fignum=False so it doesn't overlap Graph 1
cax = plt.matshow(df_status, cmap='Blues', vmin=1, vmax=4, fignum=False, aspect='auto') 

plt.title('Evolution of Predictive Status by Sample Size\n', fontsize=14, fontweight='bold')
plt.xlabel('Sample Size (Number of Records)', fontsize=12)

plt.xticks(range(len(sample_sizes)), sample_sizes)
plt.yticks(range(len(pollutants)), pollutants)

# Custom colorbar for categorical labels
cbar = plt.colorbar(cax, ticks=[1.375, 2.125, 2.875, 3.625])
cbar.ax.set_yticklabels(status_labels)

# Save figure cleanly without warnings
plt.savefig('status_vs_samplesize.png', dpi=300, bbox_inches='tight')
plt.show()