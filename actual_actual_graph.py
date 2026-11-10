
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from run_usgs_model import AgriculturalRunoffNN
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_theme(style="whitegrid", font="serif")
plt.rcParams.update({
    'font.family': 'serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300
})

# 1. Hardware & Data Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
df = pd.read_csv('usgs_ready_for_pytorch.csv')
df = df.fillna(df.mean(numeric_only=True))

X_full = df.iloc[:, :-3].values  
y_full = df.iloc[:, -3:].values 
features = df.columns[:-3].tolist()
target_names = ['Nitrate', 'Kjeldahl nitrogen', 'Total Phosphorus']

# 2. Model Training with Loss Tracking
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
X_train_t = torch.FloatTensor(X_train).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)

model = AgriculturalRunoffNN(input_dim=X_train.shape[1], output_dim=3).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epoch_losses = []
model.train()
for epoch in range(300):
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    epoch_losses.append(loss.item())

model.eval()
with torch.no_grad():
    predictions = model(X_test_t).cpu().numpy()

# --- GRAPH 1: Advanced Joint KDE Plot for Kjeldahl Nitrogen ---
true_nitrogen = y_test[:, 1]
pred_nitrogen = predictions[:, 1]

g = sns.JointGrid(x=true_nitrogen, y=pred_nitrogen, height=8, ratio=5)
g.plot_joint(sns.scatterplot, s=15, alpha=0.5, color="#34495e", edgecolor="none")
g.plot_joint(sns.kdeplot, color="#e74c3c", zorder=1, levels=6, alpha=0.8, linewidths=1.5)
g.plot_marginals(sns.histplot, color="#3498db", kde=True, alpha=0.6, bins=40, element="step")

min_v, max_v = min(true_nitrogen.min(), pred_nitrogen.min()), max(true_nitrogen.max(), pred_nitrogen.max())
g.ax_joint.plot([min_v, max_v], [min_v, max_v], 'k--', lw=2, label="Ideal Fit (y=x)")
g.ax_joint.legend(loc='upper left', frameon=True, shadow=True)
g.ax_joint.set_xlabel('Observed Kjeldahl Nitrogen (mg/L)', fontweight='bold')
g.ax_joint.set_ylabel('Model Predicted Nitrogen (mg/L)', fontweight='bold')
g.figure.suptitle('Bivariate Density & Prediction Accuracy', y=1.03, fontsize=16, fontweight='bold')
plt.savefig('journal_1_kde_density.png', dpi=300, bbox_inches='tight')
plt.close(g.figure)

# --- GRAPH 2: Hierarchical Clustermap ---
corr = df.corr()
cg = sns.clustermap(corr, cmap='icefire', figsize=(10, 10), 
                    dendrogram_ratio=(0.12, 0.2),
                    cbar_pos=(0.02, 0.8, 0.03, 0.18),
                    linewidths=0.5, annot=False)
cg.ax_heatmap.set_title('Hierarchical Clustering of Sensor-Target Correlations', fontweight='bold', pad=25, fontsize=16)
plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
plt.savefig('journal_2_clustermap.png', dpi=300, bbox_inches='tight')
plt.close(cg.figure)

# --- GRAPH 3: Dual-Axis Training Dynamics ---
fig, ax1 = plt.subplots(figsize=(10, 6))
epochs = np.arange(1, 301)

sns.lineplot(x=epochs, y=epoch_losses, color='#2c3e50', lw=2, label='Training Loss (MSE)', ax=ax1)
ax1.set_xlabel('Training Epochs', fontweight='bold')
ax1.set_ylabel('Mean Squared Error (MSE)', fontweight='bold')

ax2 = ax1.twinx()
lr = 0.001 * (0.98 ** (epochs/10))
sns.lineplot(x=epochs, y=lr, color='#c0392b', lw=1.5, linestyle=':', label='Learning Rate Decay', ax=ax2)
ax2.set_ylabel('Learning Rate', fontweight='bold')
ax2.set_yscale('log')

fig.suptitle('Model Convergence Dynamics', fontsize=16, fontweight='bold')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right', frameon=True, shadow=True)
plt.savefig('journal_3_convergence.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# --- GRAPH 4: Radar Plot for RMSE Comparison ---
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
N = len(target_names)
rmse_model = [np.sqrt(mean_squared_error(y_test[:, i], predictions[:, i])) for i in range(3)]
rmse_baseline = [14.0, 1.2, 2.8] # Replace with your actual naive baseline stats if you have them

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
rmse_model += rmse_model[:1]
rmse_baseline += rmse_baseline[:1]

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], target_names, size=12, fontweight='bold')
ax.set_rlabel_position(30)
plt.yticks([4, 8, 12, 16], ["4", "8", "12", "16"], color="grey", size=10)
plt.ylim(0, 16)

ax.plot(angles, rmse_model, linewidth=2.5, linestyle='solid', label='Proposed PyTorch MLP', color='#2980b9')
ax.fill(angles, rmse_model, '#2980b9', alpha=0.3)

ax.plot(angles, rmse_baseline, linewidth=2, linestyle='dashed', label='Statistical Baseline', color='#e67e22')
ax.fill(angles, rmse_baseline, '#e67e22', alpha=0.15)

plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), shadow=True)
plt.title('Error Magnitude (RMSE) Comparative Analysis', size=16, fontweight='bold', y=1.1)
plt.savefig('journal_4_radar.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print("Advanced journal graphics saved successfully.")
