
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

# 1. Hardware & Data Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
df = pd.read_csv('usgs_ready_for_pytorch.csv')
df = df.fillna(df.mean(numeric_only=True))

# Extract features and targets
X_full = df.iloc[:, :-3].values  
y_full = df.iloc[:, -3:].values 
target_names = ['Nitrate', 'Kjeldahl nitrogen', 'Total Phosphorus']

# --- GRAPH 1: Feature Correlation Heatmap ---
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Dataset Feature & Target Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('journal_1_correlation_heatmap.png', dpi=300)
plt.close()

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

# --- GRAPH 2: Training Loss Curve ---
plt.figure(figsize=(8, 5))
plt.plot(range(300), epoch_losses, color='purple', linewidth=2)
plt.title('Model Convergence: Training Loss over 300 Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Mean Squared Error (MSE) Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('journal_2_loss_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Final Evaluation & Predictions
model.eval()
with torch.no_grad():
    predictions = model(X_test_t).cpu().numpy()

# --- GRAPH 3: Actual vs Predicted (Kjeldahl Nitrogen) ---
# Kjeldahl nitrogen is at index 1
true_nitrogen = y_test[:, 1]
pred_nitrogen = predictions[:, 1]

plt.figure(figsize=(8, 5))
plt.scatter(true_nitrogen, pred_nitrogen, alpha=0.6, color='#ff7f0e', edgecolor='k')
# Plotting the perfect prediction diagonal line
min_val = min(true_nitrogen.min(), pred_nitrogen.min())
max_val = max(true_nitrogen.max(), pred_nitrogen.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

plt.title('Actual vs. Predicted: Kjeldahl Nitrogen', fontsize=14, fontweight='bold')
plt.xlabel('True USGS Measurements', fontsize=12)
plt.ylabel('Model Predictions', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('journal_3_scatter_nitrogen.png', dpi=300, bbox_inches='tight')
plt.close()

# --- GRAPH 4: Final RMSE Bar Chart ---
rmse_values = [np.sqrt(mean_squared_error(y_test[:, i], predictions[:, i])) for i in range(3)]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

plt.figure(figsize=(8, 5))
bars = plt.bar(target_names, rmse_values, color=colors, edgecolor='black')
plt.title('Final Predictive Error (RMSE) by Pollutant', fontsize=14, fontweight='bold')
plt.ylabel('Root Mean Squared Error', fontsize=12)

# Add exact values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('journal_4_rmse_bars.png', dpi=300, bbox_inches='tight')
plt.close()

print("All 4 journal-ready graphs generated successfully!")