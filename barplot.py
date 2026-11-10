import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from run_usgs_model import AgriculturalRunoffNN
import warnings
warnings.filterwarnings('ignore')

# 1. Publication-Quality Formatting
plt.style.use('seaborn-v0_8-paper')
sns.set_theme(style="whitegrid", font="serif")
plt.rcParams.update({'font.family': 'serif', 'figure.dpi': 300})

# 2. Hardware & Data Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
df = pd.read_csv('usgs_ready_for_pytorch.csv')
df = df.fillna(df.mean(numeric_only=True))

X_full = df.iloc[:, :-3].values  
y_full = df.iloc[:, -3:].values 
target_names = ['Nitrate', 'Kjeldahl Nitrogen', 'Total Phosphorus']

# 3. Fast Training Loop for Predictions
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
X_train_t = torch.FloatTensor(X_train).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)

model = AgriculturalRunoffNN(input_dim=X_train.shape[1], output_dim=3).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(300):
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

model.eval()
with torch.no_grad():
    predictions = model(X_test_t).cpu().numpy()

# 4. Calculate the System-Wide Averages
true_means = y_test.mean(axis=0)
pred_means = predictions.mean(axis=0)

# 5. Create the Grouped Bar Plot
x = np.arange(len(target_names))
width = 0.35  

fig, ax = plt.subplots(figsize=(9, 6))
rects1 = ax.bar(x - width/2, true_means, width, label='Observed (USGS Data)', color='#2c3e50', edgecolor='black', linewidth=1.2)
rects2 = ax.bar(x + width/2, pred_means, width, label='Model Predicted', color='#27ae60', edgecolor='black', linewidth=1.2)

# 6. Formatting and Labels
ax.set_ylabel('Average Concentration (mg/L)', fontsize=12, fontweight='bold')
ax.set_title('System-Wide Environmental Load: Observed vs. Predicted', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(target_names, fontsize=11, fontweight='bold')
ax.legend(fontsize=11, loc='upper right', frameon=True, shadow=True)

# Add exact value labels on top of each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('journal_concentration_bars.png', dpi=300)
print("System alignment bar chart generated and saved successfully!")