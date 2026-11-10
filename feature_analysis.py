import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv('usgs_ready_for_pytorch.csv')
df = df.fillna(df.mean(numeric_only=True))

# 2. Isolate features (X) and targets (Y)
features = df.columns[:-3]
targets = df.columns[-3:]

# 3. Calculate Pearson correlation
corr_matrix = df.corr()
target_correlations = corr_matrix.loc[features, targets]

# 4. Generate and save the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(target_correlations, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title('Correlation Matrix: Physical Sensors vs. Target Pollutants')
plt.ylabel('Physical Sensor Parameters')
plt.xlabel('Target Pollutants')
plt.tight_layout()

plt.savefig('fig_correlation.png', dpi=300)
print("Saved fig_correlation.png for your manuscript.")