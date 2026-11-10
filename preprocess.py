
import pandas as pd

# Load your dataset
df = pd.read_csv('Water_Quality_Assessment.csv')

# 1. Statistical Summary (Mean, Min, Max, Standard Deviation for each column)
print("--- Statistical Summary ---")
print(df.describe())

# 2. Correlation Matrix (Checking how parameters relate to BOD or your target variable)
# Replace 'BOD' with the exact name of your target column if it's spelled differently in your CSV
print("\n--- Correlation with BOD ---")
correlation = df.corr(numeric_only=True)['BOD (mg/L)'].sort_values(ascending=False)
print(correlation)