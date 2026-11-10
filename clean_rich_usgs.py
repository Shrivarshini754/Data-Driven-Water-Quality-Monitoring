
import pandas as pd
import numpy as np

print("Processing raw government data...")
df = pd.read_csv('usgs_raw_rich.csv', low_memory=False)

# Keep essential mapping columns
df = df[['MonitoringLocationIdentifier', 'ActivityStartDate', 'CharacteristicName', 'ResultMeasureValue']]

# Convert results to numeric, dropping text-only artifacts
df['ResultMeasureValue'] = pd.to_numeric(df['ResultMeasureValue'], errors='coerce')
df = df.dropna(subset=['ResultMeasureValue'])

# Pivot from Long Format to Wide Format
print("Pivoting data matrix...")
df_wide = df.pivot_table(
    index=['MonitoringLocationIdentifier', 'ActivityStartDate'],
    columns='CharacteristicName',
    values='ResultMeasureValue',
    aggfunc='mean'
).reset_index()

# Drop identifier text columns
df_wide = df_wide.drop(columns=['MonitoringLocationIdentifier', 'ActivityStartDate'])

# Keep columns that have a strong density of data (at least 40% valid rows to ensure robust multi-target outputs)
thresh = int(len(df_wide) * 0.4)
df_wide = df_wide.dropna(axis=1, thresh=thresh)

# Fill remaining isolated blanks with column means
df_wide = df_wide.fillna(df_wide.mean())

# Save final multi-target dataset
df_wide.to_csv('usgs_multitarget_ready.csv', index=False)
print("\nSuccess! Saved as 'usgs_multitarget_ready.csv'.")
print(f"Total Columns Available: {list(df_wide.columns)}")
