
import pandas as pd
import numpy as np

print("Loading raw USGS data (this might take a few seconds)...")
# low_memory=False prevents pandas from crashing on large government files
df = pd.read_csv('usgs_raw.csv', low_memory=False)

# 1. Keep only the necessary columns
columns_to_keep = [
    'MonitoringLocationIdentifier', # The specific river/stream site
    'ActivityStartDate',            # The date the sample was taken
    'CharacteristicName',           # The name of the sensor/pollutant (e.g., 'pH')
    'ResultMeasureValue'            # The actual numeric reading
]
df = df[columns_to_keep]

# 2. Clean the numeric values (Government data often has text like "< 0.05" mixed with numbers)
df['ResultMeasureValue'] = pd.to_numeric(df['ResultMeasureValue'], errors='coerce')

# Drop any rows where the value was empty or text-only
df = df.dropna(subset=['ResultMeasureValue'])

# 3. Pivot the Table (Transform from Long to Wide format)
print("Pivoting data for PyTorch...")
df_wide = df.pivot_table(
    index=['MonitoringLocationIdentifier', 'ActivityStartDate'], 
    columns='CharacteristicName', 
    values='ResultMeasureValue', 
    aggfunc='mean' # If they took two pH readings on the same day, take the average
).reset_index()

# 4. Drop the ID and Date columns (The neural network only wants math, not text)
df_wide = df_wide.drop(columns=['MonitoringLocationIdentifier', 'ActivityStartDate'])

# 5. Drop columns (pollutants/sensors) that have too much missing data (keep columns with at least 20% valid data)
thresh = int(len(df_wide) * 0.2)
df_wide = df_wide.dropna(axis=1, thresh=thresh)

# Save the final, PyTorch-ready dataset
df_wide.to_csv('usgs_ready_for_pytorch.csv', index=False)

print("\nSuccess! Data cleaned and saved as 'usgs_ready_for_pytorch.csv'.")
print(f"Final Dataset Shape: {df_wide.shape[0]} rows, {df_wide.shape[1]} features.")
print("\nAvailable Columns (Sensors and Pollutants):")
for col in df_wide.columns:
    print(f"- {col}")