import pandas as pd

# Added low_memory=False to suppress the mixed-type warning
df_raw = pd.read_csv('usgs_raw_rich.csv', low_memory=False) 

total_rows = len(df_raw)
n_train = int(total_rows * 0.8)
n_test = total_rows - n_train

print(f"Total Observations: {total_rows}")
print(f"n_train: {n_train}")
print(f"n_test: {n_test}")

if 'ActivityStartDate' in df_raw.columns:
    # Convert to proper datetime and drop any blank/corrupt cells
    valid_dates = pd.to_datetime(df_raw['ActivityStartDate'], errors='coerce').dropna()
    print(f"Date Range: {valid_dates.min().strftime('%Y-%m-%d')} to {valid_dates.max().strftime('%Y-%m-%d')}")

if 'MonitoringLocationIdentifier' in df_raw.columns:
    print(f"Distinct Monitoring Stations: {df_raw['MonitoringLocationIdentifier'].nunique()}")
    
if 'StateCode' in df_raw.columns:
    print(f"Number of States/Regions: {df_raw['StateCode'].nunique()}")