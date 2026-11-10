
import pandas as pd

# Load your dataset (make sure the filename matches what you saved)
df = pd.read_csv('Water_Quality_Assessment.csv')

# Display the first 5 rows to verify it loaded correctly
print("--- First 5 Rows ---")
print(df.head())

# Display basic info (column names, data types, and non-null counts)
print("\n--- Dataset Info ---")
print(df.info())

# Check for any missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())