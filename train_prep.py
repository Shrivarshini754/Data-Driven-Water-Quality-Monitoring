
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
df = pd.read_csv('Water_Quality_Assessment.csv')

# Note: If your BOD column has a specific name like 'BOD (mg/L)', 
# make sure to change 'BOD' below to match your exact column name!
target_column = 'BOD (mg/L)' 

# 2. Separate Features (X) and Target (y)
# X contains all input parameters (everything except BOD)
# y contains the target output we want to predict (BOD)
X = df.drop(columns=[target_column])
y = df[target_column]

# 3. Handle any missing values (filling them with the column mean)
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# 4. Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Normalize/Scale the features 
# This ensures parameters with large values don't overpower smaller ones (like pH)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Preprocessing complete!")
print(f"Training samples shape: {X_train_scaled.shape}")
print(f"Testing samples shape: {X_test_scaled.shape}")