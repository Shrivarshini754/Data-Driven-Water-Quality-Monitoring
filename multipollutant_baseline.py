
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Load the dataset
df = pd.read_csv('Water_Quality_Assessment.csv')

# 2. Define your target pollutant columns 
# (Make sure these match your exact CSV column headers if they have units like 'Nitrate (mg/L)')
target_columns = ['BOD (mg/L)', 'COD (mg/L)', 'Nitrate (NO3⁻) (mg/L)', 'Phosphate (PO4³⁻) (mg/L)', 'Ammonia (mg/L)', 'Coliform Bacteria (CFU/mL)']

# X contains all physical sensor features (everything except the targets)
X = df.drop(columns=target_columns)

# y contains ALL your distinct pollutants to predict simultaneously
y = df[target_columns]

# 3. Handle missing values
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# 4. Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train the Multi-Target Random Forest Model
print("Training the multi-pollutant baseline model across all targets...")
mt_model = RandomForestRegressor(n_estimators=100, random_state=42)
mt_model.fit(X_train_scaled, y_train)

# 7. Evaluate Model Reliability for Each Pollutant Individually
y_pred = mt_model.predict(X_test_scaled)

print("\n==============================================")
print("     MULTI-POLLUTANT BASELINE RELIABILITY     ")
print("==============================================")
for i, col in enumerate(target_columns):
    mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    
    print(f"Pollutant: {col}")
    print(f"  -> RMSE (Error Magnitude): {rmse:.4f}")
    print(f"  -> R2 Score (Reliability/Accuracy): {r2:.4f}")
    print("-" * 45)