import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load and clean data (using your exact PyTorch pipeline)
df = pd.read_csv('usgs_ready_for_pytorch.csv')
df = df.fillna(df.mean(numeric_only=True))

X_full = df.iloc[:, :-3].values
y_full = df.iloc[:, -3:].values
target_names = ['Nitrate', 'Kjeldahl Nitrogen', 'Total Phosphorus']

# 2. Split and Scale Data
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train Linear Regression Baseline
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)

# 4. Train Random Forest Baseline
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_preds = rf.predict(X_test_scaled)

# 5. Print Comparison Metrics
print("--- BASELINE RESULTS FOR MANUSCRIPT ---")
for i, target in enumerate(target_names):
    print(f"\n{target}:")
    print(f"Linear Regression -> RMSE: {np.sqrt(mean_squared_error(y_test[:, i], lr_preds[:, i])):.2f}, R2: {r2_score(y_test[:, i], lr_preds[:, i]):.2f}")
    print(f"Random Forest     -> RMSE: {np.sqrt(mean_squared_error(y_test[:, i], rf_preds[:, i])):.2f}, R2: {r2_score(y_test[:, i], rf_preds[:, i]):.2f}")