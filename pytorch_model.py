import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load your new real-world USGS dataset
df = pd.read_csv('usgs_ready_for_pytorch.csv')

# Choose your actual target pollutants from your new columns list
# (Make sure these match your exact CSV headers)
target_columns = [
    'Nitrate', 
    'Orthophosphate', 
    'Kjeldahl nitrogen', 
    'Total Phosphorus, mixed forms'
]

# X contains all your physical sensors, y contains your target pollutants
X = df.drop(columns=target_columns)
y = df[target_columns]

# Handle any remaining missing values
X = X.fillna(X.mean())
y = y.fillna(y.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features (X) and Targets (y)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Convert to PyTorch Tensors
X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.FloatTensor(y_train_scaled)
X_test_t = torch.FloatTensor(X_test_scaled)
y_test_t = torch.FloatTensor(y_test_scaled)