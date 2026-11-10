import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Import your specific model class
from run_usgs_model import AgriculturalRunoffNN

# 2. Hardware Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Active processing device: {device}\n")

# 3. Load and Clean Your Specific Data 
df = pd.read_csv('usgs_ready_for_pytorch.csv')

# FIX: Use Mean Imputation instead of dropping rows to preserve all 2,884 samples
df = df.fillna(df.mean(numeric_only=True))

X_full = df.iloc[:, :-3].values  
y_full = df.iloc[:, -3:].values 

# 4. Define Experiment Parameters
sample_sizes = [500, 1000, 2000, 2884]
target_names = ['Nitrate', 'Kjeldahl nitrogen', 'Total Phosphorus, mixed forms']

actual_rmse_data = {target: [] for target in target_names}
actual_r2_data = {target: [] for target in target_names}

# 5. The Training Loop
for size in sample_sizes:
    print(f"--- Training with Sample Size: {size} ---")
    
    # Slice the data to the current size limit
    X_subset = X_full[:size]
    y_subset = y_full[:size]
    
    # 80/20 Split
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    
    # FIX: Initialize using the exact dimension arguments your class requires
    model = AgriculturalRunoffNN(input_dim=X_train.shape[1], output_dim=3).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for 300 epochs
    model.train()
    for epoch in range(300):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        
        # FIX: Prevent gradients from exploding into NaNs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
    # Evaluate
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_t).cpu().numpy()
        
    # Log the metrics
    for i, target in enumerate(target_names):
        rmse = np.sqrt(mean_squared_error(y_test[:, i], predictions[:, i]))
        r2 = r2_score(y_test[:, i], predictions[:, i])
        
        actual_rmse_data[target].append(round(rmse, 4))
        actual_r2_data[target].append(round(r2, 4))
    
    print(f"Finished size {size} successfully.\n")

# 6. Final Output for the Graphing Script
print("==============================================")
print(" COPY THESE DICTIONARIES FOR THE GRAPH SCRIPT ")
print("==============================================")
print(f"rmse_data = {actual_rmse_data}")
print(f"r2_data = {actual_r2_data}")