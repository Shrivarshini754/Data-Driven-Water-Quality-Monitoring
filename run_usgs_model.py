
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load the real-world USGS dataset
print("Loading real-world USGS dataset...")
df = pd.read_csv('usgs_ready_for_pytorch.csv')

# 2. Define your target pollutant columns based on your new USGS columns
# Make sure these match your exact CSV headers from the dataset preview
target_columns = [
    'Nitrate',  
    'Kjeldahl nitrogen', 
    'Total Phosphorus, mixed forms'
]

# Separate Features (X) and Targets (y)
X = df.drop(columns=target_columns)
y = df[target_columns]

# 3. Handle any remaining missing values using column means
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# 4. Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scale both features (X) and targets (y) for stable neural network performance
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

# 6. Define the Custom Multi-Target Neural Network Architecture
class AgriculturalRunoffNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AgriculturalRunoffNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # Regularization to prevent overfitting
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim) # Outputs all target pollutants simultaneously
        )
        
    def forward(self, x):
        return self.network(x)

input_dim = X_train_t.shape[1]
output_dim = y_train_t.shape[1]
model = AgriculturalRunoffNN(input_dim, output_dim)

# 7. Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 8. Training Loop
epochs = 300
print("\nTraining PyTorch Model on Real-World USGS Data...")
for epoch in range(epochs):
    model.train()
    
    # Forward pass
    predictions = model(X_train_t)
    loss = criterion(predictions, y_train_t)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}")

# 9. Evaluation & Inverse Transformation
model.eval()
with torch.no_grad():
    # Predict on Test set and transform back to original units
    test_preds_scaled = model(X_test_t).numpy()
    test_preds = scaler_y.inverse_transform(test_preds_scaled)
    y_test_np = y_test.values

print("\n==============================================")
print("   REAL-WORLD USGS MODEL PERFORMANCE (R2)     ")
print("==============================================")
for i, col in enumerate(target_columns):
    rmse = np.sqrt(mean_squared_error(y_test_np[:, i], test_preds[:, i]))
    r2 = r2_score(y_test_np[:, i], test_preds[:, i])
    
    print(f"Pollutant: {col}")
    print(f"  -> RMSE: {rmse:.4f}")
    print(f"  -> R2 Score (Accuracy): {r2:.4f}")
    print("-" * 45)