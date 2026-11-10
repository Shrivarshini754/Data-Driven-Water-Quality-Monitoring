
import pandas as pd
import numpy as np
import torch

# 1. Configure Hardware Acceleration
# This ensures PyTorch utilizes Apple's Metal Performance Shaders (MPS) for faster training
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Active processing device: {device}")

# 2. Import Your Existing PyTorch Model
# TODO: Replace 'your_model_file' with the name of the python file containing your model
# TODO: Replace 'YourCustomMLP' with the actual name of your model class
from run_usgs_model import AgriculturalRunoffNN

# 3. Load the Preprocessed USGS Dataset
# TODO: Replace 'usgs_preprocessed_data.csv' with your actual dataset file name
df = pd.read_csv('usgs_ready_for_pytorch.csv')

# 4. Separate Inputs (Features) and Targets (Labels)
# This assumes your targets (Nitrogen, Phosphorus, Nitrate) are the last 3 columns
X_full = df.iloc[:, :-3].values  
y_full = df.iloc[:, -3:].values  

print(f"Total dataset loaded successfully.")
print(f"Shape of physical sensor inputs (X): {X_full.shape}")
print(f"Shape of chemical targets (y): {y_full.shape}")