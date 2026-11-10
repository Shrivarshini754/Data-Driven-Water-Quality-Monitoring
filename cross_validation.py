import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from run_usgs_model import AgriculturalRunoffNN

def main():
    print("Loading data...")
    df = pd.read_csv('usgs_ready_for_pytorch.csv')
    df = df.fillna(df.mean(numeric_only=True))
    X = df.iloc[:, :-3].values
    y = df.iloc[:, -3:].values
    target_names = ['Nitrate', 'Kjeldahl Nitrogen', 'Total Phosphorus']

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    mlp_rmse, mlp_r2 = {t: [] for t in target_names}, {t: [] for t in target_names}
    rf_rmse, rf_r2 = {t: [] for t in target_names}, {t: [] for t in target_names}

    print("\nRunning Optimized Bounded 5-Fold Cross Validation...")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"  -> Processing Fold {fold + 1}/5...")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale Features robustly to ignore input outliers
        x_scaler = RobustScaler()
        X_train_s = x_scaler.fit_transform(X_train)
        X_test_s = x_scaler.transform(X_test)
        
        # Scale targets using StandardScaler so the mean is explicitly 0 (aligns with R2)
        y_scaler = StandardScaler()
        y_train_s = y_scaler.fit_transform(y_train)
        
        # -----------------------------------------
        # Train Random Forest (Baseline)
        # -----------------------------------------
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train_s, y_train)
        rf_preds = rf.predict(X_test_s)
        
        for i, t in enumerate(target_names):
            rf_rmse[t].append(np.sqrt(mean_squared_error(y_test[:, i], rf_preds[:, i])))
            rf_r2[t].append(r2_score(y_test[:, i], rf_preds[:, i]))

        # -----------------------------------------
        # Train PyTorch MLP
        # -----------------------------------------
        X_train_t = torch.FloatTensor(X_train_s).to(device)
        X_test_t = torch.FloatTensor(X_test_s).to(device)
        y_train_t = torch.FloatTensor(y_train_s).to(device)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        model = AgriculturalRunoffNN(input_dim=X_train.shape[1], output_dim=3).to(device)
        
        # Use SmoothL1Loss to gracefully handle variance without destroying Kjeldahl
        criterion = nn.SmoothL1Loss() 
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
        
        model.train()
        # 150 Epochs to allow Kjeldahl enough time to map complex patterns
        for epoch in range(150): 
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
        model.eval()
        with torch.no_grad():
            mlp_preds_scaled = model(X_test_t).cpu().numpy()
            
        # THE CLAMP: Find the absolute min and max of the training targets
        y_min = y_train_s.min(axis=0)
        y_max = y_train_s.max(axis=0)
        
        # Force predictions to stay within known physical limits, protecting Nitrate R2
        mlp_preds_scaled = np.clip(mlp_preds_scaled, y_min, y_max)
            
        # Inverse transform back to real mg/L
        mlp_preds = y_scaler.inverse_transform(mlp_preds_scaled)
            
        for i, t in enumerate(target_names):
            mlp_rmse[t].append(np.sqrt(mean_squared_error(y_test[:, i], mlp_preds[:, i])))
            mlp_r2[t].append(r2_score(y_test[:, i], mlp_preds[:, i]))

    print("\n=============================================")
    print("      5-FOLD CV RESULTS (Mean ± SD)          ")
    print("=============================================")
    
    with open("cv_results_summary.txt", "w") as f:
        f.write("5-FOLD CV RESULTS (Mean ± SD)\n")
        f.write("=============================================\n")
        
        for t in target_names:
            result_str = (
                f"\nTarget: {t}\n"
                f"  MLP -> RMSE: {np.mean(mlp_rmse[t]):.3f} ± {np.std(mlp_rmse[t]):.3f} | R²: {np.mean(mlp_r2[t]):.3f} ± {np.std(mlp_r2[t]):.3f}\n"
                f"  RF  -> RMSE: {np.mean(rf_rmse[t]):.3f} ± {np.std(rf_rmse[t]):.3f} | R²: {np.mean(rf_r2[t]):.3f} ± {np.std(rf_r2[t]):.3f}\n"
            )
            print(result_str.strip())
            f.write(result_str)

if __name__ == "__main__":
    main()