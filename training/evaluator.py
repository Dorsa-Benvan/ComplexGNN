"""
Model evaluation and metrics calculation
"""

import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt


def calculate_metrics(y_true, y_pred, dataset_type=""):
    """Calculate and print evaluation metrics"""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    if dataset_type:
        print(f"\nMetrics for {dataset_type} dataset:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  Mean Absolute Error (MAE): {mae:.4f}")
        print(f"  Mean Squared Error (MSE): {mse:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    return {
        'R2': r2,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    }


def evaluate_and_plot(model, loader, device, dataset_type, y_scaler=None):
    """Evaluate model and create prediction plots"""
    model.eval()
    y_true_list, y_pred_list = [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            y_true_list.append(data.y.cpu().numpy().ravel())
            y_pred_list.append(out.cpu().numpy().ravel())
    
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    
    # Inverse scaling if scaler provided
    if y_scaler is not None:
        y_true = y_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, dataset_type)
    
    return y_true, y_pred, metrics