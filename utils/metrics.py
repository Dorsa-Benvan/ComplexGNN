"""
Utility functions for metrics calculation
"""

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def calculate_leverage(X: np.ndarray) -> np.ndarray:
    """Compute the hat-matrix diagonal (leverage)"""
    H = X @ np.linalg.pinv(X)
    return np.clip(np.diag(H), 0.0, 1.0)


def save_metrics_to_file(metrics_dict, filename='metrics_results.txt'):
    """Save metrics to a text file"""
    with open(filename, 'w') as f:
        f.write("Model Evaluation Metrics\n")
        f.write("=" * 40 + "\n\n")
        
        for dataset_name, metrics in metrics_dict.items():
            f.write(f"{dataset_name} Set:\n")
            f.write(f"  R²  = {metrics['R2']:.4f}\n")
            f.write(f"  MAE = {metrics['MAE']:.4f}\n")
            f.write(f"  MSE = {metrics['MSE']:.4f}\n")
            f.write(f"  RMSE = {metrics.get('RMSE', 0):.4f}\n\n")
    
    print(f"✅ Saved metrics to {filename}")