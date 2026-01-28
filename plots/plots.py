"""
Complete visualization functions from original script
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.linear_model import LinearRegression
import itertools
import os


def plot_predicted_vs_true(y_true, y_pred, dataset_type, save_dir="plots"):
    """Plot predicted vs true values with confidence intervals"""
    reg = LinearRegression().fit(y_true.reshape(-1, 1), y_pred)
    y_fit = reg.predict(y_true.reshape(-1, 1)).ravel()
    residuals = y_pred - y_fit
    n = len(y_true)
    std_res = np.std(residuals)
    se = std_res * np.sqrt(
        1 + 1/n + ((y_true - y_true.mean())**2) / np.sum((y_true - y_true.mean())**2)
    )
    t_val = stats.t.ppf(0.975, n-2)
    upper = y_fit + t_val * se
    lower = y_fit - t_val * se

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, label='Data')
    plt.plot(y_true, y_fit, 'r-', lw=2, label='Fit')
    r2 = r2_score(y_true, y_pred)
    
    # Get the trendline equation
    slope = reg.coef_[0]
    intercept = reg.intercept_
    trendline_formula = f'y = {slope:.2f}x + {intercept:.2f}'

    # Add R^2 and trendline formula
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    plt.text(0.05, 0.88, trendline_formula, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

    plt.fill_between(y_true, lower, upper, color='gray', alpha=0.4, label='95% CI')
    plt.title(f'Predicted vs True ({dataset_type})')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/predicted_vs_true_{dataset_type}.svg', format='svg')
    plt.show()


def plot_pca(X, labels, title='PCA of Feature Matrix', save_path='pca_plot.svg'):
    """Plot PCA of feature matrix"""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df['Split'] = labels

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='Split', palette='Set2', s=60, alpha=0.7)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, format='svg')
    plt.show()


def plot_residual_distribution(y_true, y_pred, dataset_name, save_dir="plots"):
    """Plot residual distribution with Gaussian fit"""
    from scipy.stats import norm
    
    residuals = y_pred - y_true
    mu, std = norm.fit(residuals)

    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=False, bins=40, stat='density', 
                 color='skyblue', label='Residuals Histogram')

    # Plot fitted Gaussian curve
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r--', linewidth=2, 
             label=f'Normal Fit\n$\mu$={mu:.3f}, $\sigma$={std:.3f}')

    plt.title(f'Residual Distribution - {dataset_name} Set')
    plt.xlabel('Residual (Predicted - True)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/residual_distribution_{dataset_name.lower()}.svg", 
                format='svg', dpi=300)
    plt.show()


# Add more plotting functions as needed...