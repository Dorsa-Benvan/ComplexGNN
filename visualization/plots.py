"""
Complete visualization module for DualcomplexGNN
Includes all plotting functions from the original script
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import itertools
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy import stats
from scipy.spatial import distance
import umap
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from scipy.cluster import hierarchy as sch
from collections import defaultdict
import csv

# -----------------------------------------------------------------------------
# 1. PAIRWISE SCATTER PLOTS
# -----------------------------------------------------------------------------

def plot_pairwise_scatter_features(features_array, feature_names, ids, output_dir="paired_features_plots"):
    """
    Generate pairwise scatter plots for all combinations of features
    
    Args:
        features_array: numpy array of shape (n_samples, n_features)
        feature_names: list of feature names
        ids: sample IDs for indexing
        output_dir: directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame
    df_feats = pd.DataFrame(features_array, index=ids, columns=feature_names)
    
    # Generate all pairwise combinations
    n_plots = 0
    for x_feat, y_feat in itertools.combinations(feature_names, 2):
        plt.figure(figsize=(6, 6))
        plt.scatter(df_feats[x_feat], df_feats[y_feat], alpha=0.6, s=30)
        plt.xlabel(x_feat)
        plt.ylabel(y_feat)
        plt.title(f"{x_feat} vs {y_feat}")
        plt.tight_layout()
        
        # Save plot
        plot_filename = os.path.join(output_dir, f"{x_feat}_vs_{y_feat}.svg")
        plt.savefig(plot_filename, format="svg", dpi=300)
        plt.close()
        n_plots += 1
        
        if n_plots % 50 == 0:
            print(f"  Generated {n_plots} plots...")
    
    print(f"✅ Generated {n_plots} pairwise scatter plots in '{output_dir}/'")
    return df_feats

# -----------------------------------------------------------------------------
# 2. PREDICTION PLOTS
# -----------------------------------------------------------------------------

def plot_predicted_vs_true(y_true, y_pred, dataset_type="", 
                          show_ci=True, save_dir="plots", **kwargs):
    """
    Plot predicted vs true values with regression line and confidence intervals
    
    Args:
        y_true: true values
        y_pred: predicted values
        dataset_type: name of dataset for title
        show_ci: whether to show confidence interval
        save_dir: directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Fit linear regression
    reg = LinearRegression().fit(y_true.reshape(-1, 1), y_pred)
    y_fit = reg.predict(y_true.reshape(-1, 1)).ravel()
    
    # Calculate R²
    r2 = r2_score(y_true, y_pred)
    
    # Calculate confidence intervals if requested
    if show_ci:
        residuals = y_pred - y_fit
        n = len(y_true)
        std_res = np.std(residuals)
        se = std_res * np.sqrt(
            1 + 1/n + ((y_true - y_true.mean())**2) / np.sum((y_true - y_true.mean())**2)
        )
        t_val = stats.t.ppf(0.975, n-2)
        upper = y_fit + t_val * se
        lower = y_fit - t_val * se
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    plt.plot(y_true, y_fit, 'r-', lw=2, label='Linear fit')
    
    if show_ci:
        plt.fill_between(y_true, lower, upper, color='gray', alpha=0.3, label='95% CI')
    
    # Add R² and regression equation
    slope = reg.coef_[0]
    intercept = reg.intercept_
    eq_text = f'y = {slope:.3f}x + {intercept:.3f}'
    
    plt.text(0.05, 0.95, f'R² = {r2:.4f}\n{eq_text}', 
             transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'Predicted vs True - {dataset_type}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    filename = f'pred_vs_true_{dataset_type.lower()}.svg'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Saved prediction plot: {save_path}")
    return r2, slope, intercept

# -----------------------------------------------------------------------------
# 3. WILLIAMS PLOT
# -----------------------------------------------------------------------------

def plot_williams(y_true, X, dataset_type="", save_dir="plots"):
    """
    Generate Williams plot (leverage vs standardized residuals)
    
    Args:
        y_true: true target values
        X: feature matrix used for predictions
        dataset_type: name of dataset
        save_dir: directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Add constant for OLS
    X_with_const = sm.add_constant(X)
    
    # Fit OLS model
    model = sm.OLS(y_true, X_with_const).fit()
    influence = model.get_influence()
    
    # Get standardized residuals and leverage
    standardized_residuals = influence.resid_studentized_internal
    leverages = influence.hat_matrix_diag
    
    # Calculate critical leverage threshold
    leverage_threshold = 3 * X_with_const.shape[1] / X_with_const.shape[0]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(leverages, standardized_residuals, alpha=0.7,
                        edgecolor='k', linewidth=0.5, s=60)
    
    # Add threshold lines
    ax.axhline(y=3, color='r', linestyle='--', label='Residual Threshold (±3)')
    ax.axhline(y=-3, color='r', linestyle='--')
    ax.axvline(x=leverage_threshold, color='g', linestyle='--',
               label=f'Leverage Threshold (h*={leverage_threshold:.3f})')
    
    # Label outliers
    outlier_mask = (np.abs(standardized_residuals) > 3) | (leverages > leverage_threshold)
    if np.any(outlier_mask):
        outlier_indices = np.where(outlier_mask)[0]
        ax.scatter(leverages[outlier_mask], standardized_residuals[outlier_mask],
                  color='red', s=80, edgecolor='k', linewidth=1.5,
                  label=f'Outliers ({len(outlier_indices)})')
    
    ax.set_xlabel('Leverage (Hat Values)', fontsize=12)
    ax.set_ylabel('Standardized Residuals (Studentized)', fontsize=12)
    ax.set_title(f'Williams Plot - {dataset_type}', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    filename = f'williams_plot_{dataset_type.lower()}.svg'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Saved Williams plot: {save_path}")
    return standardized_residuals, leverages, outlier_mask

# -----------------------------------------------------------------------------
# 4. Q vs T² PLOT
# -----------------------------------------------------------------------------

def plot_q_t2(X, dataset_name="Dataset", variance_threshold=0.98, save_dir="plots"):
    """
    Generate Q vs T² plot for PCA analysis
    
    Args:
        X: data matrix (samples × features)
        dataset_name: name of dataset
        variance_threshold: variance to explain
        save_dir: directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA on all components
    pca_full = PCA().fit(X_scaled)
    explained_var_ratio = pca_full.explained_variance_ratio_
    cumulative_var_ratio = np.cumsum(explained_var_ratio)
    
    # Select number of components to explain variance threshold
    n_components = np.searchsorted(cumulative_var_ratio, variance_threshold) + 1
    print(f"🔹 Selected {n_components} components to explain "
          f"{cumulative_var_ratio[n_components-1]:.2%} variance.")
    
    # Fit PCA with selected components
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)
    X_recon_scaled = pca.inverse_transform(scores)
    X_recon = scaler.inverse_transform(X_recon_scaled)
    
    # Compute Q residuals in original space
    residuals = X - X_recon
    Q = np.sum(residuals**2, axis=1)
    
    # Compute Hotelling's T²
    eigenvalues = pca.explained_variance_
    T_squared = np.sum((scores**2) / eigenvalues, axis=1)
    
    # Variance explained in original data space
    total_variance = np.sum(np.var(X, axis=0, ddof=1))
    reconstructed_variance = np.sum(np.var(X_recon, axis=0, ddof=1))
    explained = (reconstructed_variance / total_variance) * 100
    residual = 100 - explained
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(T_squared, Q, alpha=0.7, edgecolor='k', linewidth=0.5, s=50)
    
    # Add threshold lines
    plt.axhline(np.percentile(Q, 98), color='r', linestyle='--', 
                label=f'98% Q limit ({np.percentile(Q, 98):.2f})')
    plt.axvline(np.percentile(T_squared, 98), color='g', linestyle='--', 
                label=f'98% T² limit ({np.percentile(T_squared, 98):.2f})')
    
    plt.xlabel(f"T² (captures {explained:.1f}% variance)", fontsize=12)
    plt.ylabel(f"Q (missed {residual:.1f}% variance)", fontsize=12)
    plt.title(f"Q vs T² Plot - {dataset_name} ({n_components} PCs)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    filename = f'q_vs_t2_{dataset_name.lower()}.svg'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Saved Q vs T² plot: {save_path}")
    return Q, T_squared, n_components, explained

# -----------------------------------------------------------------------------
# 5. RESIDUAL DISTRIBUTION PLOTS
# -----------------------------------------------------------------------------

def plot_residual_distribution(y_true, y_pred, dataset_name="", save_dir="plots"):
    """
    Plot residual distribution with Gaussian fit
    
    Args:
        y_true: true values
        y_pred: predicted values
        dataset_name: name of dataset
        save_dir: directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    residuals = y_pred - y_true
    mu, std = stats.norm.fit(residuals)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram with KDE
    sns.histplot(residuals, kde=True, ax=ax1, bins=30, 
                 color='skyblue', edgecolor='black', linewidth=0.5)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Residual (Predicted - True)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title(f'Residual Distribution - {dataset_name}', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add text box with statistics
    textstr = f'$\mu$ = {mu:.3f}\n$\sigma$ = {std:.3f}\nN = {len(residuals)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title(f'Q-Q Plot - {dataset_name}', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f'residual_distribution_{dataset_name.lower()}.svg'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Saved residual distribution plot: {save_path}")
    return residuals, mu, std

# -----------------------------------------------------------------------------
# 6. PCA VISUALIZATION
# -----------------------------------------------------------------------------

def plot_pca_2d(X, labels=None, title="PCA - 2D Projection", 
                color_by=None, cmap='viridis', save_dir="plots"):
    """
    2D PCA plot
    
    Args:
        X: data matrix
        labels: cluster labels or other categorical labels
        title: plot title
        color_by: continuous values for coloring
        cmap: colormap
        save_dir: directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Calculate explained variance
    var_exp = pca.explained_variance_ratio_ * 100
    
    # Create plot
    plt.figure(figsize=(10, 7))
    
    if labels is not None and color_by is None:
        # Color by categorical labels
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       color=colors[i], label=f'Cluster {label}', 
                       alpha=0.7, s=50, edgecolors='k', linewidth=0.5)
        plt.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
        
    elif color_by is not None:
        # Color by continuous values
        sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                        c=color_by, cmap=cmap, 
                        alpha=0.7, s=50, edgecolors='k', linewidth=0.5)
        plt.colorbar(sc, label='Value')
        
    else:
        # No coloring
        plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                   alpha=0.7, s=50, edgecolors='k', linewidth=0.5)
    
    plt.xlabel(f'PC1 ({var_exp[0]:.1f}% variance)', fontsize=12)
    plt.ylabel(f'PC2 ({var_exp[1]:.1f}% variance)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    filename = 'pca_2d.svg'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Saved 2D PCA plot: {save_path}")
    return X_pca, var_exp

def plot_pca_3d(X, labels=None, title="PCA - 3D Projection", 
                color_by=None, cmap='viridis', save_dir="plots"):
    """
    3D PCA plot
    
    Args:
        X: data matrix
        labels: cluster labels
        title: plot title
        color_by: continuous values for coloring
        cmap: colormap
        save_dir: directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Perform PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    # Calculate explained variance
    var_exp = pca.explained_variance_ratio_ * 100
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    if labels is not None and color_by is None:
        # Color by categorical labels
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                      color=colors[i], label=f'Cluster {label}', 
                      alpha=0.7, s=30, edgecolors='k', linewidth=0.3)
        ax.legend(title='Clusters', bbox_to_anchor=(1.15, 1), loc='upper left')
        
    elif color_by is not None:
        # Color by continuous values
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                       c=color_by, cmap=cmap, 
                       alpha=0.7, s=30, edgecolors='k', linewidth=0.3)
        plt.colorbar(sc, ax=ax, label='Value', pad=0.1)
        
    else:
        # No coloring
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                  alpha=0.7, s=30, edgecolors='k', linewidth=0.3)
    
    ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}% variance)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}% variance)', fontsize=11)
    ax.set_zlabel(f'PC3 ({var_exp[2]:.1f}% variance)', fontsize=11)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    # Save plot
    filename = 'pca_3d.svg'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Saved 3D PCA plot: {save_path}")
    return X_pca, var_exp

def plot_pca_explained_variance(X, save_dir="plots"):
    """
    Plot PCA explained variance (scree plot)
    
    Args:
        X: data matrix
        save_dir: directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Perform PCA on all components
    pca = PCA()
    pca.fit(X)
    
    # Calculate explained variance
    explained_var = pca.explained_variance_ratio_ * 100
    cumulative_var = np.cumsum(explained_var)
    
    # Create scree plot
    plt.figure(figsize=(12, 5))
    
    # Individual explained variance
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7)
    plt.axhline(y=5, color='r', linestyle='--', alpha=0.7, label='5% threshold')
    plt.xlabel('Principal Component', fontsize=11)
    plt.ylabel('Explained Variance (%)', fontsize=11)
    plt.title('Scree Plot - Individual Variance', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Cumulative explained variance
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'o-', linewidth=2)
    plt.axhline(y=95, color='r', linestyle='--', alpha=0.7, label='95% threshold')
    plt.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='80% threshold')
    plt.xlabel('Number of Principal Components', fontsize=11)
    plt.ylabel('Cumulative Explained Variance (%)', fontsize=11)
    plt.title('Cumulative Explained Variance', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = 'pca_explained_variance.svg'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Find number of components for 95% variance
    n_components_95 = np.argmax(cumulative_var >= 95) + 1
    n_components_80 = np.argmax(cumulative_var >= 80) + 1
    
    print(f"✅ Saved PCA explained variance plot: {save_path}")
    print(f"🔹 Components for 80% variance: {n_components_80}")
    print(f"🔹 Components for 95% variance: {n_components_95}")
    
    return explained_var, cumulative_var

# -----------------------------------------------------------------------------
# 7. CLUSTERING VISUALIZATION
# -----------------------------------------------------------------------------

def plot_elbow_method(X, max_k=50, save_dir="plots"):
    """
    Plot elbow method for KMeans clustering
    
    Args:
        X: data matrix
        max_k: maximum number of clusters to test
        save_dir: directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    inertias = []
    K_range = range(1, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    # Calculate second derivative to find elbow
    diff1 = np.diff(inertias)
    diff2 = np.diff(diff1)
    elbow_idx = np.argmax(np.abs(diff2)) + 2 if len(diff2) > 0 else 3
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=6)
    plt.axvline(x=elbow_idx, color='r', linestyle='--', 
                label=f'Elbow at k={elbow_idx}')
    
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia', fontsize=12)
    plt.title('Elbow Method for Optimal k', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    filename = 'elbow_method.svg'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Saved elbow method plot: {save_path}")
    print(f"🔹 Suggested k (elbow point): {elbow_idx}")
    
    return inertias, elbow_idx

def plot_dendrogram(X, method='ward', save_dir="plots"):
    """
    Plot hierarchical clustering dendrogram
    
    Args:
        X: data matrix
        method: linkage method
        save_dir: directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate linkage matrix
    Z = sch.linkage(X, method=method)
    
    # Create dendrogram
    plt.figure(figsize=(12, 6))
    sch.dendrogram(Z, truncate_mode='lastp', p=30, 
                   show_leaf_counts=True, leaf_rotation=90)
    
    plt.xlabel('Sample index or (cluster size)', fontsize=11)
    plt.ylabel('Distance', fontsize=11)
    plt.title(f'Hierarchical Clustering Dendrogram ({method} linkage)', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save plot
    filename = f'dendrogram_{method}.svg'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Saved dendrogram: {save_path}")
    return Z

# -----------------------------------------------------------------------------
# 8. t-SNE VISUALIZATION
# -----------------------------------------------------------------------------

def plot_tsne_2d(X, labels=None, perplexity=30, title="t-SNE - 2D Projection",
                 with_ellipses=False, save_dir="plots"):
    """
    2D t-SNE visualization
    
    Args:
        X: data matrix
        labels: cluster labels
        perplexity: t-SNE perplexity
        title: plot title
        with_ellipses: whether to draw ellipses around clusters
        save_dir: directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, 
                random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    if labels is not None:
        # Color by cluster labels
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            cluster_points = X_tsne[mask]
            
            # Plot points
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                       color=colors[i], label=f'Cluster {label}',
                       alpha=0.7, s=50, edgecolors='k', linewidth=0.5)
            
            # Plot centroid
            centroid = cluster_points.mean(axis=0)
            plt.scatter(centroid[0], centroid[1], color=colors[i],
                       s=200, marker='*', edgecolors='k', linewidth=1.5)
            
            # Add cluster number text
            plt.text(centroid[0], centroid[1], f'{label}', 
                    fontsize=12, fontweight='bold',
                    ha='center', va='center', color='white')
            
            # Draw ellipses if requested
            if with_ellipses and len(cluster_points) > 2:
                # Calculate covariance matrix
                cov = np.cov(cluster_points.T)
                if cov.shape == (2, 2):
                    # Calculate eigenvalues and eigenvectors
                    vals, vecs = np.linalg.eigh(cov)
                    order = vals.argsort()[::-1]
                    vals, vecs = vals[order], vecs[:, order]
                    
                    # Calculate ellipse parameters
                    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                    width, height = 2 * np.sqrt(vals * 5.991)  # 95% confidence
                    
                    # Draw ellipse
                    ellipse = Ellipse(xy=centroid, width=width, height=height,
                                     angle=angle, edgecolor=colors[i],
                                     facecolor='none', linewidth=2, alpha=0.7)
                    plt.gca().add_patch(ellipse)
                    
                    # Draw spider lines from centroid to points
                    for point in cluster_points[:10]:  # Limit to first 10 points
                        plt.plot([centroid[0], point[0]], [centroid[1], point[1]],
                                color=colors[i], alpha=0.3, linewidth=0.8)
        
        plt.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # No labels
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7, s=50, 
                   edgecolors='k', linewidth=0.5)
    
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    suffix = "_with_ellipses" if with_ellipses else ""
    filename = f'tsne_2d{suffix}.svg'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Saved 2D t-SNE plot: {save_path}")
    return X_tsne

def plot_tsne_3d(X, labels=None, perplexity=30, title="t-SNE - 3D Projection",
                 with_spiders=False, save_dir="plots"):
    """
    3D t-SNE visualization
    
    Args:
        X: data matrix
        labels: cluster labels
        perplexity: t-SNE perplexity
        title: plot title
        with_spiders: whether to draw spider lines
        save_dir: directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Perform t-SNE
    tsne = TSNE(n_components=3, perplexity=perplexity,
                random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X)
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if labels is not None:
        # Color by cluster labels
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            cluster_points = X_tsne[mask]
            
            # Plot points
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                      color=colors[i], label=f'Cluster {label}',
                      alpha=0.7, s=30, edgecolors='k', linewidth=0.3)
            
            # Plot centroid
            centroid = cluster_points.mean(axis=0)
            ax.scatter(centroid[0], centroid[1], centroid[2], color=colors[i],
                      s=200, marker='*', edgecolors='k', linewidth=1.5)
            
            # Draw spider lines if requested
            if with_spiders:
                for point in cluster_points[:8]:  # Limit to first 8 points
                    ax.plot([centroid[0], point[0]], 
                           [centroid[1], point[1]],
                           [centroid[2], point[2]],
                           color=colors[i], alpha=0.3, linewidth=0.8)
        
        ax.legend(title='Clusters', bbox_to_anchor=(1.15, 1), loc='upper left')
    else:
        # No labels
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2],
                  alpha=0.7, s=30, edgecolors='k', linewidth=0.3)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.set_zlabel('t-SNE Dimension 3', fontsize=11)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    # Save plot
    suffix = "_with_spiders" if with_spiders else ""
    filename = f'tsne_3d{suffix}.svg'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Saved 3D t-SNE plot: {save_path}")
    return X_tsne

# -----------------------------------------------------------------------------
# 9. UMAP VISUALIZATION
# -----------------------------------------------------------------------------

def plot_umap_2d(X, labels=None, n_neighbors=15, min_dist=0.1,
                 title="UMAP - 2D Projection", save_dir="plots"):
    """
    2D UMAP visualization
    
    Args:
        X: data matrix
        labels: cluster labels
        n_neighbors: UMAP parameter
        min_dist: UMAP parameter
        title: plot title
        save_dir: directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Perform UMAP
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                       min_dist=min_dist, random_state=42)
    X_umap = reducer.fit_transform(X)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    if labels is not None:
        # Color by cluster labels
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(X_umap[mask, 0], X_umap[mask, 1],
                       color=colors[i], label=f'Cluster {label}',
                       alpha=0.7, s=50, edgecolors='k', linewidth=0.5)
        
        plt.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # No labels
        plt.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.7, s=50,
                   edgecolors='k', linewidth=0.5)
    
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    filename = 'umap_2d.svg'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Saved 2D UMAP plot: {save_path}")
    return X_umap

def plot_umap_3d(X, labels=None, n_neighbors=15, min_dist=0.1,
                 title="UMAP - 3D Projection", save_dir="plots"):
    """
    3D UMAP visualization
    
    Args:
        X: data matrix
        labels: cluster labels
        n_neighbors: UMAP parameter
        min_dist: UMAP parameter
        title: plot title
        save_dir: directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Perform UMAP
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors,
                       min_dist=min_dist, random_state=42)
    X_umap = reducer.fit_transform(X)
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if labels is not None:
        # Color by cluster labels
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1], X_umap[mask, 2],
                      color=colors[i], label=f'Cluster {label}',
                      alpha=0.7, s=30, edgecolors='k', linewidth=0.3)
        
        ax.legend(title='Clusters', bbox_to_anchor=(1.15, 1), loc='upper left')
    else:
        # No labels
        ax.scatter(X_umap[:, 0], X_umap[:, 1], X_umap[:, 2],
                  alpha=0.7, s=30, edgecolors='k', linewidth=0.3)
    
    ax.set_xlabel('UMAP Dimension 1', fontsize=11)
    ax.set_ylabel('UMAP Dimension 2', fontsize=11)
    ax.set_zlabel('UMAP Dimension 3', fontsize=11)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    # Save plot
    filename = 'umap_3d.svg'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Saved 3D UMAP plot: {save_path}")
    return X_umap

# -----------------------------------------------------------------------------
# 10. VIOLIN PLOTS
# -----------------------------------------------------------------------------

def plot_violin_per_cluster(df, features, cluster_col='cluster', 
                           save_dir="plots/violin_plots"):
    """
    Generate violin plots for each feature grouped by clusters
    
    Args:
        df: DataFrame with features and cluster labels
        features: list of feature names to plot
        cluster_col: column name for cluster labels
        save_dir: directory to save plots
    """
    import re
    
    os.makedirs(save_dir, exist_ok=True)
    
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    for feature in features:
        plt.figure(figsize=(10, 6))
        
        # Get unique clusters and sort them
        clusters = sorted(df[cluster_col].unique())
        
        # Prepare data for violin plot
        plot_data = []
        cluster_labels = []
        
        for cluster in clusters:
            cluster_data = df[df[cluster_col] == cluster][feature].dropna()
            if len(cluster_data) > 0:
                plot_data.append(cluster_data)
                cluster_labels.append(f'Cluster {cluster}')
        
        if plot_data:
            # Create violin plot
            parts = plt.violinplot(plot_data, showmeans=False, showmedians=True)
            
            # Customize colors
            colors = plt.cm.tab20(np.linspace(0, 1, len(plot_data)))
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_edgecolor('black')
                pc.set_alpha(0.7)
            
            # Customize other elements
            parts['cmedians'].set_color('red')
            parts['cmedians'].set_linewidth(2)
            
            # Set x-ticks
            plt.xticks(range(1, len(cluster_labels) + 1), cluster_labels, rotation=45)
            
            plt.ylabel(feature, fontsize=12)
            plt.title(f'{feature} Distribution per Cluster', fontsize=14)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            # Sanitize filename
            safe_feature = re.sub(r'[\/:*?"<>|()\s]', '_', feature)
            
            # Save plot
            plot_path = os.path.join(save_dir, f'{safe_feature}_violin_plot.svg')
            plt.savefig(plot_path, format='svg', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ Saved violin plot for {feature}")
        else:
            plt.close()
            print(f"  ⚠️ No data for {feature}")
    
    print(f"✅ All violin plots saved to {save_dir}")

# -----------------------------------------------------------------------------
# 11. HEATMAPS
# -----------------------------------------------------------------------------

def plot_cluster_feature_heatmap(df, features, cluster_col='cluster', 
                                save_dir="plots"):
    """
    Create heatmap of z-scored feature means per cluster
    
    Args:
        df: DataFrame with features and cluster labels
        features: list of feature names
        cluster_col: column name for cluster labels
        save_dir: directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Group by cluster and compute mean
    cluster_means = df.groupby(cluster_col)[features].mean()
    
    # Standardize (z-score) across clusters for each feature
    scaler = StandardScaler()
    z_scores = pd.DataFrame(
        scaler.fit_transform(cluster_means),
        index=cluster_means.index,
        columns=features
    )
    
    # Save z-scores to CSV
    z_scores_csv_path = os.path.join(save_dir, 'cluster_feature_zscores.csv')
    z_scores.to_csv(z_scores_csv_path)
    
    # Create heatmap
    plt.figure(figsize=(max(12, len(features) * 0.8), max(8, len(cluster_means) * 0.6)))
    ax = sns.heatmap(
        z_scores.T,  # Transpose to have features as rows
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        cbar_kws={'label': 'z-score (cluster mean)'},
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_title("Cluster × Feature Heatmap (z-scored means)", fontsize=14, pad=20)
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save heatmap
    heatmap_path = os.path.join(save_dir, 'cluster_feature_zscore_heatmap.svg')
    plt.savefig(heatmap_path, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Saved z-score matrix to {z_scores_csv_path}")
    print(f"✅ Saved heatmap to {heatmap_path}")
    
    return z_scores

# -----------------------------------------------------------------------------
# 12. INTERACTIVE PLOTS (Plotly)
# -----------------------------------------------------------------------------

def create_interactive_3d_pca(X, labels, values=None, title="Interactive 3D PCA", 
                             save_path="interactive_3d_pca.html"):
    """
    Create interactive 3D PCA plot using Plotly
    
    Args:
        X: data matrix
        labels: cluster labels
        values: continuous values for coloring
        title: plot title
        save_path: path to save HTML file
    """
    # Perform PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    # Create DataFrame for plotly
    df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'PC3': X_pca[:, 2],
        'Cluster': labels.astype(str)
    })
    
    if values is not None:
        df['Value'] = values
    
    # Calculate explained variance
    var_exp = pca.explained_variance_ratio_ * 100
    
    # Create interactive plot
    if values is not None:
        fig = px.scatter_3d(
            df, x='PC1', y='PC2', z='PC3',
            color='Value',
            color_continuous_scale='viridis',
            hover_data=['Cluster'],
            title=f"{title}<br>PC1: {var_exp[0]:.1f}%, PC2: {var_exp[1]:.1f}%, PC3: {var_exp[2]:.1f}%",
            labels={'PC1': f'PC1 ({var_exp[0]:.1f}%)',
                    'PC2': f'PC2 ({var_exp[1]:.1f}%)',
                    'PC3': f'PC3 ({var_exp[2]:.1f}%)'}
        )
    else:
        fig = px.scatter_3d(
            df, x='PC1', y='PC2', z='PC3',
            color='Cluster',
            hover_data=['Cluster'],
            title=f"{title}<br>PC1: {var_exp[0]:.1f}%, PC2: {var_exp[1]:.1f}%, PC3: {var_exp[2]:.1f}%",
            labels={'PC1': f'PC1 ({var_exp[0]:.1f}%)',
                    'PC2': f'PC2 ({var_exp[1]:.1f}%)',
                    'PC3': f'PC3 ({var_exp[2]:.1f}%)'}
        )
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title=f'PC1 ({var_exp[0]:.1f}%)',
            yaxis_title=f'PC2 ({var_exp[1]:.1f}%)',
            zaxis_title=f'PC3 ({var_exp[2]:.1f}%)'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        )
    )
    
    # Save interactive plot
    fig.write_html(save_path)
    print(f"✅ Saved interactive 3D PCA plot: {save_path}")
    
    return fig

def create_interactive_cluster_selector(X, labels, values=None, 
                                       save_path="interactive_cluster_selector.html"):
    """
    Create interactive plot with cluster selector buttons
    
    Args:
        X: data matrix
        labels: cluster labels
        values: continuous values for coloring
        save_path: path to save HTML file
    """
    # Perform PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    # Create DataFrame
    df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'PC3': X_pca[:, 2],
        'Cluster': labels.astype(str)
    })
    
    if values is not None:
        df['Value'] = values
    
    # Calculate explained variance
    var_exp = pca.explained_variance_ratio_ * 100
    
    # Create figure
    fig = go.Figure()
    
    # Get unique clusters
    unique_clusters = sorted(df['Cluster'].unique())
    
    # Add traces for each cluster
    for cluster in unique_clusters:
        cluster_data = df[df['Cluster'] == cluster]
        
        if values is not None:
            # Color by continuous value
            fig.add_trace(go.Scatter3d(
                x=cluster_data['PC1'],
                y=cluster_data['PC2'],
                z=cluster_data['PC3'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=cluster_data['Value'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Value')
                ),
                name=f'Cluster {cluster}',
                text=[f'Cluster: {cluster}<br>Value: {v:.3f}' 
                      for v in cluster_data['Value']] if 'Value' in cluster_data else f'Cluster {cluster}',
                hoverinfo='text',
                visible=True
            ))
        else:
            # Color by cluster
            fig.add_trace(go.Scatter3d(
                x=cluster_data['PC1'],
                y=cluster_data['PC2'],
                z=cluster_data['PC3'],
                mode='markers',
                marker=dict(size=5),
                name=f'Cluster {cluster}',
                text=f'Cluster {cluster}',
                hoverinfo='text',
                visible=True
            ))
    
    # Create buttons for each cluster
    buttons = []
    for i, cluster in enumerate(unique_clusters):
        visibility = [False] * len(unique_clusters)
        visibility[i] = True
        buttons.append(dict(
            label=f'Cluster {cluster}',
            method='update',
            args=[{'visible': visibility},
                  {'title': f'3D PCA - Cluster {cluster}'}]
        ))
    
    # Add "All" button
    buttons.insert(0, dict(
        label='All Clusters',
        method='update',
        args=[{'visible': [True] * len(unique_clusters)},
              {'title': '3D PCA - All Clusters'}]
    ))
    
    # Update layout
    fig.update_layout(
        title='3D PCA - All Clusters',
        updatemenus=[dict(
            type="dropdown",
            buttons=buttons,
            x=0.05,
            xanchor='left',
            y=1.15,
            yanchor='top'
        )],
        scene=dict(
            xaxis_title=f'PC1 ({var_exp[0]:.1f}%)',
            yaxis_title=f'PC2 ({var_exp[1]:.1f}%)',
            zaxis_title=f'PC3 ({var_exp[2]:.1f}%)'
        )
    )
    
    # Save interactive plot
    fig.write_html(save_path)
    print(f"✅ Saved interactive cluster selector: {save_path}")
    
    return fig

# -----------------------------------------------------------------------------
# 13. METRICS SUMMARY
# -----------------------------------------------------------------------------

def plot_metrics_summary(metrics_dict, save_dir="plots"):
    """
    Create bar plot of metrics across datasets
    
    Args:
        metrics_dict: dictionary with metrics for each dataset
        save_dir: directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data for plotting
    datasets = list(metrics_dict.keys())
    metrics = ['R2', 'MAE', 'MSE', 'RMSE']
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Extract values for this metric
        values = []
        for dataset in datasets:
            if metric in metrics_dict[dataset]:
                values.append(metrics_dict[dataset][metric])
            else:
                values.append(0)
        
        # Create bar plot
        bars = ax.bar(datasets, values, color=plt.cm.Set3(np.arange(len(datasets))))
        ax.set_title(f'{metric} Score', fontsize=12)
        ax.set_ylabel(metric, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Model Performance Metrics', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save plot
    filename = 'metrics_summary.svg'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Saved metrics summary plot: {save_path}")

# -----------------------------------------------------------------------------
# 14. TRAINING HISTORY
# -----------------------------------------------------------------------------

def plot_training_history(train_losses, val_losses, val_accuracies=None, 
                         save_dir="plots"):
    """
    Plot training and validation loss history
    
    Args:
        train_losses: list of training losses
        val_losses: list of validation losses
        val_accuracies: optional list of validation accuracies
        save_dir: directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    if val_accuracies is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot losses
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Training and Validation Loss', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracies
        ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Accuracy', fontsize=11)
        ax2.set_title('Validation Accuracy', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training History', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = 'training_history.svg'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Saved training history plot: {save_path}")

# -----------------------------------------------------------------------------
# 15. UTILITY FUNCTION TO RUN ALL PLOTS
# -----------------------------------------------------------------------------

def generate_all_plots(X_combined, y_true, y_pred, labels=None, 
                      feature_names=None, dataset_name="Dataset", 
                      output_dir="plots"):
    """
    Generate all standard plots for model evaluation
    
    Args:
        X_combined: combined feature matrix
        y_true: true target values
        y_pred: predicted target values
        labels: cluster labels (optional)
        feature_names: names of features (optional)
        dataset_name: name of dataset
        output_dir: base output directory
    """
    print("=" * 60)
    print(f"Generating All Plots for {dataset_name}")
    print("=" * 60)
    
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Prediction vs True plot
    print("\n1. Generating Prediction vs True plot...")
    plot_predicted_vs_true(y_true, y_pred, dataset_name, save_dir=output_dir)
    
    # 2. Residual distribution plot
    print("\n2. Generating Residual Distribution plot...")
    plot_residual_distribution(y_true, y_pred, dataset_name, save_dir=output_dir)
    
    # 3. Williams plot
    print("\n3. Generating Williams plot...")
    plot_williams(y_true, X_combined, dataset_name, save_dir=output_dir)
    
    # 4. Q vs T² plot
    print("\n4. Generating Q vs T² plot...")
    plot_q_t2(X_combined, dataset_name, save_dir=output_dir)
    
    # 5. PCA plots
    print("\n5. Generating PCA plots...")
    pca_dir = os.path.join(output_dir, "pca")
    os.makedirs(pca_dir, exist_ok=True)
    
    # PCA explained variance
    plot_pca_explained_variance(X_combined, save_dir=pca_dir)
    
    # 2D PCA
    if labels is not None:
        plot_pca_2d(X_combined, labels=labels, 
                   title=f"PCA - {dataset_name} (Colored by Cluster)",
                   save_dir=pca_dir)
    else:
        plot_pca_2d(X_combined, title=f"PCA - {dataset_name}",
                   save_dir=pca_dir)
    
    # 3D PCA
    if labels is not None:
        plot_pca_3d(X_combined, labels=labels,
                   title=f"3D PCA - {dataset_name} (Colored by Cluster)",
                   save_dir=pca_dir)
    
    # 6. Clustering plots (if labels provided)
    if labels is not None:
        print("\n6. Generating Clustering plots...")
        cluster_dir = os.path.join(output_dir, "clustering")
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Elbow method
        plot_elbow_method(X_combined, save_dir=cluster_dir)
        
        # t-SNE plots
        tsne_dir = os.path.join(cluster_dir, "tsne")
        os.makedirs(tsne_dir, exist_ok=True)
        
        plot_tsne_2d(X_combined, labels=labels, 
                    title=f"t-SNE - {dataset_name}",
                    save_dir=tsne_dir)
        
        plot_tsne_3d(X_combined, labels=labels,
                    title=f"3D t-SNE - {dataset_name}",
                    save_dir=tsne_dir)
        
        # UMAP plots
        umap_dir = os.path.join(cluster_dir, "umap")
        os.makedirs(umap_dir, exist_ok=True)
        
        plot_umap_2d(X_combined, labels=labels,
                    title=f"UMAP - {dataset_name}",
                    save_dir=umap_dir)
        
        plot_umap_3d(X_combined, labels=labels,
                    title=f"3D UMAP - {dataset_name}",
                    save_dir=umap_dir)
    
    # 7. Pairwise scatter plots (if feature names provided)
    if feature_names is not None and len(feature_names) <= 20:  # Limit to reasonable number
        print("\n7. Generating Pairwise Scatter plots...")
        pairwise_dir = os.path.join(output_dir, "pairwise_scatter")
        os.makedirs(pairwise_dir, exist_ok=True)
        
        # Create dummy IDs
        ids = [f"Sample_{i}" for i in range(len(X_combined))]
        
        # Only plot if we have reasonable number of features
        if len(feature_names) <= 15:
            plot_pairwise_scatter_features(
                X_combined, feature_names[:15], ids, output_dir=pairwise_dir
            )
    
    print("\n" + "=" * 60)
    print(f"✅ All plots generated and saved to: {output_dir}")
    print("=" * 60)
    
    # Return summary of generated plots
    generated_plots = {
        "prediction_plots": ["pred_vs_true", "residual_distribution"],
        "diagnostic_plots": ["williams_plot", "q_vs_t2"],
        "dimensionality_reduction": ["pca_2d", "pca_3d", "pca_explained_variance"]
    }
    
    if labels is not None:
        generated_plots["clustering"] = [
            "elbow_method", "tsne_2d", "tsne_3d", "umap_2d", "umap_3d"
        ]
    
    return generated_plots

# -----------------------------------------------------------------------------
# 16. CONVENIENCE WRAPPERS
# -----------------------------------------------------------------------------

class Visualizer:
    """
    Convenience class for organizing visualization functions
    """
    
    def __init__(self, save_dir="plots"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def predictions(self, y_true, y_pred, dataset_name=""):
        """Plot prediction-related visualizations"""
        print(f"Generating prediction plots for {dataset_name}...")
        
        # Prediction vs True
        r2, slope, intercept = plot_predicted_vs_true(
            y_true, y_pred, dataset_name, save_dir=self.save_dir
        )
        
        # Residual distribution
        residuals, mu, std = plot_residual_distribution(
            y_true, y_pred, dataset_name, save_dir=self.save_dir
        )
        
        return {
            'r2': r2,
            'slope': slope,
            'intercept': intercept,
            'residuals_mean': mu,
            'residuals_std': std
        }
    
    def diagnostics(self, y_true, X, dataset_name=""):
        """Plot diagnostic plots"""
        print(f"Generating diagnostic plots for {dataset_name}...")
        
        # Williams plot
        std_residuals, leverages, outliers = plot_williams(
            y_true, X, dataset_name, save_dir=self.save_dir
        )
        
        # Q vs T² plot
        Q, T2, n_components, explained_var = plot_q_t2(
            X, dataset_name, save_dir=self.save_dir
        )
        
        return {
            'n_outliers': np.sum(outliers),
            'n_components': n_components,
            'explained_variance': explained_var
        }
    
    def dimensionality_reduction(self, X, labels=None, values=None, 
                               dataset_name=""):
        """Plot dimensionality reduction visualizations"""
        print(f"Generating dimensionality reduction plots for {dataset_name}...")
        
        dr_dir = os.path.join(self.save_dir, "dimensionality_reduction")
        os.makedirs(dr_dir, exist_ok=True)
        
        # PCA explained variance
        explained_var, cumulative_var = plot_pca_explained_variance(X, save_dir=dr_dir)
        
        # 2D PCA
        X_pca_2d, var_exp_2d = plot_pca_2d(
            X, labels=labels, color_by=values,
            title=f"PCA - {dataset_name}",
            save_dir=dr_dir
        )
        
        # 3D PCA if we have enough samples
        if X.shape[0] > 10:
            X_pca_3d, var_exp_3d = plot_pca_3d(
                X, labels=labels, color_by=values,
                title=f"3D PCA - {dataset_name}",
                save_dir=dr_dir
            )
        else:
            X_pca_3d, var_exp_3d = None, None
        
        return {
            'explained_variance': explained_var,
            'cumulative_variance': cumulative_var,
            'pca_2d_variance': var_exp_2d,
            'pca_3d_variance': var_exp_3d if var_exp_3d else None
        }
    
    def clustering(self, X, labels, dataset_name=""):
        """Plot clustering visualizations"""
        print(f"Generating clustering plots for {dataset_name}...")
        
        cluster_dir = os.path.join(self.save_dir, "clustering")
        os.makedirs(cluster_dir, exist_ok=True)
        
        # t-SNE plots
        tsne_dir = os.path.join(cluster_dir, "tsne")
        os.makedirs(tsne_dir, exist_ok=True)
        
        X_tsne_2d = plot_tsne_2d(
            X, labels=labels,
            title=f"t-SNE - {dataset_name}",
            save_dir=tsne_dir
        )
        
        X_tsne_3d = plot_tsne_3d(
            X, labels=labels,
            title=f"3D t-SNE - {dataset_name}",
            save_dir=tsne_dir
        )
        
        # UMAP plots
        umap_dir = os.path.join(cluster_dir, "umap")
        os.makedirs(umap_dir, exist_ok=True)
        
        X_umap_2d = plot_umap_2d(
            X, labels=labels,
            title=f"UMAP - {dataset_name}",
            save_dir=umap_dir
        )
        
        X_umap_3d = plot_umap_3d(
            X, labels=labels,
            title=f"3D UMAP - {dataset_name}",
            save_dir=umap_dir
        )
        
        return {
            'tsne_2d': X_tsne_2d,
            'tsne_3d': X_tsne_3d,
            'umap_2d': X_umap_2d,
            'umap_3d': X_umap_3d
        }
    
    def interactive(self, X, labels, values=None, dataset_name=""):
        """Create interactive visualizations"""
        print(f"Generating interactive plots for {dataset_name}...")
        
        interactive_dir = os.path.join(self.save_dir, "interactive")
        os.makedirs(interactive_dir, exist_ok=True)
        
        # Interactive 3D PCA
        fig_pca = create_interactive_3d_pca(
            X, labels, values,
            title=f"Interactive 3D PCA - {dataset_name}",
            save_path=os.path.join(interactive_dir, "interactive_3d_pca.html")
        )
        
        # Interactive cluster selector
        fig_selector = create_interactive_cluster_selector(
            X, labels, values,
            save_path=os.path.join(interactive_dir, "interactive_cluster_selector.html")
        )
        
        return {
            'interactive_pca': fig_pca,
            'interactive_selector': fig_selector
        }

# -----------------------------------------------------------------------------
# MAIN TEST FUNCTION
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("DualcomplexGNN Visualization Module")
    print("=" * 50)
    print("This module contains comprehensive plotting functions.")
    print("Import and use the functions as needed.")
    print("\nExample usage:")
    print("  from visualization.plots import Visualizer")
    print("  viz = Visualizer(save_dir='my_plots')")
    print("  viz.predictions(y_true, y_pred, 'My Dataset')")
    print("=" * 50)