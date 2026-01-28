"""
Script to generate all plots from saved data
"""

import sys
import os
import numpy as np
import pandas as pd

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.plots import Visualizer, generate_all_plots

def main():
    print("Generating Plots from Saved Data")
    print("=" * 50)
    
    # Load saved data
    print("Loading saved data...")
    try:
        X_combined = np.load("features/combined_features.npy")
        y_true = np.load("results/y_true.npy")
        y_pred = np.load("results/y_pred.npy")
        labels = np.load("results/cluster_labels.npy")
        
        print(f"Loaded data shapes:")
        print(f"  X_combined: {X_combined.shape}")
        print(f"  y_true: {y_true.shape}")
        print(f"  y_pred: {y_pred.shape}")
        print(f"  labels: {labels.shape}")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please run training first or check file paths.")
        return
    
    # Create visualizer
    viz = Visualizer(save_dir="plots_comprehensive")
    
    # Generate all plots
    print("\nGenerating comprehensive visualizations...")
    
    # Prediction plots
    pred_results = viz.predictions(y_true, y_pred, "Combined Dataset")
    
    # Diagnostic plots
    diag_results = viz.diagnostics(y_true, X_combined, "Combined Dataset")
    
    # Dimensionality reduction
    dr_results = viz.dimensionality_reduction(X_combined, labels, 
                                              dataset_name="Combined Dataset")
    
    # Clustering visualization
    cluster_results = viz.clustering(X_combined, labels, 
                                     dataset_name="Combined Dataset")
    
    # Interactive plots
    interactive_results = viz.interactive(X_combined, labels, 
                                          dataset_name="Combined Dataset")
    
    print("\n" + "=" * 50)
    print("✅ All visualizations generated successfully!")
    print("Check the 'plots_comprehensive' directory for output.")
    print("=" * 50)

if __name__ == "__main__":
    main()