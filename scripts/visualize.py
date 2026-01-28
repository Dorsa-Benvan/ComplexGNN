import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from visualization.plots import generate_all_plots

def main():
    print("Generating visualizations...")
    
    # Load saved data
    # Note: Adjust the paths as needed
    X_combined = np.load("features/combined_features.npy")
    y_true = np.load("results/y_true.npy")
    y_pred = np.load("results/y_pred.npy")
    labels = np.load("results/cluster_labels.npy")
    
    # Generate all plots
    generate_all_plots(
        X_combined, y_true, y_pred, 
        labels=labels,
        dataset_name="My Dataset",
        output_dir="plots"
    )
    
    print("Visualization completed.")

if __name__ == "__main__":
    main()