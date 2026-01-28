"""
Dataset creation and DataLoader preparation
"""

import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pymatgen.core import Structure

from data.featurizer import OfflineCGCNNFeaturizer
from data.global_features import extract_global_features, build_global_feature_vector
from config.paths import DATA_DIR


def structure_to_pyg_graph(df, data_dir=DATA_DIR):
    """Convert crystal structures to PyTorch Geometric graphs"""
    featurizer = OfflineCGCNNFeaturizer()
    data_list = []
    
    for idx, lbl in zip(df.iloc[:, 0].astype(str), df.iloc[:, 1]):
        cif = os.path.join(data_dir, f"{idx}.cif")
        
        try:
            raw = featurizer.featurize([Structure.from_file(cif)])
            feat = raw
            while isinstance(feat, (np.ndarray, list)):
                feat = feat[0] if len(feat) > 0 else None
            
            if feat is None:
                continue
                
            data = feat.to_pyg_graph()
            gf = build_global_feature_vector(extract_global_features(cif))
            data.global_features = torch.tensor(gf, dtype=torch.float).unsqueeze(0)
            data.y = torch.tensor([lbl], dtype=torch.float)
            data.cif_id = idx
            data_list.append(data)
        except Exception as e:
            print(f"Error processing {idx}: {e}")
            continue
            
    return data_list


def create_data_loaders(dataset, batch_size=32, train_split=0.7, val_split=0.15, random_state=42):
    """Create train/validation/test DataLoaders with scaling"""
    # Calculate split sizes
    test_split = 1.0 - train_split - val_split
    
    # Initial split
    train_ds, temp_ds = train_test_split(
        dataset, test_size=(val_split + test_split), 
        random_state=random_state, shuffle=True
    )
    
    # Second split
    val_ds, test_ds = train_test_split(
        temp_ds, test_size=test_split/(val_split + test_split),
        random_state=random_state, shuffle=True
    )
    
    # Extract training data for scaling
    X_train = np.vstack([d.global_features.numpy() for d in train_ds])
    y_train = np.concatenate([d.y.numpy() for d in train_ds]).reshape(-1, 1)
    
    # Fit scalers on training data only
    gf_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    
    # Apply scaling to all datasets
    for ds in (train_ds, val_ds, test_ds):
        for d in ds:
            gf = d.global_features.numpy()
            gf_scaled = gf_scaler.transform(gf)
            d.global_features = torch.tensor(gf_scaled, dtype=torch.float)
            
            yt = d.y.numpy().reshape(1, 1)
            yt_scaled = y_scaler.transform(yt)
            d.y = torch.tensor(yt_scaled.ravel(), dtype=torch.float)
    
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, gf_scaler, y_scaler