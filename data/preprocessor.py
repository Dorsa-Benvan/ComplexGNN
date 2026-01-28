"""
Data preprocessing, loading, and preparation functions
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from config.paths import DATA_DIR
from data.global_features import extract_global_features, build_global_feature_vector


def load_and_extract_global_features(id_file_path="id_prop_filtered.xlsx", 
                                    data_dir=DATA_DIR, 
                                    save_features=True,
                                    max_samples=None):
    """
    Load data from Excel file and extract global features from CIF files
    
    Args:
        id_file_path: Path to Excel file with IDs and properties
        data_dir: Directory containing CIF files
        save_features: Whether to save extracted features to disk
        max_samples: Limit number of samples (for debugging)
        
    Returns:
        all_gf: Global features array
        all_y: Target values array
        row_labels: Crystal IDs
    """
    print("📊 Loading data and extracting global features...")
    
    # Load ID and property file
    id_file = pd.read_excel(id_file_path)
    
    if max_samples:
        id_file = id_file.head(max_samples)
        print(f"   (Limited to {max_samples} samples for debugging)")
    
    all_gf = []
    all_y = []
    row_labels = []
    failed_files = []
    
    # Extract features for each CIF
    total_files = len(id_file)
    for idx, (cif_id, target) in enumerate(zip(id_file.iloc[:, 0].astype(str), id_file.iloc[:, 1])):
        cif_path = os.path.join(data_dir, f"{cif_id}.cif")
        
        try:
            # Extract global features
            features = extract_global_features(cif_path)
            if not features:
                print(f"⚠️  Skipping {cif_id}: Empty features")
                failed_files.append(cif_id)
                continue
                
            # Build feature vector
            gf = build_global_feature_vector(features)
            all_gf.append(gf)
            all_y.append(target)
            row_labels.append(cif_id)
            
            # Progress update
            if (idx + 1) % 100 == 0:
                print(f"   Processed {idx + 1}/{total_files} CIFs")
                
        except Exception as e:
            print(f"❌ Error processing {cif_id}: {e}")
            failed_files.append(cif_id)
            continue
    
    # Convert to numpy arrays
    if all_gf:
        all_gf = np.stack(all_gf)
        all_y = np.array(all_y).reshape(-1, 1)
        
        print(f"\n✅ Successfully processed {len(all_gf)}/{total_files} CIFs")
        print(f"   Failed: {len(failed_files)} CIFs")
        
        if failed_files:
            print(f"   Failed IDs: {failed_files[:10]}{'...' if len(failed_files) > 10 else ''}")
    else:
        raise ValueError("No features extracted! Check your CIF files and paths.")
    
    # Save features if requested
    if save_features and len(all_gf) > 0:
        os.makedirs("features", exist_ok=True)
        np.save("features/all_global_features.npy", all_gf)
        np.save("features/all_targets.npy", all_y)
        np.save("features/crystal_ids.npy", np.array(row_labels))
        
        print(f"💾 Saved features to 'features/' directory")
        print(f"   - all_global_features.npy: {all_gf.shape}")
        print(f"   - all_targets.npy: {all_y.shape}")
        print(f"   - crystal_ids.npy: {len(row_labels)} IDs")
    
    return all_gf, all_y, row_labels


def load_saved_features(features_dir="features"):
    """
    Load previously saved features from disk
    
    Args:
        features_dir: Directory containing saved features
        
    Returns:
        all_gf: Global features array
        all_y: Target values array
        row_labels: Crystal IDs
    """
    print(f"📂 Loading saved features from '{features_dir}/'...")
    
    all_gf = np.load(os.path.join(features_dir, "all_global_features.npy"))
    all_y = np.load(os.path.join(features_dir, "all_targets.npy"))
    row_labels = np.load(os.path.join(features_dir, "crystal_ids.npy"))
    
    print(f"✅ Loaded:")
    print(f"   - Global features: {all_gf.shape}")
    print(f"   - Targets: {all_y.shape}")
    print(f"   - Crystal IDs: {len(row_labels)}")
    
    return all_gf, all_y, row_labels


def create_train_val_test_split(features, targets, test_size=0.3, val_size=0.15, random_state=42):
    """
    Create train/validation/test splits
    
    Args:
        features: Feature matrix
        targets: Target values
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed for reproducibility
        
    Returns:
        Splits for features and targets
    """
    print("✂️  Creating train/validation/test splits...")
    
    # Calculate split sizes
    train_size = 1.0 - test_size - val_size
    
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, targets, 
        test_size=(val_size + test_size),
        random_state=random_state,
        shuffle=True
    )
    
    # Second split: val vs test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_size/(val_size + test_size),
        random_state=random_state,
        shuffle=True
    )
    
    print(f"✅ Split sizes:")
    print(f"   - Train: {len(X_train)} samples ({train_size*100:.1f}%)")
    print(f"   - Validation: {len(X_val)} samples ({val_size*100:.1f}%)")
    print(f"   - Test: {len(X_test)} samples ({test_size*100:.1f}%)")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }


def fit_scalers(X_train, y_train):
    """
    Fit scalers on training data
    
    Args:
        X_train: Training features
        y_train: Training targets
        
    Returns:
        gf_scaler: Fitted StandardScaler for features
        y_scaler: Fitted StandardScaler for targets
    """
    print("⚖️  Fitting scalers on training data...")
    
    gf_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    gf_scaler.fit(X_train)
    y_scaler.fit(y_train)
    
    print(f"✅ Scalers fitted:")
    print(f"   - Feature scaler: mean={gf_scaler.mean_.shape}, std={gf_scaler.scale_.shape}")
    print(f"   - Target scaler: mean={y_scaler.mean_.shape}, std={y_scaler.scale_.shape}")
    
    return gf_scaler, y_scaler


def apply_scaling(data_dict, gf_scaler, y_scaler):
    """
    Apply scaling to all splits
    
    Args:
        data_dict: Dictionary containing X_train, y_train, X_val, y_val, X_test, y_test
        gf_scaler: Fitted StandardScaler for features
        y_scaler: Fitted StandardScaler for targets
        
    Returns:
        Scaled data dictionary
    """
    print("🔧 Applying scaling to all splits...")
    
    scaled_data = {}
    
    for split in ['train', 'val', 'test']:
        X_key = f'X_{split}'
        y_key = f'y_{split}'
        
        if X_key in data_dict and y_key in data_dict:
            scaled_data[X_key] = gf_scaler.transform(data_dict[X_key])
            scaled_data[y_key] = y_scaler.transform(data_dict[y_key])
            
            print(f"   - {split.capitalize()}: {scaled_data[X_key].shape}")
    
    return scaled_data


def calculate_leverage(X):
    """
    Calculate leverage (hat matrix diagonal)
    
    Args:
        X: Feature matrix
        
    Returns:
        leverage values
    """
    H = X @ np.linalg.pinv(X)
    return np.clip(np.diag(H), 0.0, 1.0)


def get_feature_names():
    """
    Get names of the 17 continuous global features
    
    Returns:
        List of feature names
    """
    feature_names = [
        'a', 'b', 'c',
        'alpha', 'beta', 'gamma',
        'volume', 'packing_density',
        'ratio_c_a', 'ratio_b_a',
        'num_atoms',
        'avg_bond_length', 'avg_coordination_number',
        'avg_bond_angle',
        'num_distinct_wyckoffs',
        'avg_en_diff', 'polyhedra_shape_index'
    ]
    
    return feature_names


def create_feature_dataframe(features_array, crystal_ids=None, include_one_hot=False):
    """
    Create DataFrame from features array
    
    Args:
        features_array: Features array from build_global_feature_vector
        crystal_ids: Optional list of crystal IDs
        include_one_hot: Whether to include one-hot encoded columns
        
    Returns:
        DataFrame with features
    """
    # Get base feature names (17 continuous features)
    base_features = get_feature_names()
    
    # Determine number of one-hot features
    n_base = len(base_features)
    n_total = features_array.shape[1]
    n_one_hot = n_total - n_base
    
    # Create column names
    if include_one_hot and n_one_hot > 0:
        # We don't know the exact one-hot categories here
        # They're defined in global_features.py
        one_hot_names = [f'one_hot_{i}' for i in range(n_one_hot)]
        column_names = base_features + one_hot_names
    else:
        # Use only base features
        column_names = base_features
        features_array = features_array[:, :n_base]
    
    # Create DataFrame
    df = pd.DataFrame(features_array, columns=column_names)
    
    # Add crystal IDs if provided
    if crystal_ids is not None:
        df.insert(0, 'crystal_id', crystal_ids)
    
    return df


def check_data_quality(features, targets, feature_names=None):
    """
    Perform basic data quality checks
    
    Args:
        features: Feature matrix
        targets: Target values
        feature_names: Optional list of feature names
        
    Returns:
        Dictionary with quality metrics
    """
    print("🔍 Performing data quality checks...")
    
    quality_report = {
        'n_samples': features.shape[0],
        'n_features': features.shape[1],
        'missing_values': np.isnan(features).sum(),
        'infinite_values': np.isinf(features).sum(),
        'feature_stats': {}
    }
    
    # Check for missing/infinite values
    if quality_report['missing_values'] > 0:
        print(f"⚠️  Found {quality_report['missing_values']} missing values")
    
    if quality_report['infinite_values'] > 0:
        print(f"⚠️  Found {quality_report['infinite_values']} infinite values")
    
    # Basic statistics for each feature
    if feature_names is not None and len(feature_names) == features.shape[1]:
        for i, name in enumerate(feature_names):
            quality_report['feature_stats'][name] = {
                'mean': np.mean(features[:, i]),
                'std': np.std(features[:, i]),
                'min': np.min(features[:, i]),
                'max': np.max(features[:, i])
            }
    
    # Target statistics
    quality_report['target_stats'] = {
        'mean': np.mean(targets),
        'std': np.std(targets),
        'min': np.min(targets),
        'max': np.max(targets)
    }
    
    print(f"✅ Data quality check complete:")
    print(f"   - Samples: {quality_report['n_samples']}")
    print(f"   - Features: {quality_report['n_features']}")
    print(f"   - Target range: {quality_report['target_stats']['min']:.3f} to "
          f"{quality_report['target_stats']['max']:.3f}")
    
    return quality_report


# -----------------------------------------------------------------------------
# MAIN PREPROCESSING PIPELINE
# -----------------------------------------------------------------------------

def run_preprocessing_pipeline(id_file_path="id_prop_filtered.xlsx",
                              data_dir=DATA_DIR,
                              test_size=0.3,
                              val_size=0.15,
                              random_state=42,
                              force_extract=False,
                              max_samples=None):
    """
    Complete preprocessing pipeline
    
    Args:
        id_file_path: Path to Excel file
        data_dir: Directory with CIF files
        test_size: Test set proportion
        val_size: Validation set proportion
        random_state: Random seed
        force_extract: Force re-extraction even if saved features exist
        max_samples: Limit number of samples
        
    Returns:
        Dictionary with all preprocessed data
    """
    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Check if features already exist
    features_exist = (os.path.exists("features/all_global_features.npy") and
                     os.path.exists("features/all_targets.npy") and
                     os.path.exists("features/crystal_ids.npy"))
    
    if features_exist and not force_extract:
        print("📂 Found saved features. Loading...")
        all_gf, all_y, row_labels = load_saved_features()
    else:
        print("🔧 Extracting features from CIF files...")
        all_gf, all_y, row_labels = load_and_extract_global_features(
            id_file_path, data_dir, save_features=True, max_samples=max_samples
        )
    
    # Data quality check
    feature_names = get_feature_names()
    quality_report = check_data_quality(all_gf[:, :len(feature_names)], all_y, feature_names)
    
    # Create splits
    splits = create_train_val_test_split(
        all_gf, all_y, 
        test_size=test_size, 
        val_size=val_size, 
        random_state=random_state
    )
    
    # Fit scalers on training data
    gf_scaler, y_scaler = fit_scalers(splits['X_train'], splits['y_train'])
    
    # Apply scaling
    scaled_splits = apply_scaling(splits, gf_scaler, y_scaler)
    
    # Prepare final output
    processed_data = {
        'raw': {
            'features': all_gf,
            'targets': all_y,
            'ids': row_labels,
            'feature_names': feature_names
        },
        'splits': {
            'train': {'X': splits['X_train'], 'y': splits['y_train']},
            'val': {'X': splits['X_val'], 'y': splits['y_val']},
            'test': {'X': splits['X_test'], 'y': splits['y_test']}
        },
        'scaled_splits': {
            'train': {'X': scaled_splits['X_train'], 'y': scaled_splits['y_train']},
            'val': {'X': scaled_splits['X_val'], 'y': scaled_splits['y_val']},
            'test': {'X': scaled_splits['X_test'], 'y': scaled_splits['y_test']}
        },
        'scalers': {
            'features': gf_scaler,
            'targets': y_scaler
        },
        'quality_report': quality_report,
        'split_info': {
            'test_size': test_size,
            'val_size': val_size,
            'train_size': 1.0 - test_size - val_size,
            'random_state': random_state
        }
    }
    
    print("\n" + "=" * 60)
    print("✅ PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"📊 Summary:")
    print(f"   - Total samples: {len(all_gf)}")
    print(f"   - Features per sample: {all_gf.shape[1]}")
    print(f"   - Train set: {len(splits['X_train'])} samples")
    print(f"   - Validation set: {len(splits['X_val'])} samples")
    print(f"   - Test set: {len(splits['X_test'])} samples")
    
    return processed_data


# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------

def save_processed_data(processed_data, output_dir="processed_data"):
    """
    Save processed data to disk
    
    Args:
        processed_data: Dictionary from run_preprocessing_pipeline
        output_dir: Directory to save data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw data
    np.save(os.path.join(output_dir, "raw_features.npy"), processed_data['raw']['features'])
    np.save(os.path.join(output_dir, "raw_targets.npy"), processed_data['raw']['targets'])
    np.save(os.path.join(output_dir, "raw_ids.npy"), processed_data['raw']['ids'])
    
    # Save splits
    for split_name in ['train', 'val', 'test']:
        split_data = processed_data['splits'][split_name]
        np.save(os.path.join(output_dir, f"{split_name}_X.npy"), split_data['X'])
        np.save(os.path.join(output_dir, f"{split_name}_y.npy"), split_data['y'])
    
    # Save scaled splits
    for split_name in ['train', 'val', 'test']:
        scaled_data = processed_data['scaled_splits'][split_name]
        np.save(os.path.join(output_dir, f"{split_name}_X_scaled.npy"), scaled_data['X'])
        np.save(os.path.join(output_dir, f"{split_name}_y_scaled.npy"), scaled_data['y'])
    
    # Save scalers
    import joblib
    joblib.dump(processed_data['scalers']['features'], 
                os.path.join(output_dir, "feature_scaler.joblib"))
    joblib.dump(processed_data['scalers']['targets'], 
                os.path.join(output_dir, "target_scaler.joblib"))
    
    # Save metadata
    import json
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump({
            'feature_names': processed_data['raw']['feature_names'],
            'split_info': processed_data['split_info'],
            'quality_report': processed_data['quality_report']
        }, f, indent=2)
    
    print(f"💾 Saved processed data to '{output_dir}/'")


def load_processed_data(input_dir="processed_data"):
    """
    Load previously processed data from disk
    
    Args:
        input_dir: Directory containing processed data
        
    Returns:
        Dictionary with loaded data
    """
    import joblib
    import json
    
    print(f"📂 Loading processed data from '{input_dir}/'...")
    
    # Load metadata
    with open(os.path.join(input_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    # Load raw data
    raw_features = np.load(os.path.join(input_dir, "raw_features.npy"))
    raw_targets = np.load(os.path.join(input_dir, "raw_targets.npy"))
    raw_ids = np.load(os.path.join(input_dir, "raw_ids.npy"))
    
    # Load splits
    splits = {}
    scaled_splits = {}
    
    for split_name in ['train', 'val', 'test']:
        splits[split_name] = {
            'X': np.load(os.path.join(input_dir, f"{split_name}_X.npy")),
            'y': np.load(os.path.join(input_dir, f"{split_name}_y.npy"))
        }
        
        scaled_splits[split_name] = {
            'X': np.load(os.path.join(input_dir, f"{split_name}_X_scaled.npy")),
            'y': np.load(os.path.join(input_dir, f"{split_name}_y_scaled.npy"))
        }
    
    # Load scalers
    feature_scaler = joblib.load(os.path.join(input_dir, "feature_scaler.joblib"))
    target_scaler = joblib.load(os.path.join(input_dir, "target_scaler.joblib"))
    
    # Reconstruct processed data dictionary
    processed_data = {
        'raw': {
            'features': raw_features,
            'targets': raw_targets,
            'ids': raw_ids,
            'feature_names': metadata['feature_names']
        },
        'splits': splits,
        'scaled_splits': scaled_splits,
        'scalers': {
            'features': feature_scaler,
            'targets': target_scaler
        },
        'quality_report': metadata['quality_report'],
        'split_info': metadata['split_info']
    }
    
    print(f"✅ Loaded processed data:")
    print(f"   - Raw data: {raw_features.shape[0]} samples")
    print(f"   - Features: {raw_features.shape[1]} per sample")
    
    return processed_data


# -----------------------------------------------------------------------------
# MAIN FUNCTION FOR STANDALONE USE
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing preprocessor module...")
    
    # Example usage
    try:
        # Run complete preprocessing pipeline
        processed_data = run_preprocessing_pipeline(
            max_samples=100  # Limit for testing
        )
        
        # Save processed data
        save_processed_data(processed_data, "test_processed_data")
        
        print("\n✅ Preprocessor test successful!")
        
    except Exception as e:
        print(f"❌ Error in preprocessor test: {e}")
        import traceback
        traceback.print_exc()