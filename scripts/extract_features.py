import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from data.global_features import extract_global_features, build_global_feature_vector
from config.paths import DATA_DIR, ID_PROP_FILE

def main():
    print("Extracting global features...")
    
    # Load the Excel file
    id_file = pd.read_excel(ID_PROP_FILE)
    
    all_gf = []
    all_y = []
    row_labels = []
    
    for idx, lbl in zip(id_file.iloc[:, 0].astype(str), id_file.iloc[:, 1]):
        cif_path = os.path.join(DATA_DIR, f"{idx}.cif")
        try:
            print(f"Processing CIF: {idx}")
            features = extract_global_features(cif_path)
            if not features:
                print(f"Skipping {idx} due to empty features.")
                continue
            gf = build_global_feature_vector(features)
            all_gf.append(gf)
            all_y.append(lbl)
            row_labels.append(idx)
            
            if len(all_gf) % 100 == 0:
                print(f"{len(all_gf)} CIFs processed.")
                
        except Exception as e:
            print(f"Error on {idx}: {e}")
    
    # Convert to numpy arrays
    all_gf = np.stack(all_gf)
    all_y = np.array(all_y).reshape(-1, 1)
    
    # Save the features, targets, and IDs
    np.save("all_global_features.npy", all_gf)
    np.save("all_targets.npy", all_y)
    np.save("crystal_ids.npy", np.array(row_labels))
    
    print(f"✅ Saved features for {len(all_gf)} crystals.")

if __name__ == "__main__":
    main()