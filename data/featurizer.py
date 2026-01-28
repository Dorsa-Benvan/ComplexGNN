# Update data/featurizer.py
import json
import os
import requests
from deepchem.feat.material_featurizers.cgcnn_featurizer import CGCNNFeaturizer
from deepchem.utils.data_utils import get_data_dir


class OfflineCGCNNFeaturizer(CGCNNFeaturizer):
    def __init__(self, radius: float = 8.0, max_neighbors: int = 12, step: float = 0.2,
                 auto_download: bool = True):
        self.radius = radius
        self.max_neighbors = int(max_neighbors)
        self.step = step
        
        data_dir = get_data_dir()
        json_path = os.path.join(data_dir, "atom_init.json")
        
        # Auto-download if file doesn't exist
        if not os.path.exists(json_path) and auto_download:
            print(f"⚠️ atom_init.json not found at {json_path}")
            print("🌐 Attempting to download...")
            self._download_atom_init(json_path)
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"Need atom_init.json in {data_dir}.\n"
                f"Please download from:\n"
                f"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/atom_init.json\n"
                f"Or run: python scripts/download_atom_init.py"
            )
        
        with open(json_path, "r") as f:
            atom_init = json.load(f)
        
        import numpy as _np
        self.atom_features = {
            int(k): _np.array(v, dtype=_np.float32) for k, v in atom_init.items()
        }
        self.valid_atom_number = set(self.atom_features.keys())
    
    def _download_atom_init(self, output_path):
        """Download atom_init.json from DeepChem's S3 bucket"""
        url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/atom_init.json'
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Downloaded atom_init.json to {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False