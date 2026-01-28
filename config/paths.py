"""
Path configurations for the project
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_DIR = r"C:\Users\Dorsa\Desktop\dataforproject\lessdata100"
Atom_init_DIR = r"C:\Users\Dorsa\DEEPCHEM_DATA_DIR"

# File paths
ID_PROP_FILE =  r"C:\Users\Dorsa\Desktop\dataforproject\id_prop_less100.xlsx"

# Output directories
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FEATURES_DIR = os.path.join(BASE_DIR, "features")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
for directory in [SAVED_MODELS_DIR, PLOTS_DIR, RESULTS_DIR, FEATURES_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)