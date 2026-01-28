# scripts/train.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from data.dataset import structure_to_pyg_graph, create_data_loaders
from models.complex_gnn import ComplexGNN
from training.trainer import Trainer
from config.paths import DATA_DIR, ID_PROP_FILE, SAVED_MODELS_DIR

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    id_file = pd.read_excel(ID_PROP_FILE)
    dataset = structure_to_pyg_graph(id_file, DATA_DIR)
    
    # Create data loaders
    train_loader, val_loader, test_loader, gf_scaler, y_scaler = create_data_loaders(dataset)
    
    # Create model
    global_dim = gf_scaler.mean_.shape[0]
    model = ComplexGNN(
        in_dim=92, edge_dim=41, num_heads=4,
        hidden_dims=[512, 256, 128, 64], out_dim=1,
        global_dim=global_dim,
        dropout_rate=0.01, use_bn=True, activation='leaky_relu'
    ).to(device)
    
    # Train
    trainer = Trainer(model, device)
    model_save_path = os.path.join(SAVED_MODELS_DIR, "best_model.pth")
    trainer.train(train_loader, val_loader, model_save_path=model_save_path)
    
    print("✅ Training complete!")
    return model, train_loader, val_loader, test_loader, gf_scaler, y_scaler

if __name__ == "__main__":
    train_model()