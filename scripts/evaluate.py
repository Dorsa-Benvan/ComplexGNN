# scripts/evaluate.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from data.dataset import structure_to_pyg_graph, create_data_loaders
from models.complex_gnn import ComplexGNN
from training.evaluator import evaluate_and_plot
from utils.metrics import save_metrics_to_file
from config.paths import DATA_DIR, ID_PROP_FILE, SAVED_MODELS_DIR

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    id_file = pd.read_excel(ID_PROP_FILE)
    dataset = structure_to_pyg_graph(id_file, DATA_DIR)
    
    # Create data loaders
    train_loader, val_loader, test_loader, gf_scaler, y_scaler = create_data_loaders(dataset)
    
    # Load trained model
    global_dim = gf_scaler.mean_.shape[0]
    model = ComplexGNN(
        in_dim=92, edge_dim=41, num_heads=4,
        hidden_dims=[512, 256, 128, 64], out_dim=1,
        global_dim=global_dim,
        dropout_rate=0.01, use_bn=True, activation='leaky_relu'
    ).to(device)
    
    model_path = os.path.join(SAVED_MODELS_DIR, "best_model.pth")
    model.load_state_dict(torch.load(model_path))
    
    # Evaluate
    train_true, train_pred, train_metrics = evaluate_and_plot(
        model, train_loader, device, "Training", y_scaler
    )
    val_true, val_pred, val_metrics = evaluate_and_plot(
        model, val_loader, device, "Validation", y_scaler
    )
    test_true, test_pred, test_metrics = evaluate_and_plot(
        model, test_loader, device, "Testing", y_scaler
    )
    
    # Save metrics
    metrics_dict = {
        'Training': train_metrics,
        'Validation': val_metrics,
        'Testing': test_metrics
    }
    save_metrics_to_file(metrics_dict)
    
    print("✅ Evaluation complete!")

if __name__ == "__main__":
    evaluate_model()