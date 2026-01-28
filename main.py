"""
Main entry point for the ComplexGNN project
"""

import os
import sys
import torch
import pandas as pd
import numpy as np

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.paths import DATA_DIR, ID_PROP_FILE, SAVED_MODELS_DIR
from data.dataset import structure_to_pyg_graph, create_data_loaders
from models.complex_gnn import ComplexGNN
from training.trainer import Trainer
from training.evaluator import evaluate_and_plot, calculate_metrics
from utils.metrics import save_metrics_to_file


def main():
    print("=" * 60)
    print("DualcomplexGNN - Crystal Property Prediction")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Step 1: Load data
    print("\n1. Loading data...")
    try:
        id_file = pd.read_excel(ID_PROP_FILE)
        print(f"   Loaded {len(id_file)} entries from {ID_PROP_FILE}")
    except Exception as e:
        print(f"   Error loading data: {e}")
        return
    
    # Step 2: Create dataset
    print("2. Creating dataset...")
    dataset = structure_to_pyg_graph(id_file, DATA_DIR)
    print(f"   Created {len(dataset)} graph samples")
    
    # Step 3: Create data loaders
    print("3. Creating data loaders...")
    train_loader, val_loader, test_loader, gf_scaler, y_scaler = create_data_loaders(
        dataset, batch_size=32, train_split=0.7, val_split=0.15
    )
    print(f"   Train: {len(train_loader.dataset)} samples")
    print(f"   Validation: {len(val_loader.dataset)} samples")
    print(f"   Test: {len(test_loader.dataset)} samples")
    
    # Step 4: Create model
    print("4. Creating model...")
    global_dim = gf_scaler.mean_.shape[0]
    model = ComplexGNN(
        in_dim=92, edge_dim=41, num_heads=4,
        hidden_dims=[512, 256, 128, 64], out_dim=1,
        global_dim=global_dim,
        dropout_rate=0.01, use_bn=True, activation='leaky_relu'
    ).to(device)
    print(f"   Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Step 5: Train model
    print("\n5. Training model...")
    trainer = Trainer(model, device)
    model_save_path = os.path.join(SAVED_MODELS_DIR, "best_model.pth")
    val_history = trainer.train(
        train_loader, val_loader,
        epochs=200, lr=1e-3, weight_decay=1e-4,
        patience=20, model_save_path=model_save_path
    )
    
    # Step 6: Evaluate model
    print("\n6. Evaluating model...")
    
    # Evaluate on all splits
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
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Best model saved to: {model_save_path}")
    print("Plots saved to: plots/")
    print("Metrics saved to: results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
