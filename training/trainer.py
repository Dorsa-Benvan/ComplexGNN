"""
Training loop and model training utilities
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np


class Trainer:
    def __init__(self, model, device, criterion=nn.MSELoss()):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        train_loss = 0
        
        for data in train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(data), data.y.view(-1, 1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            train_loss += loss.item()
            
        return train_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                val_loss += self.criterion(self.model(data), data.y.view(-1, 1)).item()
                
        return val_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=200, lr=1e-3, 
              weight_decay=1e-4, patience=20, model_save_path='best_model.pth'):
        """Main training loop with early stopping"""
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10)
        
        no_improve = 0
        val_history = []
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            val_history.append(val_loss)
            
            print(f"Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping and model checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                no_improve = 0
                torch.save(self.model.state_dict(), model_save_path)
                print(f"  -> Saved best model (val_loss: {val_loss:.4f})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load(model_save_path))
        print(f"Training completed. Best validation loss: {self.best_val_loss:.4f}")
        
        return val_history