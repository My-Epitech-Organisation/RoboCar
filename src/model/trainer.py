"""
Training module for RoboCar neural networks.

This module:
1. Sets up training parameters and optimization
2. Handles the training loop and validation
3. Tracks metrics and saves checkpoints
4. Supports early stopping and other training features
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.best_weights = None
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, model, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                # Keep a copy of the best weights
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    # Restore the best weights
                    model.load_state_dict(self.best_weights)


class TrainingHistory:
    """Class to store and retrieve training metrics"""
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_times': []
        }
    
    def update(self, train_loss, val_loss, epoch_time):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['epoch_times'].append(epoch_time)
    
    def get_history(self):
        return self.history


def train_model(model, X_train, y_train, X_val, y_val, 
                epochs=100, batch_size=32, learning_rate=0.001,
                weight_decay=0.0001, patience=10, 
                project_root=None, checkpoint_dir=None):
    """
    Train a PyTorch model.
    
    Args:
        model (nn.Module): PyTorch model to train
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        X_val (np.ndarray): Validation features
        y_val (np.ndarray): Validation targets
        epochs (int): Maximum number of epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        weight_decay (float): L2 regularization parameter
        patience (int): Early stopping patience
        project_root (str): Path to project root (for saving)
        checkpoint_dir (str): Directory for checkpoints
    
    Returns:
        tuple: (trained model, training history)
    """
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(device)
    
    # Create datasets and data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Initialize early stopping and training history
    early_stopping = EarlyStopping(patience=patience)
    history = TrainingHistory()
    
    # Set up checkpoint directory
    if project_root is not None and checkpoint_dir is None:
        checkpoint_dir = os.path.join(project_root, "checkpoints")
    
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Starting training for {epochs} epochs")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_losses = []
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Calculate average training loss
        avg_train_loss = np.mean(train_losses)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Update history
        history.update(avg_train_loss, val_loss, epoch_time)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train loss: {avg_train_loss:.4f}, "
              f"Val loss: {val_loss:.4f}, "
              f"Time: {epoch_time:.2f}s")
        
        # Save checkpoint
        if checkpoint_dir is not None and (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Check for early stopping
        early_stopping(model, val_loss)
        if early_stopping.should_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    print("Training completed")
    
    return model, history.get_history()


def save_model(model, path, input_size=None, model_type=None, metadata=None):
    """
    Save trained model with metadata.
    
    Args:
        model (nn.Module): Trained PyTorch model
        path (str): Path to save the model
        input_size (int): Input size of the model
        model_type (str): Type of the model architecture
        metadata (dict): Additional metadata to save
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Prepare metadata
    if metadata is None:
        metadata = {}
    
    if input_size is not None:
        metadata['input_size'] = input_size
    
    if model_type is not None:
        metadata['model_type'] = model_type
    
    metadata['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Save model state
    model_state = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata
    }
    
    # Save to file
    torch.save(model_state, path)
    print(f"Model saved to {path}")
