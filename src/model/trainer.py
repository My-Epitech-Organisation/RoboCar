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


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, 
                learning_rate=0.001, project_root=".", use_scheduler=True,
                early_stopping_patience=15, weight_decay=1e-4):
    """
    Entraînement amélioré avec:
    - Planificateur de taux d'apprentissage
    - Régularisation L2 (weight_decay)
    - Early stopping amélioré
    - Sauvegarde du meilleur modèle
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Convertir les données en tenseurs
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Préparer les datasets et dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Optimiseur avec régularisation L2
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Planificateur de taux d'apprentissage
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=7, verbose=True
        )
    
    # Fonction de perte: MSE pour la direction, MAE pour l'accélération
    criterion_steering = nn.MSELoss()
    criterion_accel = nn.L1Loss()
    
    # Tracking des métriques
    best_val_loss = float('inf')
    best_model_path = os.path.join(project_root, "model_checkpoint.pth")
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_steering_mae': [], 'val_accel_mae': []}

    # Boucle d'entraînement
    for epoch in range(epochs):
        # Mode entraînement
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            # Forward pass
            outputs = model(inputs)
            
            # Gérer le cas où le modèle renvoie un tuple (sortie, hidden_state)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Extraire seulement la sortie, ignorer hidden_state
            
            # Calculer la perte
            loss_steering = criterion_steering(outputs[:, 0], targets[:, 0])
            loss_accel = criterion_accel(outputs[:, 1], targets[:, 1])
            loss = 0.7 * loss_steering + 1.3 * loss_accel  # Donner plus d'importance à l'accélération
            
            # Backward pass et optimisation
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping pour stabilité
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Mode évaluation
        model.eval()
        val_loss = 0.0
        steering_errors = []
        accel_errors = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                
                # Gérer le cas où le modèle renvoie un tuple (sortie, hidden_state)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Extraire seulement la sortie, ignorer hidden_state
                
                # Calculer les métriques
                steering_error = torch.abs(outputs[:, 0] - targets[:, 0])
                accel_error = torch.abs(outputs[:, 1] - targets[:, 1])
                
                steering_errors.extend(steering_error.cpu().numpy())
                accel_errors.extend(accel_error.cpu().numpy())
                
                # Perte totale
                loss_steering = criterion_steering(outputs[:, 0], targets[:, 0])
                loss_accel = criterion_accel(outputs[:, 1], targets[:, 1])
                loss = loss_steering + 0.5 * loss_accel
                val_loss += loss.item()
        
        # Calcul des moyennes
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_steering_mae = np.mean(steering_errors)
        val_accel_mae = np.mean(accel_errors)
        
        # Mettre à jour le scheduler
        if scheduler:
            scheduler.step(val_loss)
        
        # Sauvegarder les métriques
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_steering_mae'].append(val_steering_mae)
        history['val_accel_mae'].append(val_accel_mae)
        
        # Afficher progression
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Steering MAE: {val_steering_mae:.4f}, Accel MAE: {val_accel_mae:.4f}")
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Déterminer si le modèle utilise uniquement des raycasts
            use_only_raycasts = hasattr(model, 'has_other_features') and not model.has_other_features
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'input_size': X_train.shape[1],
                # Correction - ne pas soustraire 1 si on utilise seulement les raycasts
                'num_rays': X_train.shape[1] if use_only_raycasts else X_train.shape[1] - 1,
                'model_type': model.__class__.__name__.lower().replace('model', '')
            }, best_model_path)
            patience_counter = 0
            print(f"Meilleur modèle sauvegardé (perte: {val_loss:.4f})")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping activé après {epoch+1} époques")
            break
    
    # Charger le meilleur modèle
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history


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
