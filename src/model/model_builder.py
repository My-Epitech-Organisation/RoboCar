"""
Neural network model builder for RoboCar.

This module defines different neural network architectures for:
1. Processing raycast data
2. Predicting steering commands
3. Different complexity levels based on data availability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    """
    Simple fully connected neural network for raycast processing.
    Suitable for small datasets or initial testing.
    """
    def __init__(self, input_size, hidden_size=64):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation for regression output
        return x


class CNNModel(nn.Module):
    """
    1D Convolutional neural network for spatial processing of raycast data.
    Better captures spatial relationships in raycast patterns.
    """
    def __init__(self, input_size, num_rays):
        super(CNNModel, self).__init__()
        self.num_rays = num_rays
        
        # Convolutional layers for raycast processing
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        
        # Fully connected layers after convolution
        # Calculate size after convolution to determine input size for FC layer
        conv_output_size = 32 * num_rays
        
        # Input for fully connected contains flattened conv output + additional features
        fc_input_size = conv_output_size + (input_size - num_rays)
        
        self.fc1 = nn.Linear(fc_input_size, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        # Split input into raycasts and other features
        raycasts = x[:, :self.num_rays]
        other_features = x[:, self.num_rays:]
        
        # Process raycasts with CNN
        raycasts = raycasts.unsqueeze(1)  # Add channel dimension [batch, 1, num_rays]
        x_conv = F.relu(self.conv1(raycasts))
        x_conv = F.relu(self.conv2(x_conv))
        x_conv = x_conv.flatten(1)  # Flatten CNN output
        
        # Combine with other features
        x_combined = torch.cat([x_conv, other_features], dim=1)
        
        # Process through fully connected layers
        x = F.relu(self.fc1(x_combined))
        x = self.fc2(x)
        
        return x


class LSTMModel(nn.Module):
    """
    LSTM-based model for sequence processing.
    Useful when considering temporal aspects of driving.
    
    Note: This requires sequence data preprocessing.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: [batch, sequence_length, features]
        lstm_out, _ = self.lstm(x)
        # Use only the last output for prediction
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output


class MultiInputModel(nn.Module):
    """
    Model with separate branches for different input types.
    Allows for specialized processing of raycasts and other features.
    """
    def __init__(self, raycast_size, other_feature_size):
        super(MultiInputModel, self).__init__()
        # Raycast processing branch
        self.raycast_branch = nn.Sequential(
            nn.Linear(raycast_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Other features branch (like speed)
        self.feature_branch = nn.Sequential(
            nn.Linear(other_feature_size, 8),
            nn.ReLU()
        )
        
        # Combined processing
        self.combined = nn.Sequential(
            nn.Linear(32 + 8, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # Split input
        raycasts = x[:, :-1]  # All but the last feature (assuming speed is last)
        other_features = x[:, -1].unsqueeze(1)  # Just the speed feature
        
        # Process through branches
        raycast_features = self.raycast_branch(raycasts)
        other_features = self.feature_branch(other_features)
        
        # Combine and process
        combined = torch.cat([raycast_features, other_features], dim=1)
        output = self.combined(combined)
        
        return output


class HybridModel(nn.Module):
    """
    Modèle hybride combinant CNN pour le traitement spatial des raycasts
    et LSTM pour capturer les dépendances temporelles.
    """
    def __init__(self, input_size, num_rays, seq_length=10):
        super(HybridModel, self).__init__()
        
        # Partie CNN pour traiter les raycasts
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(8),  # Réduire à une taille fixe
            nn.Flatten()
        )
        
        # Calculer la taille de sortie du CNN
        cnn_output_size = 32 * 8
        
        # LSTM pour traiter la séquence temporelle
        self.lstm = nn.LSTM(
            input_size=cnn_output_size + 1,  # +1 pour la vitesse
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Couches fully connected pour la sortie
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)  # 2 sorties: direction et accélération
        )
        
        self.num_rays = num_rays
        self.seq_length = seq_length
    
    def forward(self, x, hidden=None):
        # Check if input is 2D [batch, features] or 3D [batch, seq_len, features]
        if len(x.shape) == 2:
            # If 2D, add a sequence dimension (seq_len=1)
            x = x.unsqueeze(1)  # [batch, features] -> [batch, 1, features]
            
        # Now x is guaranteed to be 3D
        batch_size, seq_len, _ = x.shape
        
        # Traiter chaque pas de temps avec CNN
        cnn_outputs = []
        for t in range(seq_len):
            # Séparer raycasts et vitesse
            raycasts = x[:, t, :self.num_rays].unsqueeze(1)  # Ajouter dimension canal
            speed = x[:, t, -1].unsqueeze(1)  # Isoler la vitesse
            
            # Traiter raycasts avec CNN
            cnn_out = self.cnn(raycasts)
            
            # Concaténer avec vitesse
            combined = torch.cat([cnn_out, speed], dim=1)
            cnn_outputs.append(combined)
        
        # Empiler les sorties CNN pour former une séquence
        cnn_sequence = torch.stack(cnn_outputs, dim=1)
        
        # Traiter la séquence avec LSTM
        if hidden is None:
            lstm_out, hidden = self.lstm(cnn_sequence)
        else:
            lstm_out, hidden = self.lstm(cnn_sequence, hidden)
        
        # Prendre la dernière sortie de la séquence
        last_output = lstm_out[:, -1, :]
        
        # Passer par les couches fully connected
        output = self.fc(last_output)
        return output, hidden


def create_model(model_type, input_size, num_rays=None, **kwargs):
    """
    Factory function to create models based on type.
    
    Args:
        model_type (str): Type of model to create
        input_size (int): Size of input features
        num_rays (int): Number of raycasts (required for some models)
        
    Returns:
        nn.Module: Instantiated model
    """
    if model_type == 'simple':
        return SimpleModel(input_size)
    elif model_type == 'cnn':
        if num_rays is None:
            raise ValueError("num_rays must be specified for CNN model")
        return CNNModel(input_size, num_rays)
    elif model_type == 'lstm':
        # Note: For LSTM, preprocess data as sequences
        return LSTMModel(input_size)
    elif model_type == 'multi':
        # For multi-input, assume the last feature is speed
        raycast_size = input_size - 1  # Raycasts + other features except speed
        other_feature_size = 1  # Just speed
        return MultiInputModel(raycast_size, other_feature_size)
    elif model_type == "hybrid":
        num_rays = kwargs.get("num_rays", input_size - 1)  # Par défaut, tous sauf vitesse
        seq_length = kwargs.get("seq_length", 10)
        return HybridModel(input_size, num_rays, seq_length)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
