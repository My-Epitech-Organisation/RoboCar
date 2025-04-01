"""
Utility functions for model inference.

This module provides:
- Functions to preprocess observations for model input
- Functions to postprocess model outputs for simulation control
"""

import torch
import numpy as np
import sys
import os
import pickle
import json
from pathlib import Path

# Mean and std values for normalization - will be loaded from file
scaler_means = None
scaler_stds = None

def load_scaler_values():
    """Load standardization values created during training"""
    global scaler_means, scaler_stds
    
    # Try to load scaler values if they exist
    project_root = Path(__file__).resolve().parents[2]
    scaler_path = project_root / "model" / "scaler_values.json"
    
    if scaler_path.exists():
        try:
            with open(scaler_path, "r") as f:
                scaler_data = json.load(f)
                scaler_means = np.array(scaler_data["means"])
                scaler_stds = np.array(scaler_data["stds"])
                print(f"[INFO] Loaded normalization values from {scaler_path}")
                return True
        except Exception as e:
            print(f"[WARNING] Could not load scaler values: {e}")
    
    print("[WARNING] Using default normalization - this may affect model performance!")
    # Default values (better than nothing, but less accurate)
    # These values are approximate based on common ranges in the dataset
    dummy_features = 15  # steering_input, accel_input, 10 raycasts, pos_x, pos_y, pos_z
    scaler_means = np.zeros(dummy_features)
    scaler_stds = np.ones(dummy_features)
    return False


def preprocess_observation(obs_array, num_rays=10):
    """
    Preprocess observations for model input.
    
    Args:
        obs_array: Raw observation array from environment
        num_rays: Number of raycasts
        
    Returns:
        Tensor: Preprocessed tensor for model input
    """
    # Load scaler values if not already loaded
    if scaler_means is None:
        load_scaler_values()
        
    try:
        # Extract raycasts (first 'num_rays' elements)
        raycasts = obs_array[:num_rays].tolist()
        
        # Extract additional observations
        try:
            speed = float(obs_array[-5]) if len(obs_array) >= 5 else 0.0
            obs_steering = float(obs_array[-4]) if len(obs_array) >= 4 else 0.0
            position_x = float(obs_array[-3]) if len(obs_array) >= 3 else 0.0
            position_y = float(obs_array[-2]) if len(obs_array) >= 2 else 0.0
            position_z = float(obs_array[-1]) if len(obs_array) >= 1 else 0.0
        except IndexError:
            speed, obs_steering = 0.0, 0.0
            position_x, position_y, position_z = 0.0, 0.0, 0.0
        
        # Create features array matching what model expects from training:
        # [steering_input, acceleration_input, 10 raycasts, position_x, position_y, position_z]
        features = [
            obs_steering,  # Use current steering as input
            speed,         # Use current speed as accel input
            *raycasts,     # Raycasts from environment
            position_x, position_y, position_z  # Position data
        ]
        
        # Normalize features using the same scaling as training
        features_np = np.array(features, dtype=np.float32)
        normalized_features = (features_np - scaler_means) / scaler_stds
        
        # Print diagnostic info occasionally
        if np.random.random() < 0.01:  # Print ~1% of the time
            print(f"[DEBUG] Raw features: {features_np[:3]}...")
            print(f"[DEBUG] Normalized: {normalized_features[:3]}...")
        
        # Convert to tensor with batch dimension
        input_tensor = torch.tensor(normalized_features, dtype=torch.float32).unsqueeze(0)
        return input_tensor
    
    except Exception as e:
        print(f"[ERROR] Error preprocessing observation: {e}")
        # Return a zeros tensor as fallback
        return torch.zeros((1, 15), dtype=torch.float32)


def postprocess_action(action_tensor):
    """
    Convert model output to simulation actions.
    
    Args:
        action_tensor: Raw model output tensor [batch, 2]
        
    Returns:
        tuple: (steering, acceleration) values for simulation
    """
    try:
        # Extract values from tensor
        action_np = action_tensor.cpu().numpy()[0]
        
        # Diagnostic output (every ~100 frames)
        if np.random.random() < 0.01:
            print(f"[DEBUG] Raw model output: {action_np}")
        
        # The model is trained to predict [speed, steering]
        # But we need to return [steering, acceleration]
        predicted_speed = float(action_np[0])
        predicted_steering = float(action_np[1])
        
        # Use predicted steering directly
        steering = predicted_steering
        
        # For acceleration, we need to determine the appropriate value
        # This is a simple approach - you might want to develop a more sophisticated controller
        accel = 0.5  # Default moderate acceleration
        
        # Reduce acceleration in turns
        if abs(steering) > 0.3:
            accel = 0.3  # Slow down in sharper turns
        
        # Clip values to valid range
        steering = max(min(steering, 1.0), -1.0)
        accel = max(min(accel, 1.0), -0.2)  # Limit negative acceleration (braking)
        
        return steering, accel
    
    except Exception as e:
        print(f"[ERROR] Error postprocessing action: {e}")
        return 0.0, 0.0  # Return neutral values as fallback
