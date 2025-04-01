"""
Data preprocessing module for RoboCar neural network training.

This module:
1. Loads CSV data collected from the simulator
2. Parses raycasts and other features
3. Normalizes and prepares data for training
4. Handles data augmentation and splitting
"""

import os
import ast
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_session(file_path):
    """
    Load a single session CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded and parsed dataframe
    """
    print(f"Loading data from {file_path}")
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded {len(data)} samples")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_all_sessions(data_dir):
    """
    Load all CSV files from a directory.
    
    Args:
        data_dir (str): Directory containing CSV files
        
    Returns:
        pd.DataFrame: Combined dataframe of all sessions
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files in {data_dir}")
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    all_data = []
    for file in csv_files:
        data = load_session(file)
        if data is not None:
            all_data.append(data)
    
    if not all_data:
        raise ValueError("Failed to load any valid data")
    
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Combined dataset contains {len(combined_data)} samples")
    return combined_data


def parse_raycasts(data):
    """
    Parse raycast strings into numerical arrays.
    
    Args:
        data (pd.DataFrame): Dataframe with raycast column
        
    Returns:
        pd.DataFrame: Dataframe with parsed raycasts
    """
    # Create a copy to avoid modifying the original
    processed_data = data.copy()
    
    # Parse raycast strings to numeric arrays
    processed_data['raycasts_parsed'] = processed_data['raycasts'].apply(
        lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array([])
    )
    
    # Filter out rows with empty raycasts
    valid_mask = processed_data['raycasts_parsed'].apply(lambda x: len(x) > 0)
    processed_data = processed_data[valid_mask]
    
    # Get the number of rays
    if len(processed_data) > 0:
        num_rays = len(processed_data.iloc[0]['raycasts_parsed'])
        print(f"Raycast data has {num_rays} rays per sample")
    
    return processed_data


def normalize_data(data):
    """
    Normalize features to [0, 1] range.
    
    Args:
        data (pd.DataFrame): Dataframe with features
        
    Returns:
        pd.DataFrame: Normalized dataframe
    """
    # Create a copy to avoid modifying the original
    normalized_data = data.copy()
    
    # Normalize raycasts to [0, 1] range
    max_raycast_value = 20.0  # Max distance observable by raycast
    
    normalized_data['raycasts_normalized'] = normalized_data['raycasts_parsed'].apply(
        lambda rays: np.clip(rays / max_raycast_value, 0, 1)
    )
    
    # Normalize speed
    speed_scaler = MinMaxScaler()
    normalized_data['speed_normalized'] = speed_scaler.fit_transform(
        normalized_data[['speed']]
    )
    
    # Steering is already in [-1, 1] range
    
    return normalized_data


def generate_features(data):
    """
    Generate additional features for training.
    
    Args:
        data (pd.DataFrame): Processed dataframe
        
    Returns:
        tuple: X (features) and y (targets) for training
    """
    # Extract raycast features
    X_raycasts = np.stack(data['raycasts_normalized'].values)
    
    # Additional features
    X_speed = data['speed_normalized'].values.reshape(-1, 1)
    
    # Combine features
    X = np.hstack((X_raycasts, X_speed))
    
    # Target is steering_input (what the user did)
    y = data['steering_input'].values
    
    print(f"Generated feature array with shape {X.shape} and target array with shape {y.shape}")
    return X, y


def augment_data(X, y, mirror_prob=0.5, noise_prob=0.3, noise_scale=0.05):
    """
    Augment training data with mirroring and noise.
    
    Args:
        X (np.ndarray): Feature array
        y (np.ndarray): Target array
        mirror_prob (float): Probability of mirroring
        noise_prob (float): Probability of adding noise
        noise_scale (float): Scale of noise to add
        
    Returns:
        tuple: Augmented X and y arrays
    """
    print("Augmenting data...")
    
    # Original data
    X_augmented = [X]
    y_augmented = [y]
    
    # Mirror data (flip raycasts and reverse steering)
    if mirror_prob > 0:
        num_rays = X.shape[1] - 1  # Subtract 1 for speed feature
        
        # Create mirrored version (raycasts and steering)
        X_mirrored = X.copy()
        X_mirrored[:, :num_rays] = np.flip(X_mirrored[:, :num_rays], axis=1)
        y_mirrored = -y.copy()
        
        X_augmented.append(X_mirrored)
        y_augmented.append(y_mirrored)
    
    # Add noise
    if noise_prob > 0:
        # Add noise to raycasts
        num_rays = X.shape[1] - 1  # Subtract 1 for speed feature
        
        X_noisy = X.copy()
        noise = np.random.normal(0, noise_scale, size=X_noisy[:, :num_rays].shape)
        X_noisy[:, :num_rays] = np.clip(X_noisy[:, :num_rays] + noise, 0, 1)
        
        X_augmented.append(X_noisy)
        y_augmented.append(y.copy())
    
    # Combine all augmented datasets
    X_final = np.vstack(X_augmented)
    y_final = np.concatenate(y_augmented)
    
    print(f"Data augmented from {len(X)} to {len(X_final)} samples")
    return X_final, y_final


def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split data into training, validation, and test sets.
    
    Args:
        X (np.ndarray): Feature array
        y (np.ndarray): Target array
        test_size (float): Proportion of data for testing
        val_size (float): Proportion of remaining data for validation
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split out test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Then split the remaining data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )
    
    print(f"Data split: {len(X_train)} training, {len(X_val)} validation, {len(X_test)} test samples")
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_data(data, normalize=True, augment=True):
    """
    Full preprocessing pipeline.
    
    Args:
        data (pd.DataFrame): Raw data from CSV
        normalize (bool): Whether to normalize data
        augment (bool): Whether to augment data
        
    Returns:
        tuple: X (features) and y (targets) ready for training
    """
    # Parse raycasts
    processed_data = parse_raycasts(data)
    
    # Normalize if requested
    if normalize:
        processed_data = normalize_data(processed_data)
    
    # Generate features and targets
    X, y = generate_features(processed_data)
    
    # Augment if requested
    if augment:
        X, y = augment_data(X, y)
    
    return X, y


def get_num_rays_from_config(project_root):
    """
    Get number of rays from agent configuration.
    
    Args:
        project_root (str): Path to project root
        
    Returns:
        int: Number of rays
    """
    config_path = os.path.join(project_root, "config", "agent_config.json")
    try:
        with open(config_path, 'r') as f:
            import json
            config = json.load(f)
            num_rays = config["agents"][0]["nbRay"]
            print(f"Loaded ray count from config: {num_rays}")
            return num_rays
    except Exception as e:
        print(f"Error loading ray count from config: {e}")
        print("Using default ray count: 10")
        return 10
