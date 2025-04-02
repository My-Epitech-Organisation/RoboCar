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
import random
import json
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

    # Target includes both steering and acceleration inputs
    y_steering = data['steering_input'].values.reshape(-1, 1)

    # Add acceleration input if available, otherwise use zeros
    if 'acceleration_input' in data.columns:
        y_accel = data['acceleration_input'].values.reshape(-1, 1)
    else:
        y_accel = np.zeros_like(y_steering)

    # Combine targets
    y = np.hstack((y_steering, y_accel))

    print(f"Generated feature array with shape {X.shape} and target array with shape {y.shape}")
    return X, y


def augment_data(X, y, noise_level=0.05, mirror_prob=0.5):
    """
    Augmentation améliorée des données avec plusieurs techniques:
    - Ajout de bruit aléatoire
    - Effet miroir (inversion gauche-droite)
    - Variation légère de la vitesse
    """
    augmented_X, augmented_y = [], []

    # Vérifier si y est 1D ou 2D
    if len(y.shape) == 1 or y.shape[1] == 1:
        # Si y est 1D (juste steering), le convertir en 2D
        y = y.reshape(-1, 1)
        # Ajouter une colonne d'accélération avec des zéros
        y = np.hstack((y, np.zeros((y.shape[0], 1))))
        print("Converted 1D target array to 2D with zeros for acceleration")

    # Ajouter les données originales
    augmented_X.extend(X)
    augmented_y.extend(y)

    # Miroir des données (inversion gauche-droite)
    for i in range(len(X)):
        if random.random() < mirror_prob:
            # Supposons que X[i] contient [raycast_1, raycast_2, ..., raycast_n, speed]
            # Et que les raycasts sont symétriques (le premier à gauche, le dernier à droite)
            num_rays = len(X[i]) - 1  # Soustrait 1 pour la vitesse
            mirrored_input = X[i].copy()

            # Inverser l'ordre des raycasts
            mirrored_input[:num_rays] = mirrored_input[:num_rays][::-1]

            # Inverser la direction
            mirrored_output = np.array([-y[i][0], y[i][1]])  # Inverser steering, garder acceleration

            augmented_X.append(mirrored_input)
            augmented_y.append(mirrored_output)

    # Ajout de bruit aléatoire
    for i in range(len(X)):
        if random.random() < 0.3:  # 30% de chance d'ajouter du bruit
            noisy_input = X[i].copy()
            # Ajouter du bruit aux raycasts uniquement
            for j in range(len(noisy_input) - 1):  # Tous sauf vitesse
                noise = random.uniform(-noise_level, noise_level)
                noisy_input[j] = max(0, min(1, noisy_input[j] + noise))  # Garder entre 0 et 1

            augmented_X.append(noisy_input)
            augmented_y.append(y[i])

    return np.array(augmented_X), np.array(augmented_y)


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


def preprocess_data(data, normalize=True, use_only_raycasts=False):
    """
    Prétraite les données de conduite.

    Args:
        data (DataFrame): Données brutes
        normalize (bool): Si True, normalise les raycasts
        use_only_raycasts (bool): Si True, n'utilise que les raycasts comme entrées

    Returns:
        X (ndarray): Caractéristiques d'entrée
        y (ndarray): Cibles (direction, accélération)
    """
    # Extraire les raycasts
    X_raycasts = []
    for raycast_str in data['raycasts']:
        rays = np.array(json.loads(raycast_str))
        X_raycasts.append(rays)

    X_raycasts = np.array(X_raycasts)

    # Normaliser les raycasts si demandé
    if normalize:
        # Normalisation des raycasts (limiter entre 0-1)
        max_raycast = 260.0  # Distance maximale des raycasts
        X_raycasts = np.clip(X_raycasts / max_raycast, 0, 1)

    # Si on utilise uniquement les raycasts
    if use_only_raycasts:
        X = X_raycasts
    else:
        # Combiner avec d'autres caractéristiques (vitesse, etc.)
        X_speed = np.array(data['speed']).reshape(-1, 1)
        X = np.hstack((X_raycasts, X_speed))

    # Extraire les sorties: direction et accélération
    y_steering = np.array(data['steering_input']).reshape(-1, 1)
    y_accel = np.array(data['acceleration_input']).reshape(-1, 1)
    y = np.hstack((y_steering, y_accel))

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
            config = json.load(f)
            num_rays = config["agents"][0]["nbRay"]
            print(f"Loaded ray count from config: {num_rays}")
            return num_rays
    except Exception as e:
        print(f"Error loading ray count from config: {e}")
        print("Using default ray count: 10")
        return 10
