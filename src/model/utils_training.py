"""
Fonctions utilitaires pour l'entraînement (Dataset, collate_fn, etc.)
"""

import torch
from torch.utils.data import Dataset

class DrivingDataset(Dataset):
    """
    Dataset personnalisé pour la conduite.
    Suppose un DataFrame avec colonnes:
    ["raycasts", "speed", "steering_input", "acceleration_input", ...]
    """
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Exemple: conversion de la colonne "raycasts" (string) en liste/array
        # A adapter selon votre format
        raycasts = eval(row["raycasts"])  # attention à eval, mieux vaut un parsing plus sûr
        speed = row.get("speed", 0.0)
        
        # Features
        inputs = raycasts + [speed]  # liste ou array
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

        # Targets (steering, accel)
        steering = row["steering_input"]
        accel = row["acceleration_input"]
        targets_tensor = torch.tensor([steering, accel], dtype=torch.float32)

        return inputs_tensor, targets_tensor

def collate_fn(batch):
    """
    Agrège les échantillons en un seul batch.
    batch: liste de tuples (inputs_tensor, targets_tensor)
    """
    inputs = []
    targets = []

    for inp, tgt in batch:
        inputs.append(inp)
        targets.append(tgt)
    
    inputs = torch.stack(inputs, dim=0)   # [B, input_size]
    targets = torch.stack(targets, dim=0) # [B, output_size]

    return inputs, targets
