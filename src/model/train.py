"""
src/model/train.py
Lit train_config.yaml pour les hyperparamètres,
charge le dataset, entraîne le modèle, puis le sauvegarde.
"""

import os
import torch
import yaml
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from .model import MLPController
from .utils_training import DrivingDataset, collate_fn

def main():
    # 1) Charger la config d'entraînement (train_config.yaml)
    config_path = os.path.join("..", "..", "..", "config", "train_config.yaml")
    with open(config_path, 'r') as f:
        train_config = yaml.safe_load(f)

    # Extraction des paramètres
    input_size = train_config["training"]["model"]["input_size"]
    hidden_size = train_config["training"]["model"]["hidden_size"]
    output_size = train_config["training"]["model"]["output_size"]

    batch_size = train_config["training"]["hyperparams"]["batch_size"]
    learning_rate = train_config["training"]["hyperparams"]["learning_rate"]
    epochs = train_config["training"]["hyperparams"]["epochs"]

    data_processed_path = train_config["paths"]["data_processed"]
    model_output_path = train_config["paths"]["model_output"]

    # 2) Charger le dataset
    train_csv = os.path.join(data_processed_path, "train.csv")
    df = pd.read_csv(train_csv)
    dataset = DrivingDataset(df)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 3) Instancier le modèle et définir l'optimiseur
    model = MLPController(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4) Boucle d'entraînement
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    # 5) Sauvegarde du modèle
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    torch.save(model.state_dict(), model_output_path)
    print(f"Modèle sauvegardé : {model_output_path}")

if __name__ == "__main__":
    main()
