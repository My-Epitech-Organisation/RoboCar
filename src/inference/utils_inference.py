"""
Fonctions utilitaires pour l'inférence.
"""

import torch

def preprocess_observation(obs):
    """
    Convertit l'observation brute (liste numpy) en un tenseur PyTorch shape [1, input_size].
    """
    if not hasattr(obs, "__len__"):
        obs = [obs]
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    return obs_tensor

def postprocess_action(action_tensor):
    """
    Convertit la sortie du réseau en deux valeurs (steering, accel).
    action_tensor: shape [1, 2]
    """
    action_np = action_tensor.cpu().numpy()[0]
    steering, accel = float(action_np[0]), float(action_np[1])
    return steering, accel
