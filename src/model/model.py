"""
Définition du réseau de neurones (MLP) en PyTorch pour prédire l'angle et/ou l'accélération.
"""

import torch
import torch.nn as nn

class MLPController(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, output_size=2):
        """
        input_size: nombre de features en entrée (raycasts + vitesse + angle précédent, etc.)
        output_size: 2 si on prédit (steering, acceleration)
        """
        super(MLPController, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)
