"""
Fonctions utilitaires pour la collecte de données.
Gère la lecture des entrées depuis clavier ou manette, etc.
"""

import pygame

def parse_user_input(joystick=None):
    """
    Lit les entrées de commande (volant, accélération).
    Retourne un tuple (steering, acceleration).
    """
    # Valeurs par défaut
    steering = 0.0
    acceleration = 0.0

    # Si joystick/manette présent
    if joystick is not None:
        # Exemple: steering sur axe 0, acceleration sur axe 1
        steering = joystick.get_axis(0)
        # On peut considérer qu'un axe négatif = frein, positif = accélération
        accel_axis = joystick.get_axis(1)
        acceleration = -accel_axis  # inversion si besoin
    else:
        # Sinon, lecture du clavier
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            steering = -1.0
        elif keys[pygame.K_RIGHT]:
            steering = 1.0
        
        if keys[pygame.K_UP]:
            acceleration = 1.0
        elif keys[pygame.K_DOWN]:
            acceleration = -1.0

    return steering, acceleration
