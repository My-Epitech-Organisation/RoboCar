"""
Fonctions utilitaires pour la collecte de données.

Ce module contient les fonctions nécessaires pour:
- Lire les entrées utilisateur (clavier ou manette)
- Convertir ces entrées en commandes pour la voiture
"""

import pygame


def parse_user_input(joystick=None):
    """
    Lit les entrées utilisateur et les convertit en commandes de direction.

    Args:
        joystick: Instance du joystick Pygame (None pour utiliser le clavier)

    Returns:
        tuple: (steering, acceleration) - Valeurs entre -1.0 et 1.0
    """
    # Valeurs par défaut
    steering = 0.0
    acceleration = 0.0

    # Lecture des entrées avec manette
    if joystick is not None:
        # Deadzone pour éviter les mouvements involontaires
        deadzone = 0.1

        # Axe 1 (vertical) pour la direction, axe 0 (horizontal) pour l'accélération
        steering_axis = joystick.get_axis(1)
        accel_axis = joystick.get_axis(0)

        # Application de la deadzone
        if abs(steering_axis) < deadzone:
            steering_axis = 0.0
        if abs(accel_axis) < deadzone:
            accel_axis = 0.0

        # Inversion de l'axe de direction pour correspondre à l'intuition
        steering = -steering_axis
        acceleration = accel_axis
    else:
        # Lecture des entrées clavier
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
