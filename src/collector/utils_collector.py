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
        # Appliquer une deadzone pour éviter les mouvements involontaires
        deadzone = 0.1
        
        # Inversion du mapping: axe 1 vertical pour la direction, axe 0 horizontal pour l'accélération
        steering_axis = joystick.get_axis(1)  # Maintenant l'axe 1 (vertical) pour la direction
        accel_axis = joystick.get_axis(0)     # Maintenant l'axe 0 (horizontal) pour l'accélération
        
        # Appliquer la deadzone
        if abs(steering_axis) < deadzone:
            steering_axis = 0.0
        if abs(accel_axis) < deadzone:
            accel_axis = 0.0
            
        # Mapping: axe vertical pour la direction
        steering = -steering_axis
        
        # Correction de l'axe d'accélération: négatif = avancer, positif = reculer
        # Sur la plupart des manettes, pousser le stick vers le bas donne une valeur positive
        acceleration = accel_axis  # Inverser l'axe horizontal comme on l'a fait avec l'axe vertical
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
