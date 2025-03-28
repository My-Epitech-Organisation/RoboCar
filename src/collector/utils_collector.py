"""
Fonctions utilitaires pour la collecte de données.

Ce module contient les fonctions nécessaires pour:
- Lire les entrées utilisateur (clavier ou manette)
- Convertir ces entrées en commandes pour la voiture
"""

import pygame
from pynput import keyboard

# Variables globales pour stocker l'état des touches
key_states = {
    'left': False,
    'right': False,
    'up': False,
    'down': False,
    'q': False,
    'd': False,
    'z': False,
    's': False
}

def on_press(key):
    """Callback quand une touche est pressée."""
    try:
        k = key.char.lower()  # Conversion en minuscule
        if k == 'q':
            key_states['q'] = True
        elif k == 'd':
            key_states['d'] = True
        elif k == 'z':
            key_states['z'] = True
        elif k == 's':
            key_states['s'] = True
    except AttributeError:
        # Touches spéciales
        if key == keyboard.Key.left:
            key_states['left'] = True
        elif key == keyboard.Key.right:
            key_states['right'] = True
        elif key == keyboard.Key.up:
            key_states['up'] = True
        elif key == keyboard.Key.down:
            key_states['down'] = True

def on_release(key):
    """Callback quand une touche est relâchée."""
    try:
        k = key.char.lower()  # Conversion en minuscule
        if k == 'q':
            key_states['q'] = False
        elif k == 'd':
            key_states['d'] = False
        elif k == 'z':
            key_states['z'] = False
        elif k == 's':
            key_states['s'] = False
    except AttributeError:
        # Touches spéciales
        if key == keyboard.Key.left:
            key_states['left'] = False
        elif key == keyboard.Key.right:
            key_states['right'] = False
        elif key == keyboard.Key.up:
            key_states['up'] = False
        elif key == keyboard.Key.down:
            key_states['down'] = False

def setup_keyboard_listener():
    """Configure et retourne un écouteur de clavier."""
    return keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )

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
        steering_axis = joystick.get_axis(0)
        accel_axis = joystick.get_axis(1)

        # Application de la deadzone
        if abs(steering_axis) < deadzone:
            steering_axis = 0.0
        if abs(accel_axis) < deadzone:
            accel_axis = 0.0

        # Inversion de l'axe de direction pour correspondre à l'intuition
        steering = steering_axis
        acceleration = -accel_axis
    
    # Lecture des entrées clavier depuis les variables globales
    # Contrôles ZQSD explicites 
    if key_states['q']:
        steering = -1.0  # Q = Gauche
    elif key_states['d']:
        steering = 1.0   # D = Droite
        
    if key_states['z']:
        acceleration = 1.0  # Z = Avancer
    elif key_states['s']:
        acceleration = -1.0  # S = Reculer
        
    # Contrôles flèches comme alternative
    if key_states['left']:
        steering = -1.0  # Flèche gauche = Gauche
    elif key_states['right']:
        steering = 1.0   # Flèche droite = Droite
        
    if key_states['up']:
        acceleration = 1.0  # Flèche haut = Avancer
    elif key_states['down']:
        acceleration = -1.0  # Flèche bas = Reculer

    return steering, acceleration
