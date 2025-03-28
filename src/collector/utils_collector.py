"""
Fonctions utilitaires pour la collecte de données.

Ce module fournit:
- Des fonctions pour lire les entrées utilisateur (clavier ou manette)
- Des fonctions pour convertir ces entrées en commandes pour la voiture
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
        if k in key_states:
            key_states[k] = True
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
        if k in key_states:
            key_states[k] = False
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


def get_keyboard_input():
    """
    Lit les entrées clavier et retourne les commandes correspondantes.

    Returns:
        tuple: (steering, acceleration) - Valeurs entre -1.0 et 1.0
    """
    steering = 0.0
    acceleration = 0.0

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


def get_joystick_input(joystick):
    """
    Lit les entrées du joystick et retourne les commandes correspondantes.

    Args:
        joystick: Instance du joystick Pygame

    Returns:
        tuple: (steering, acceleration) - Valeurs entre -1.0 et 1.0
    """
    if joystick is None:
        return 0.0, 0.0

    # Deadzone pour éviter les mouvements involontaires
    deadzone = 0.1

    # Axe 0 (horizontal) pour la direction, axe 1 (vertical) pour l'accélération
    steering_axis = joystick.get_axis(0)
    accel_axis = joystick.get_axis(1)

    # Application de la deadzone
    if abs(steering_axis) < deadzone:
        steering_axis = 0.0
    if abs(accel_axis) < deadzone:
        accel_axis = 0.0

    # Inversion de l'axe d'accélération pour correspondre à l'intuition
    return steering_axis, -accel_axis


def parse_user_input(joystick=None):
    """
    Lit les entrées utilisateur et les convertit en commandes de direction.

    Args:
        joystick: Instance du joystick Pygame (None pour utiliser le clavier)

    Returns:
        tuple: (steering, acceleration) - Valeurs entre -1.0 et 1.0
    """
    # Par défaut, utiliser les entrées clavier
    steering, acceleration = get_keyboard_input()

    # Si un joystick est disponible et que le clavier n'est pas utilisé,
    # utiliser les entrées du joystick
    if joystick is not None and steering == 0.0 and acceleration == 0.0:
        steering, acceleration = get_joystick_input(joystick)

    return steering, acceleration
