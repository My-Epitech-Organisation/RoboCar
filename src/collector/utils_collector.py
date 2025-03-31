"""
Fonctions utilitaires pour la collecte de données.

Ce module fournit:
- Des fonctions pour lire les entrées utilisateur (clavier ou manette)
- Des fonctions pour convertir ces entrées en commandes pour la voiture
"""

import pygame
import json
import os
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
    's': False,
    'c': False  # Ajout de la touche 'c' pour la calibration
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


class JoystickCalibration:
    """Classe pour gérer la calibration du joystick."""
    
    def __init__(self):
        self.calibration_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                           "joystick_calibration.json")
        self.calib_data = {
            "steering": {"min": -1.0, "max": 1.0},
            "acceleration": {"min": -1.0, "max": 1.0}
        }
        self.load_calibration()
        
    def load_calibration(self):
        """Charge les données de calibration depuis le fichier."""
        try:
            if os.path.exists(self.calibration_file):
                with open(self.calibration_file, 'r') as f:
                    self.calib_data = json.load(f)
                    print(f"[INFO] Calibration du joystick chargée: {self.calib_data}")
        except Exception as e:
            print(f"[AVERTISSEMENT] Erreur lors du chargement de la calibration: {e}")
            
    def save_calibration(self):
        """Enregistre les données de calibration dans un fichier."""
        try:
            with open(self.calibration_file, 'w') as f:
                json.dump(self.calib_data, f, indent=4)
                print(f"[INFO] Calibration du joystick enregistrée: {self.calib_data}")
        except Exception as e:
            print(f"[ERREUR] Impossible de sauvegarder la calibration: {e}")
    
    def apply_calibration(self, axis_value, axis_type):
        """
        Applique la calibration à une valeur d'axe.
        
        Args:
            axis_value: Valeur brute de l'axe
            axis_type: 'steering' ou 'acceleration'
            
        Returns:
            float: Valeur normalisée entre -1.0 et 1.0
        """
        min_val = self.calib_data[axis_type]["min"]
        max_val = self.calib_data[axis_type]["max"]
        
        # Éviter la division par zéro
        if min_val == max_val:
            return 0.0
            
        # Normalisation entre -1 et 1
        if axis_value >= 0:
            return axis_value / max_val if max_val != 0 else 0.0
        else:
            return axis_value / abs(min_val) if min_val != 0 else 0.0


# Création d'une instance globale de calibration
joystick_calibration = JoystickCalibration()


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
    
    # Vérifier que le joystick est toujours initialisé et actif
    try:
        if not joystick.get_init():
            try:
                joystick.init()
                print("[INFO] Joystick réinitialisé automatiquement")
            except pygame.error:
                return 0.0, 0.0
    except (pygame.error, AttributeError):
        # Le joystick a été déconnecté ou est en état d'erreur
        return 0.0, 0.0

    # Deadzone pour éviter les mouvements involontaires
    deadzone = 0.1

    try:
        # Axe 0 (horizontal) pour la direction, axe 1 (vertical) pour l'accélération
        steering_axis = joystick.get_axis(0)
        accel_axis = joystick.get_axis(1)
    except (pygame.error, IndexError):
        # En cas d'erreur lors de la lecture des axes
        print("[ERREUR] Impossible de lire les axes du joystick")
        return 0.0, 0.0

    # Application de la deadzone
    if abs(steering_axis) < deadzone:
        steering_axis = 0.0
    if abs(accel_axis) < deadzone:
        accel_axis = 0.0

    # Application de la calibration
    steering = joystick_calibration.apply_calibration(steering_axis, "steering")
    accel = -joystick_calibration.apply_calibration(accel_axis, "acceleration")  # Inversion maintenue

    return steering, accel


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
