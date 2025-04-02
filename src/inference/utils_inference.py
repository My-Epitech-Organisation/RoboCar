"""
Utility functions for autonomous driving inference.

This module provides:
- Helper functions for model loading and inference
- Observation processing and normalization
- Steering smoothing and performance monitoring
"""

import os
import json
import numpy as np
import torch
import time
import math


def load_model_metadata(model_path):
    """
    Load model metadata from saved checkpoint.

    Args:
        model_path (str): Path to the model checkpoint

    Returns:
        dict: Model metadata or None if not available
    """
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict) and 'metadata' in checkpoint:
            return checkpoint['metadata']
        return None
    except Exception as e:
        print(f"[WARNING] Could not load model metadata: {e}")
        return None


def smooth_steering(new_value, history, max_history=3, max_change=0.1):
    """
    Apply smoothing to steering predictions to reduce oscillations.

    Args:
        new_value (float): New steering prediction
        history (list): History of previous steering values
        max_history (int): Maximum history length
        max_change (float): Maximum allowed change per step

    Returns:
        float: Smoothed steering value
    """
    # Add new value to history
    history.append(new_value)

    # Limit history length
    if len(history) > max_history:
        history.pop(0)

    # Simple moving average
    avg_value = sum(history) / len(history)

    # Limit rate of change if previous values exist
    if len(history) > 1:
        prev_value = history[-2]
        change = avg_value - prev_value
        if abs(change) > max_change:
            # Limit change to max_change
            limited_change = max_change if change > 0 else -max_change
            avg_value = prev_value + limited_change

    return avg_value


def normalize_observations(obs_array, num_rays, max_raycast=260.0, max_speed=70.0):
    """
    Normalize observation values to match training data normalization.

    Args:
        obs_array (np.ndarray): Raw observation array
        num_rays (int): Number of raycasts
        max_raycast (float): Maximum raycast value for normalization
        max_speed (float): Maximum speed value for normalization

    Returns:
        np.ndarray: Normalized features array for model input
    """
    # Extract and normalize raycasts
    raycasts = obs_array[:num_rays]
    raycasts_normalized = np.clip(raycasts / max_raycast, 0, 1)

    # Extract and normalize speed
    speed = float(obs_array[-5]) if len(obs_array) >= 5 else 0.0
    speed_normalized = min(max(speed / max_speed, 0.0), 1.0)

    # Combine features - always include speed as this is now required by our model
    features = np.concatenate([raycasts_normalized, [speed_normalized]])

    return features, raycasts, speed


class PerformanceMonitor:
    """Class to monitor inference performance."""

    def __init__(self, log_interval=100):
        """
        Initialize performance monitor.

        Args:
            log_interval (int): Frames between logging performance stats
        """
        self.frame_count = 0
        self.log_interval = log_interval
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.inference_times = []

    def start_frame(self):
        """Mark the start of a frame processing."""
        self.frame_start_time = time.time()

    def end_frame(self):
        """
        Mark the end of a frame processing and log performance.

        Returns:
            float: Frame processing time in milliseconds
        """
        frame_time = (time.time() - self.frame_start_time) * 1000  # ms
        self.inference_times.append(frame_time)
        self.frame_count += 1

        # Log performance periodically
        if self.frame_count % self.log_interval == 0:
            self._log_performance()

        return frame_time

    def _log_performance(self):
        """Log performance metrics."""
        current_time = time.time()
        elapsed = current_time - self.last_log_time
        fps = self.log_interval / elapsed if elapsed > 0 else 0

        avg_inference = np.mean(self.inference_times[-self.log_interval:])
        max_inference = np.max(self.inference_times[-self.log_interval:])

        print(f"[PERF] FPS: {fps:.1f}, "
              f"Avg inference: {avg_inference:.2f}ms, "
              f"Max inference: {max_inference:.2f}ms")

        self.last_log_time = current_time
        self.inference_times = self.inference_times[-self.log_interval:]


def check_model_compatibility(model_input_size, num_rays):
    """
    Check if the model is compatible with the current configuration.

    Args:
        model_input_size (int): Model input size
        num_rays (int): Current number of rays from config

    Returns:
        bool: True if compatible, False otherwise
    """
    expected_input_size = num_rays + 1  # Rays + speed

    if model_input_size != expected_input_size:
        print(f"[WARNING] Model input size ({model_input_size}) does not match "
              f"expected size ({expected_input_size}) for {num_rays} rays")
        return False

    return True


class AdvancedSteeringController:
    """
    Contrôleur de direction avancé avec:
    - Lissage exponentiel
    - Contraintes physiques de direction
    - Adaptation à la vitesse
    """
    def __init__(self, history_size=10, alpha=0.3):
        self.history = []
        self.history_size = history_size
        self.alpha = alpha  # Facteur de lissage exponentiel
        self.last_steering = 0.0

    def update(self, predicted_steering, current_speed, dt=0.05):
        """
        Met à jour la commande de direction en appliquant un lissage
        et des contraintes adaptées à la vitesse.

        Args:
            predicted_steering: Prédiction brute du modèle
            current_speed: Vitesse actuelle du véhicule
            dt: Intervalle de temps entre deux prédictions
        """
        # Ajouter à l'historique
        self.history.append(predicted_steering)
        if len(self.history) > self.history_size:
            self.history.pop(0)

        # Lissage exponentiel
        smoothed = self.last_steering
        for steer in self.history:
            smoothed = self.alpha * steer + (1 - self.alpha) * smoothed

        # Limiter le taux de changement basé sur la vitesse
        # Plus la vitesse est élevée, plus les changements sont progressifs
        max_change_rate = 2.0  # radians/seconde
        speed_factor = 1.0 / (1.0 + 0.1 * current_speed)  # Ralentir les changements à haute vitesse
        max_change = max_change_rate * dt * speed_factor

        # Appliquer la limite de taux de changement
        delta = smoothed - self.last_steering
        if abs(delta) > max_change:
            delta = math.copysign(max_change, delta)

        # Calculer la nouvelle valeur de direction
        new_steering = self.last_steering + delta

        # Stocker pour la prochaine itération
        self.last_steering = new_steering

        return new_steering


class AccelerationController:
    """
    Contrôleur d'accélération intelligent qui ajuste la vitesse
    en fonction des conditions de la piste.
    """
    def __init__(self, max_speed=1.0, caution_distance=0.5):
        self.max_speed = max_speed
        self.caution_distance = caution_distance
        self.last_accel = 0.0
        self.speed_history = []

    def compute_acceleration(self, predicted_accel, raycasts, steering_angle):
        """
        Calcule l'accélération appropriée basée sur la prédiction du modèle,
        les distances des raycasts et l'angle de direction.
        """
        # Trouver la distance minimale devant le véhicule
        forward_rays = raycasts[len(raycasts)//4:3*len(raycasts)//4]  # Rayons centraux (devant)
        min_distance = np.min(forward_rays) if len(forward_rays) > 0 else 1.0
        
        # Facteur de prudence basé sur la proximité d'obstacles
        caution_factor = min(1.0, min_distance / self.caution_distance)
        
        # Facteur de virage - réduire l'accélération dans les virages serrés
        turn_factor = 1.0 - min(1.0, abs(steering_angle) / 0.5)
        
        # Déterminer si nous devons freiner (valeur négative) ou accélérer (valeur positive)
        should_brake = False
        
        # Freiner si:
        # 1. Obstacle proche devant
        if min_distance < self.caution_distance * 0.5:
            should_brake = True
        # 2. Virage serré à haute vitesse
        elif abs(steering_angle) > 0.6:
            should_brake = True
        # 3. La prédiction du modèle est négative (le modèle a prédit un freinage)
        elif predicted_accel < 0:
            should_brake = True
        
        # Ajuster l'intensité de l'accélération ou du freinage
        if should_brake:
            # Valeur négative pour le freinage, plus forte si obstacle proche
            brake_intensity = -0.5 - (1.0 - caution_factor) * 0.5
            adjusted_accel = max(brake_intensity, predicted_accel)
        else:
            # Valeur positive pour l'accélération, modulée par les facteurs
            adjusted_accel = predicted_accel * caution_factor * turn_factor
        
        # Lisser les changements d'accélération
        smooth_accel = 0.8 * adjusted_accel + 0.2 * self.last_accel
        self.last_accel = smooth_accel
        
        return smooth_accel
