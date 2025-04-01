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


def normalize_observations(obs_array, num_rays, max_raycast=20.0, max_speed=30.0):
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
    
    # Combine features
    features = np.concatenate([raycasts_normalized, [speed_normalized]])
    
    return features


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
