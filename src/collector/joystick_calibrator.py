"""
Module for joystick calibration.

This module provides a graphical interface to calibrate the joystick
and save the minimum/maximum values for each axis.
"""

import pygame
import time
# Use absolute import instead of relative import
from src.collector.ui_components import JoystickCalibratorUI
from src.collector.utils_collector import joystick_calibration


def calibrate_joystick(joystick):
    """
    Launch the calibration interface for a joystick.

    Args:
        joystick: Pygame joystick instance to calibrate

    Returns:
        bool: True if calibration was completed, False otherwise
    """
    if not joystick:
        print("[ERROR] No joystick to calibrate")
        return False

    # Save joystick info for verification
    try:
        joystick_name = joystick.get_name()
        joystick_id = joystick.get_id()
        print(f"[INFO] Calibrating joystick: {joystick_name} (ID: {joystick_id})")
    except (pygame.error, AttributeError):
        print("[WARNING] Unable to get joystick info")

    # Launch calibration
    calibrator = JoystickCalibratorUI(joystick)
    result = calibrator.run()

    print("[INFO] Calibration completed, cleaning up pygame events...")

    # Ensure pygame is still initialized
    if not pygame.get_init():
        pygame.init()

    # Clear event queue completely
    pygame.event.clear()

    # Let pygame handle events
    for _ in range(20):
        pygame.event.pump()
        pygame.time.delay(10)

    return result
