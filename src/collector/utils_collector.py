"""
Utility functions for data collection.

This module provides:
- Functions to read user inputs (keyboard or controller)
- Functions to convert these inputs to car commands
"""

import os
import json
import pygame
from pynput import keyboard

# Global variables to store key states
key_states = {
    'left': False,
    'right': False,
    'up': False,
    'down': False,
    'q': False,
    'd': False,
    'z': False,
    's': False,
    'c': False,  # For calibration
    'v': False,  # For wheel mode toggle
    'i': False   # For input selector
}

# Global variable to track current joystick control mode
# False = dual stick mode (default, left=steering, right=acceleration)
# True = single stick mode (original, left stick for both)
single_stick_mode = False
last_y_button_state = False

# Global variable to track wheel mode
wheel_mode = False

# Global variables for custom input configuration
steering_device = None
acceleration_device = None
show_input_selector = False


def on_press(key):
    """Callback when a key is pressed."""
    try:
        k = key.char.lower()
        if k in key_states:
            key_states[k] = True
    except AttributeError:
        # Special keys
        if key == keyboard.Key.left:
            key_states['left'] = True
        elif key == keyboard.Key.right:
            key_states['right'] = True
        elif key == keyboard.Key.up:
            key_states['up'] = True
        elif key == keyboard.Key.down:
            key_states['down'] = True


def on_release(key):
    """Callback when a key is released."""
    global wheel_mode, show_input_selector
    
    try:
        k = key.char.lower()
        if k in key_states:
            key_states[k] = False
            
            # Check for wheel mode toggle on "v" release
            if k == 'v':
                wheel_mode = not wheel_mode
                print(f"[INFO] {'Enabled' if wheel_mode else 'Disabled'} wheel and pedals mode")
                
            # Check for input selector toggle on "i" release
            elif k == 'i':
                show_input_selector = True
                print("[INFO] Input device selector requested")
    except AttributeError:
        # Special keys
        if key == keyboard.Key.left:
            key_states['left'] = False
        elif key == keyboard.Key.right:
            key_states['right'] = False
        elif key == keyboard.Key.up:
            key_states['up'] = False
        elif key == keyboard.Key.down:
            key_states['down'] = False


def setup_keyboard_listener():
    """Configure and return a keyboard listener."""
    return keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )


def get_keyboard_input():
    """
    Read keyboard inputs and return corresponding commands.

    Returns:
        tuple: (steering, acceleration) - Values between -1.0 and 1.0
    """
    steering = 0.0
    acceleration = 0.0

    # ZQSD controls
    if key_states['q']:
        steering = -1.0  # Q = Left
    elif key_states['d']:
        steering = 1.0   # D = Right

    if key_states['z']:
        acceleration = 1.0  # Z = Forward
    elif key_states['s']:
        acceleration = -1.0  # S = Backward

    # Arrow controls
    if key_states['left']:
        steering = -1.0
    elif key_states['right']:
        steering = 1.0

    if key_states['up']:
        acceleration = 1.0
    elif key_states['down']:
        acceleration = -1.0

    return steering, acceleration


class JoystickCalibration:
    """Class to manage joystick calibration."""

    def __init__(self):
        """Initialize joystick calibration data."""
        self.calibration_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "joystick_calibration.json"
        )
        self.calib_data = {
            "steering": {"min": -1.0, "max": 1.0},
            "acceleration": {"min": -1.0, "max": 1.0}
        }
        self.load_calibration()

    def load_calibration(self):
        """Load calibration data from file."""
        try:
            if os.path.exists(self.calibration_file):
                with open(self.calibration_file, 'r') as f:
                    self.calib_data = json.load(f)
                    print(f"[INFO] Joystick calibration loaded: {self.calib_data}")
        except Exception as e:
            print(f"[WARNING] Error loading calibration: {e}")

    def save_calibration(self):
        """Save calibration data to file."""
        try:
            with open(self.calibration_file, 'w') as f:
                json.dump(self.calib_data, f, indent=4)
                print(f"[INFO] Calibration saved: {self.calib_data}")
        except Exception as e:
            print(f"[ERROR] Unable to save calibration: {e}")

    def apply_calibration(self, axis_value, axis_type):
        """
        Apply calibration to an axis value.

        Args:
            axis_value: Raw axis value
            axis_type: 'steering' or 'acceleration'

        Returns:
            float: Normalized value between -1.0 and 1.0
        """
        min_val = self.calib_data[axis_type]["min"]
        max_val = self.calib_data[axis_type]["max"]

        # Avoid division by zero
        if min_val == max_val:
            return 0.0

        # Normalize between -1 and 1
        if axis_value >= 0:
            return axis_value / max_val if max_val != 0 else 0.0
        else:
            return axis_value / abs(min_val) if min_val != 0 else 0.0


# Global calibration instance
joystick_calibration = JoystickCalibration()


def get_wheel_pedals_input(joystick):
    """
    Read Trust GXT 570 wheel and pedals inputs.
    
    Args:
        joystick: Pygame joystick instance representing the wheel
        
    Returns:
        tuple: (steering, acceleration) - Values between -1.0 and 1.0
    """
    if joystick is None:
        return 0.0, 0.0
        
    # Check that joystick is initialized
    try:
        if not joystick.get_init():
            try:
                joystick.init()
                print("[INFO] Wheel automatically reinitialized")
            except pygame.error:
                return 0.0, 0.0
    except (pygame.error, AttributeError):
        # Disconnected or error with wheel
        return 0.0, 0.0
        
    try:
        # Deadzone for wheel
        deadzone = 0.05
        
        # Steering from wheel rotation (axis 0)
        steering_axis = joystick.get_axis(0)
        
        # Get brake pedal and accelerator pedal
        if "GXT 570" in joystick.get_name() or "Trust" in joystick.get_name():
            brake_pedal = joystick.get_axis(2)
            accelerator_pedal = joystick.get_axis(1)
        else:
            brake_pedal = joystick.get_axis(1)
            accelerator_pedal = joystick.get_axis(2)
        
        # Apply deadzone to steering
        if abs(steering_axis) < deadzone:
            steering_axis = 0.0
            
        # Calculate acceleration (accelerator - brake)
        # Note: Pedals often return -1 when fully released and 1 when fully pressed
        # We need to map these to a range where:
        # -1 = full brake
        # 0 = no input
        # 1 = full acceleration
        
        # Convert from [-1, 1] to [0, 1] range for each pedal
        # Where 0 = released, 1 = fully pressed
        brake_value = (brake_pedal + 1) / 2
        accelerator_value = (accelerator_pedal + 1) / 2
        
        # Calculate final acceleration value (-1 to 1)
        # Full brake takes precedence over accelerator
        acceleration = accelerator_value - brake_value
        
        # Apply calibration
        steering = joystick_calibration.apply_calibration(steering_axis, "steering")
        accel = acceleration  # The acceleration is already in the right range
        
        return steering, accel
        
    except (pygame.error, IndexError) as e:
        print(f"[ERROR] Unable to read wheel inputs: {e}")
        return 0.0, 0.0


def get_joystick_input(joystick):
    """
    Read joystick inputs and return commands.

    Args:
        joystick: Pygame joystick instance

    Returns:
        tuple: (steering, acceleration) - Values between -1.0 and 1.0
    """
    global single_stick_mode, last_y_button_state
    
    if joystick is None:
        return 0.0, 0.0

    # Check that joystick is initialized
    try:
        if not joystick.get_init():
            try:
                joystick.init()
                print("[INFO] Joystick automatically reinitialized")
            except pygame.error:
                return 0.0, 0.0
    except (pygame.error, AttributeError):
        # Disconnected or error joystick
        return 0.0, 0.0

    # Deadzone to avoid involuntary movements
    deadzone = 0.1

    # Check Y button to toggle control mode
    try:
        # Assuming Y button is button 2 (index might need adjustment based on your controller)
        current_y_state = joystick.get_button(2)
        
        # Toggle mode on button press (not hold)
        if current_y_state and not last_y_button_state:
            single_stick_mode = not single_stick_mode
            mode_name = "Single Stick" if single_stick_mode else "Dual Stick"
            print(f"[INFO] Switched to {mode_name} control mode")
            
        last_y_button_state = current_y_state
    except (pygame.error, IndexError):
        # If we can't read the button, just continue with current mode
        pass

    try:
        # In both modes, steering comes from left stick horizontal
        steering_axis = joystick.get_axis(0)
        
        if single_stick_mode:
            # Original mode: left stick vertical for acceleration
            accel_axis = joystick.get_axis(1)
        else:
            # Dual stick mode: right stick vertical for acceleration
            # Typically right stick vertical is axis 3 or 4 depending on the controller
            try:
                accel_axis = joystick.get_axis(3)  # Try common right stick vertical index
            except (pygame.error, IndexError):
                try:
                    accel_axis = joystick.get_axis(4)  # Alternative right stick vertical index
                except (pygame.error, IndexError):
                    # Fall back to left stick if right stick not available
                    accel_axis = joystick.get_axis(1)
                    print("[WARNING] Right stick not detected, falling back to left stick")
    except (pygame.error, IndexError):
        print("[ERROR] Unable to read joystick axes")
        return 0.0, 0.0

    # Apply deadzone
    if abs(steering_axis) < deadzone:
        steering_axis = 0.0
    if abs(accel_axis) < deadzone:
        accel_axis = 0.0

    # Apply calibration
    steering = joystick_calibration.apply_calibration(steering_axis, "steering")
    # Inversion so up = forward
    accel = -joystick_calibration.apply_calibration(accel_axis, "acceleration")

    return steering, accel


def get_device_input(device, input_type):
    """
    Get input from a specific device.
    
    Args:
        device: Device information dictionary
        input_type: "steering" or "acceleration"
        
    Returns:
        float: Input value between -1.0 and 1.0
    """
    if not device:
        return 0.0
        
    if device["type"] == "keyboard":
        steering, acceleration = get_keyboard_input()
        return steering if input_type == "steering" else acceleration
        
    if "instance" not in device:
        # Try to initialize the device if it's a joystick
        if isinstance(device["id"], int):
            try:
                joystick = pygame.joystick.Joystick(device["id"])
                joystick.init()
                device["instance"] = joystick
            except (pygame.error, IndexError):
                return 0.0
                
    joystick = device.get("instance")
    if not joystick:
        return 0.0
        
    # Handle different device types
    if device["type"] == "wheel_gxt":
        steering, acceleration = get_wheel_pedals_input(joystick)
        return steering if input_type == "steering" else acceleration
    elif device["type"] == "wheel_logitech":
        # Similar to GXT but might have different axis mapping
        steering, acceleration = get_wheel_pedals_input(joystick)
        return steering if input_type == "steering" else acceleration
    else:
        # Generic joystick
        steering, acceleration = get_joystick_input(joystick)
        return steering if input_type == "steering" else acceleration


def show_input_device_selector(joystick):
    """
    Show the input device selector GUI.
    
    Args:
        joystick: Current joystick instance
        
    Returns:
        tuple: Updated steering_device and acceleration_device
    """
    global steering_device, acceleration_device, show_input_selector
    
    from input_selector import InputDeviceSelector
    
    # Initialize the selector
    selector = InputDeviceSelector()
    
    # Run the selector
    selected_steering, selected_acceleration = selector.run()
    
    if selected_steering and selected_acceleration:
        steering_device = selected_steering
        acceleration_device = selected_acceleration
        print(f"[INFO] Input configuration updated:")
        print(f"  - Steering: {steering_device['name']}")
        print(f"  - Acceleration: {selected_acceleration['name']}")
    
    # Reset the flag
    show_input_selector = False
    
    # Return both devices
    return steering_device, acceleration_device


def parse_user_input(joystick=None):
    """
    Read user inputs and convert to steering commands.

    Args:
        joystick: Pygame joystick instance (None for keyboard only)

    Returns:
        tuple: (steering, acceleration) - Values between -1.0 and 1.0
    """
    global wheel_mode, show_input_selector, steering_device, acceleration_device
    
    # Check if we need to show the input selector
    if show_input_selector:
        steering_device, acceleration_device = show_input_device_selector(joystick)
        # Reinitialize selected joystick instances
        if steering_device and "instance" in steering_device and steering_device["instance"]:
            steering_device["instance"].quit()
            steering_device["instance"].init()
        if acceleration_device and "instance" in acceleration_device and acceleration_device["instance"]:
            acceleration_device["instance"].quit()
            acceleration_device["instance"].init()
    
    # If we have custom device configuration
    if steering_device or acceleration_device:
        steering = get_device_input(steering_device, "steering")
        acceleration = get_device_input(acceleration_device, "acceleration")
        return steering, acceleration
    
    # Default behavior if no custom configuration
    steering, acceleration = get_keyboard_input()

    # If joystick available and keyboard not used
    if joystick is not None and steering == 0.0 and acceleration == 0.0:
        if wheel_mode:
            # Use wheel and pedals mode
            steering, acceleration = get_wheel_pedals_input(joystick)
        else:
            # Use standard joystick controls
            steering, acceleration = get_joystick_input(joystick)

    return steering, acceleration
