"""
Input device selector GUI.

This module provides a graphical interface to select input devices for steering and acceleration.
"""

import pygame
import sys
import os

if not os.environ.get("DISPLAY"):
    os.environ["SDL_VIDEODRIVER"] = "dummy"

class InputDeviceSelector:
    """Graphical interface for selecting input devices."""

    def __init__(self, available_devices=None):
        """
        Initialize the device selector interface.
        
        Args:
            available_devices: List of available input devices
        """
        # Initialize pygame if not already initialized
        if not pygame.get_init():
            pygame.init()
            
        # Initialize fonts
        pygame.font.init()
        
        # Create window
        self.width, self.height = 800, 600
        self.window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Input Device Selector")
        
        # Initialize colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (100, 100, 100)
        self.LIGHT_GRAY = (200, 200, 200)
        self.DARK_GRAY = (50, 50, 50)
        self.BLUE = (0, 122, 255)
        self.RED = (255, 59, 48)
        self.GREEN = (52, 199, 89)
        self.YELLOW = (255, 204, 0)
        self.PURPLE = (175, 82, 222)
        
        # Background gradient
        self.BACKGROUND_TOP = (22, 23, 30)
        self.BACKGROUND_BOTTOM = (43, 45, 66)
        
        # Initialize fonts
        self.font_large = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 24)
        self.font_small = pygame.font.SysFont("Arial", 18)
        
        # Initialize devices
        self.available_devices = available_devices or self._detect_devices()
        self.selected_steering = None
        self.selected_acceleration = None
        
        # Button data
        self.buttons = []
        
    def _detect_devices(self):
        """Detect all available input devices."""
        devices = [{"name": "Keyboard", "id": "keyboard", "type": "keyboard"}]
        
        # Detect joysticks
        for i in range(pygame.joystick.get_count()):
            try:
                joystick = pygame.joystick.Joystick(i)
                joystick.init()
                
                # Determine device type based on name
                device_type = "joystick"
                device_name = joystick.get_name()
                
                if "GXT 570" in device_name or "Trust" in device_name:
                    if joystick.get_numaxes() >= 3:  # If it has enough axes for steering and pedals
                        device_type = "wheel_gxt"
                elif "Logitech" in device_name:
                    device_type = "wheel_logitech"
                elif "Xbox" in device_name or "Controller" in device_name:
                    device_type = "gamepad"
                
                devices.append({
                    "name": device_name,
                    "id": i,
                    "type": device_type,
                    "instance": joystick
                })
                
            except pygame.error:
                pass
        
        return devices
        
    def draw_gradient_background(self):
        """Draw a gradient background."""
        for y in range(self.height):
            # Linear interpolation between top and bottom colors
            ratio = y / self.height
            r = int(self.BACKGROUND_TOP[0] * (1 - ratio) + self.BACKGROUND_BOTTOM[0] * ratio)
            g = int(self.BACKGROUND_TOP[1] * (1 - ratio) + self.BACKGROUND_BOTTOM[1] * ratio)
            b = int(self.BACKGROUND_TOP[2] * (1 - ratio) + self.BACKGROUND_BOTTOM[2] * ratio)
            pygame.draw.line(self.window, (r, g, b), (0, y), (self.width, y))
            
    def draw_text_with_shadow(self, text, font, color, position, shadow_offset=2):
        """Draw text with a drop shadow."""
        shadow = font.render(text, True, self.DARK_GRAY)
        text_surface = font.render(text, True, color)
        self.window.blit(shadow, (position[0] + shadow_offset, position[1] + shadow_offset))
        self.window.blit(text_surface, position)
        
    def draw_rounded_rect(self, rect, color, radius=15, border=None):
        """Draw a rectangle with rounded corners."""
        pygame.draw.rect(
            self.window,
            color,
            rect,
            0,
            border_radius=radius
        )
        
        if border:
            pygame.draw.rect(
                self.window,
                border,
                rect,
                2,
                border_radius=radius
            )
            
    def setup_interface(self):
        """Set up the interface elements."""
        self.buttons = []
        
        # Title position
        title_y = 30
        
        # Device selection section - Steering
        steering_y = title_y + 70
        steering_section_height = 180
        
        # Device selection section - Acceleration
        accel_y = steering_y + steering_section_height + 30
        accel_section_height = 180
        
        # Navigation buttons
        nav_y = accel_y + accel_section_height + 40
        
        # Add device buttons for steering
        y_pos = steering_y + 50
        for i, device in enumerate(self.available_devices):
            # Calculate position in grid (3 columns)
            row = i // 3
            col = i % 3
            
            button_width = 220
            button_height = 60
            margin = 20
            
            x = (self.width - (button_width * 3 + margin * 2)) // 2 + col * (button_width + margin)
            y = y_pos + row * (button_height + margin)
            
            # Skip if we're out of space
            if y + button_height > steering_y + steering_section_height:
                continue
                
            # Add button
            device_icon = "🎮"
            if device["type"] == "keyboard":
                device_icon = "⌨️"
            elif device["type"] in ["wheel_gxt", "wheel_logitech"]:
                device_icon = "🎮"
                
            self.buttons.append({
                "rect": pygame.Rect(x, y, button_width, button_height),
                "text": f"{device_icon} {device['name']}",
                "action": "steering",
                "device": device,
                "color": self.BLUE if self.selected_steering == device else self.DARK_GRAY,
            })
            
        # Add device buttons for acceleration
        y_pos = accel_y + 50
        for i, device in enumerate(self.available_devices):
            # Calculate position in grid (3 columns)
            row = i // 3
            col = i % 3
            
            button_width = 220
            button_height = 60
            margin = 20
            
            x = (self.width - (button_width * 3 + margin * 2)) // 2 + col * (button_width + margin)
            y = y_pos + row * (button_height + margin)
            
            # Skip if we're out of space
            if y + button_height > accel_y + accel_section_height:
                continue
                
            # Add button
            device_icon = "🎮"
            if device["type"] == "keyboard":
                device_icon = "⌨️"
            elif device["type"] in ["wheel_gxt", "wheel_logitech"]:
                device_icon = "🎮"
                
            self.buttons.append({
                "rect": pygame.Rect(x, y, button_width, button_height),
                "text": f"{device_icon} {device['name']}",
                "action": "acceleration",
                "device": device,
                "color": self.RED if self.selected_acceleration == device else self.DARK_GRAY,
            })
            
        # Add Save & Cancel buttons
        button_width = 200
        button_height = 50
        margin = 40
        
        # Save button
        save_x = self.width // 2 - button_width - margin // 2
        self.buttons.append({
            "rect": pygame.Rect(save_x, nav_y, button_width, button_height),
            "text": "Save & Continue",
            "action": "save",
            "color": self.GREEN,
        })
        
        # Cancel button
        cancel_x = self.width // 2 + margin // 2
        self.buttons.append({
            "rect": pygame.Rect(cancel_x, nav_y, button_width, button_height),
            "text": "Cancel",
            "action": "cancel",
            "color": self.RED,
        })

    def draw_interface(self):
        """Draw the complete interface."""
        # Clear screen
        self.window.fill(self.BLACK)
        
        # Draw background
        self.draw_gradient_background()
        
        # Title
        title_text = "Input Device Selector"
        title_width = self.font_large.size(title_text)[0]
        self.draw_text_with_shadow(
            title_text,
            self.font_large, 
            self.WHITE,
            (self.width // 2 - title_width // 2, 30)
        )
        
        # Steering section
        steering_text = "Select Device for Steering"
        steering_width = self.font_medium.size(steering_text)[0]
        self.draw_text_with_shadow(
            steering_text,
            self.font_medium,
            self.BLUE,
            (self.width // 2 - steering_width // 2, 100)
        )
        
        # Acceleration section
        accel_text = "Select Device for Acceleration"
        accel_width = self.font_medium.size(accel_text)[0]
        self.draw_text_with_shadow(
            accel_text,
            self.font_medium,
            self.RED,
            (self.width // 2 - accel_width // 2, 310)
        )
        
        # Draw all buttons
        for button in self.buttons:
            # Draw button
            self.draw_rounded_rect(
                button["rect"], 
                button["color"],
                radius=10,
                border=self.LIGHT_GRAY
            )
            
            # Draw button text
            text_surface = self.font_small.render(button["text"], True, self.WHITE)
            text_rect = text_surface.get_rect(center=button["rect"].center)
            self.window.blit(text_surface, text_rect)
            
        # Update the display
        pygame.display.flip()

    def handle_events(self):
        """
        Handle pygame events.
        
        Returns:
            bool: True if the selector should continue running, False otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                    
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    # Check if any button was clicked
                    mouse_pos = pygame.mouse.get_pos()
                    for button in self.buttons:
                        if button["rect"].collidepoint(mouse_pos):
                            if button["action"] == "save":
                                if self.selected_steering and self.selected_acceleration:
                                    return False  # Save and exit
                            elif button["action"] == "cancel":
                                self.selected_steering = None
                                self.selected_acceleration = None
                                return False  # Cancel and exit
                            elif button["action"] == "steering":
                                self.selected_steering = button["device"]
                                self.setup_interface()  # Refresh interface
                            elif button["action"] == "acceleration":
                                self.selected_acceleration = button["device"]
                                self.setup_interface()  # Refresh interface
                                
        return True

    def run(self):
        """
        Run the input device selector.
        
        Returns:
            tuple: (steering_device, acceleration_device) or (None, None) if cancelled
        """
        # Ensure pygame's video system is initialized
        if not pygame.display.get_init():
            pygame.display.init()
        
        self.setup_interface()
        
        # Main loop
        clock = pygame.time.Clock()
        running = True
        
        while running:
            self.draw_interface()
            running = self.handle_events()
            clock.tick(30)
            
        # Clean up
        pygame.display.quit()
        
        return self.selected_steering, self.selected_acceleration
