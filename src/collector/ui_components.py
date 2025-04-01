"""
UI components for joystick calibration.

This module contains the graphical interface components used for joystick calibration.
"""

import pygame
import time

from utils_collector import joystick_calibration


class JoystickCalibratorUI:
    """Graphical interface for joystick calibration."""

    def __init__(self, joystick):
        """
        Initialize the calibration interface.

        Args:
            joystick: Pygame joystick instance to calibrate
        """
        self.joystick = joystick
        if not joystick:
            print("[ERROR] No joystick detected for calibration")
            return

        # Initialize Pygame
        if not pygame.get_init():
            pygame.init()

        # Create window
        self.width, self.height = 800, 600
        self.window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Joystick Calibration")

        # Initialize colors, fonts and calibration variables
        self._init_colors()
        self._init_fonts()
        self._init_calibration_variables()

    def _init_colors(self):
        """Initialize color definitions."""
        # Basic colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (100, 100, 100)
        self.LIGHT_GRAY = (200, 200, 200)
        self.DARK_GRAY = (50, 50, 50)

        # Vivid colors
        self.RED = (255, 59, 48)
        self.GREEN = (52, 199, 89)
        self.BLUE = (0, 122, 255)
        self.ORANGE = (255, 149, 0)
        self.PURPLE = (175, 82, 222)
        self.TEAL = (48, 176, 199)

        # Background colors
        self.BACKGROUND_TOP = (22, 23, 30)
        self.BACKGROUND_BOTTOM = (43, 45, 66)

    def _init_fonts(self):
        """Initialize fonts for UI."""
        try:
            # Try to load a more elegant system font
            self.font = pygame.font.SysFont("Arial", 36, bold=True)
            self.small_font = pygame.font.SysFont("Arial", 24)
            self.tiny_font = pygame.font.SysFont("Arial", 18)
        except:
            # Fallback to default font
            self.font = pygame.font.SysFont(None, 36)
            self.small_font = pygame.font.SysFont(None, 24)
            self.tiny_font = pygame.font.SysFont(None, 18)

    def _init_calibration_variables(self):
        """Initialize calibration state variables."""
        # Calibration values
        self.steering_min = 0.0
        self.steering_max = 0.0
        self.accel_min = 0.0
        self.accel_max = 0.0

        # Calibration state
        self.is_calibrating = True
        self.start_time = time.time()
        self.calibration_duration = 10  # Duration in seconds

        # Animation variables
        self.pulse_value = 0
        self.pulse_direction = 1

    def update_calibration_values(self):
        """Update min/max values based on current inputs."""
        if not self.joystick:
            return

        # Read current values
        steering_value = self.joystick.get_axis(0)
        accel_value = self.joystick.get_axis(1)

        # Update min/max values for steering
        if steering_value < self.steering_min:
            self.steering_min = steering_value
        if steering_value > self.steering_max:
            self.steering_max = steering_value

        # Update min/max values for acceleration
        if accel_value < self.accel_min:
            self.accel_min = accel_value
        if accel_value > self.accel_max:
            self.accel_max = accel_value

        # Update pulsation effect
        self.pulse_value += 0.1 * self.pulse_direction
        if self.pulse_value >= 1.0:
            self.pulse_direction = -1
        elif self.pulse_value <= 0.0:
            self.pulse_direction = 1

    def save_calibration(self):
        """Save calibration values."""
        # Check values
        if self.steering_min == self.steering_max:
            self.steering_min = -1.0
            self.steering_max = 1.0

        if self.accel_min == self.accel_max:
            self.accel_min = -1.0
            self.accel_max = 1.0

        # Update and save
        joystick_calibration.calib_data["steering"]["min"] = self.steering_min
        joystick_calibration.calib_data["steering"]["max"] = self.steering_max
        joystick_calibration.calib_data["acceleration"]["min"] = self.accel_min
        joystick_calibration.calib_data["acceleration"]["max"] = self.accel_max

        joystick_calibration.save_calibration()

    def draw_gradient_background(self):
        """Draw a gradient background."""
        for y in range(self.height):
            # Linear interpolation between top and bottom colors
            ratio = y / self.height
            r = int(self.BACKGROUND_TOP[0] * (1 - ratio) +
                    self.BACKGROUND_BOTTOM[0] * ratio)
            g = int(self.BACKGROUND_TOP[1] * (1 - ratio) +
                    self.BACKGROUND_BOTTOM[1] * ratio)
            b = int(self.BACKGROUND_TOP[2] * (1 - ratio) +
                    self.BACKGROUND_BOTTOM[2] * ratio)
            pygame.draw.line(self.window, (r, g, b), (0, y), (self.width, y))

    def draw_text_with_shadow(self, text, font, color, position, shadow_offset=2):
        """Draw text with a drop shadow."""
        shadow = font.render(text, True, self.DARK_GRAY)
        text_surface = font.render(text, True, color)
        self.window.blit(
            shadow,
            (position[0] + shadow_offset, position[1] + shadow_offset)
        )
        self.window.blit(text_surface, position)

    def draw_rounded_rect(self, surface, rect, color, radius=15):
        """Draw a rectangle with rounded corners."""
        rect = pygame.Rect(rect)

        # Handle RGBA colors
        if len(color) == 4:
            # Create a surface with per-pixel alpha
            shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(
                shape_surf,
                color,
                shape_surf.get_rect(),
                border_radius=radius
            )
            surface.blit(shape_surf, rect)
            return

        # Regular color handling (RGB)
        pygame.draw.rect(surface, color, rect, border_radius=radius)

    def draw_progress_bar(self, x, y, width, height, progress,
                         bg_color, fill_color, border_color):
        """Draw a progress bar with a nice style."""
        # Bar background
        self.draw_rounded_rect(self.window, (x, y, width, height), bg_color)

        # Calculate fill width
        fill_width = int(width * progress)

        # Fill (ensure it doesn't exceed total width)
        if fill_width > 0:
            if fill_width < 2 * height:  # If fill is too small for rounded corners
                pygame.draw.rect(
                    self.window,
                    fill_color,
                    (x, y, fill_width, height)
                )
            else:
                self.draw_rounded_rect(
                    self.window,
                    (x, y, fill_width, height),
                    fill_color
                )

        # Border
        pygame.draw.rect(
            self.window,
            border_color,
            (x, y, width, height),
            2,
            border_radius=height//2
        )

    def draw_axis_visualization(self, x, y, width, height, value,
                               min_val, max_val, label, color):
        """Draw an axis visualization with a modern style."""
        # Background with rounded corners
        self.draw_rounded_rect(self.window, (x, y, width, height), self.DARK_GRAY)

        # Calculate normalized position
        normalized = self._calculate_normalized_position(value, min_val, max_val)
        pos_x = x + int(width * normalized)

        # Draw visualization elements
        self._draw_axis_cursor(pos_x, y, height, color)
        self._draw_axis_markers(x, y, width, height)

        # Draw text
        text = f"{label}: {value:.2f} (Min: {min_val:.2f}, Max: {max_val:.2f})"
        self.draw_text_with_shadow(text, self.small_font, self.WHITE, (x, y - 30))

    def _calculate_normalized_position(self, value, min_val, max_val):
        """Calculate normalized position (0-1) for axis visualization."""
        normalized = 0.5
        if min_val != max_val:
            if value >= 0:
                normalized = 0.5 + (value / max_val) * 0.5 if max_val != 0 else 0.5
            else:
                normalized = 0.5 - (value / min_val) * 0.5 if min_val != 0 else 0.5
        return normalized

    def _draw_axis_cursor(self, pos_x, y, height, color):
        """Draw cursor for axis visualization."""
        # Cursor with pulsating effect
        pulse_size = 5 + int(2 * self.pulse_value)
        pygame.draw.circle(
            self.window,
            color,
            (pos_x, y + height // 2),
            pulse_size
        )
        pygame.draw.circle(
            self.window,
            self.WHITE,
            (pos_x, y + height // 2),
            pulse_size - 2
        )

        # Vertical line at position
        pygame.draw.line(
            self.window,
            color,
            (pos_x, y),
            (pos_x, y + height),
            2
        )

    def _draw_axis_markers(self, x, y, width, height):
        """Draw markers and graduations for axis visualization."""
        # Mark for center (0)
        zero_x = x + width // 2
        pygame.draw.line(
            self.window,
            self.GREEN,
            (zero_x, y),
            (zero_x, y + height),
            2
        )

        # Add marks for extremes
        mark_min = x + width // 4
        mark_max = x + 3 * (width // 4)
        for mark_x in [mark_min, mark_max]:
            pygame.draw.line(
                self.window,
                self.LIGHT_GRAY,
                (mark_x, y),
                (mark_x, y + height),
                1
            )

        # Small graduations
        for i in range(1, 10):
            grad_x = mark_min + (mark_max - mark_min) * i / 10
            grad_height = height // 4 if i % 5 == 0 else height // 8
            pygame.draw.line(
                self.window,
                self.GRAY,
                (grad_x, y + height - grad_height),
                (grad_x, y + height),
                1
            )

    def draw_joystick_position(self, x, y, size):
        """Draw an enhanced visual representation of joystick position."""
        if not self.joystick:
            return

        # Draw joystick visualization elements
        self._draw_joystick_background(x, y, size)

        # Get current joystick position
        steering = self.joystick.get_axis(0)
        accel = -self.joystick.get_axis(1)  # Inversion for intuition

        # Draw position
        self._draw_joystick_position_path(x, y, size, steering, accel)
        self._draw_joystick_position_marker(x, y, size, steering, accel)

    def _draw_joystick_background(self, x, y, size):
        """Draw background elements for joystick visualization."""
        # Outer circle (border)
        pygame.draw.circle(self.window, self.LIGHT_GRAY, (x, y), size//2, 2)

        # Inner circle (deadzone area)
        deadzone_size = size//10
        pygame.draw.circle(self.window, self.DARK_GRAY, (x, y), deadzone_size)

        # Grid lines
        pygame.draw.line(
            self.window,
            self.GRAY,
            (x - size//2, y),
            (x + size//2, y),
            1
        )
        pygame.draw.line(
            self.window,
            self.GRAY,
            (x, y - size//2),
            (x, y + size//2),
            1
        )

        # Add concentric circles
        for r in range(1, 4):
            pygame.draw.circle(
                self.window,
                self.GRAY,
                (x, y),
                (size//2) * r // 4,
                1
            )

    def _draw_joystick_position_path(self, x, y, size, steering, accel):
        """Draw path from center to current joystick position."""
        # Path from center to position (with gradient)
        steps = 10
        for i in range(steps):
            prog = i / steps
            step_x = x + int(steering * size//2 * 0.95 * prog)
            step_y = y - int(accel * size//2 * 0.95 * prog)
            pygame.draw.circle(
                self.window,
                self.RED,
                (step_x, step_y),
                2 + i//3
            )

    def _draw_joystick_position_marker(self, x, y, size, steering, accel):
        """Draw marker for current joystick position."""
        # Calculate position
        pos_x = x + int(steering * size//2 * 0.95)  # 0.95 to avoid edge
        pos_y = y - int(accel * size//2 * 0.95)

        # Point representing current position (with pulsating effect)
        glow_size = 8 + int(4 * self.pulse_value)

        # Outer glow
        pygame.draw.circle(self.window, self.RED, (pos_x, pos_y), glow_size)

        # Inner circle
        pygame.draw.circle(
            self.window,
            self.WHITE,
            (pos_x, pos_y),
            glow_size - 4
        )

    def draw_interface(self, elapsed_time):
        """Draw the complete calibration interface."""
        # Gradient background
        self.draw_gradient_background()

        # Main frame
        self._draw_main_frame()

        # Title and progress
        self._draw_title_and_progress(elapsed_time)

        # Instructions and controls
        self._draw_instructions()

        # Joystick visualization
        self._draw_axis_visualizations()
        self.draw_joystick_position(self.width//2, 480, 200)

        # Command information
        self._draw_command_info()

    def _draw_main_frame(self):
        """Draw main UI frame."""
        margin = 20
        main_rect = (
            margin,
            margin,
            self.width - 2*margin,
            self.height - 2*margin
        )
        self.draw_rounded_rect(
            self.window,
            main_rect,
            (*self.BACKGROUND_TOP, 150)
        )

    def _draw_title_and_progress(self, elapsed_time):
        """Draw title and progress bar."""
        # Title with shadow effect
        title_y = 50
        title_text = "Joystick Calibration"
        self.draw_text_with_shadow(
            title_text,
            self.font,
            self.WHITE,
            (self.width//2 - self.font.size(title_text)[0]//2, title_y)
        )

        # Progress bar for time
        progress = min(1.0, elapsed_time / self.calibration_duration)
        progress_width = self.width - 100
        self.draw_progress_bar(
            50, title_y + 50,
            progress_width, 20,
            progress,
            self.DARK_GRAY,
            self.TEAL,
            self.LIGHT_GRAY
        )

        # Time remaining text
        time_left = max(0, self.calibration_duration - elapsed_time)
        time_text = f"Time remaining: {time_left:.1f}s"
        time_text_pos = (
            self.width//2 - self.small_font.size(time_text)[0]//2,
            title_y + 80
        )
        self.draw_text_with_shadow(
            time_text,
            self.small_font,
            self.WHITE,
            time_text_pos
        )

    def _draw_instructions(self):
        """Draw instruction text."""
        title_y = 50  # Same as in _draw_title_and_progress
        instructions = [
            "Move the joystick to all extreme directions",
            "Press ENTER to finish or ESCAPE to cancel"
        ]

        for i, text in enumerate(instructions):
            y_pos = title_y + 120 + i * 30
            self.draw_text_with_shadow(
                text,
                self.small_font,
                self.LIGHT_GRAY,
                (self.width//2 - self.small_font.size(text)[0]//2, y_pos)
            )

    def _draw_axis_visualizations(self):
        """Draw visualizations for both axes."""
        if not self.joystick:
            return

        # Steering axis
        self.draw_axis_visualization(
            80, 250, self.width - 160, 40,
            self.joystick.get_axis(0),
            self.steering_min, self.steering_max,
            "Steering", self.BLUE
        )

        # Acceleration axis
        self.draw_axis_visualization(
            80, 350, self.width - 160, 40,
            self.joystick.get_axis(1),
            self.accel_min, self.accel_max,
            "Acceleration", self.RED
        )

    def _draw_command_info(self):
        """Draw command information at bottom of screen."""
        commands = [
            "ENTER: Save and quit",
            "ESCAPE: Cancel and quit"
        ]

        for i, cmd in enumerate(commands):
            cmd_y = self.height - 50 + i * 20
            self.draw_text_with_shadow(
                cmd,
                self.tiny_font,
                self.LIGHT_GRAY,
                (self.width - self.tiny_font.size(cmd)[0] - 30, cmd_y)
            )

    def check_events(self):
        """
        Check user events.

        Returns:
            bool: True if calibration should continue, False otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_RETURN:
                    self.save_calibration()
                    return False
        return True

    def run(self):
        """Run the main calibration interface loop."""
        if not self.joystick:
            return False

        clock = pygame.time.Clock()

        while self.is_calibrating:
            # Event handling
            self.is_calibrating = self.check_events()

            # Update calibration values
            self.update_calibration_values()

            # Check elapsed time
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= self.calibration_duration:
                self.save_calibration()
                self.is_calibrating = False

            # Draw interface
            self.draw_interface(elapsed_time)

            # Update display
            pygame.display.flip()
            clock.tick(30)

        # Cleanup - don't close the window with set_mode but with flip and delay
        if pygame.display.get_init():
            pygame.display.flip()
            pygame.time.delay(100)  # Delay to let pygame process

        return True
