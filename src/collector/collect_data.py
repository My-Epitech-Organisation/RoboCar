"""
Data collection script from Unity simulation.

This script:
1. Reads configuration from raycast_config.json
2. Launches Unity simulation
3. Records user actions and sensor data
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback

import numpy as np
import pygame
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel
)

from utils_collector import parse_user_input, setup_keyboard_listener, key_states
from joystick_calibrator import calibrate_joystick


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Robocar Data Collector')
    parser.add_argument(
        '--calibrate',
        action='store_true',
        help='Launch joystick calibration at startup'
    )
    return parser.parse_args()


def get_project_root():
    """Return the project root path."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_config(project_root):
    """Load configuration from JSON file."""
    config_path = os.path.join(project_root, "config", "raycast_config.json")
    print(f"[INFO] Loading configuration from {config_path}")

    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[ERROR] Malformed configuration file: {config_path}")
        sys.exit(1)


def check_unity_executable(project_root):
    """Check existence and permissions of Unity executable."""
    unity_env_path = os.path.join(
        project_root,
        "RacingSimulatorLinux",
        "RacingSimulator.x86_64"
    )
    print(f"[INFO] Path to Unity environment: {unity_env_path}")

    if not os.path.exists(unity_env_path):
        print(f"[ERROR] Unity executable does not exist: {unity_env_path}")
        sys.exit(1)

    if not os.access(unity_env_path, os.X_OK):
        print(f"[WARNING] Adding execution permissions to {unity_env_path}")
        try:
            os.chmod(unity_env_path, 0o755)
        except Exception as e:
            print(f"[ERROR] Unable to add permissions: {e}")
            sys.exit(1)

    return unity_env_path


def init_joystick():
    """Initialize and configure joystick."""
    pygame.init()
    joystick_count = pygame.joystick.get_count()
    joystick = None

    if joystick_count > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"[INFO] Joystick detected: {joystick.get_name()}")
    else:
        print("[INFO] No joystick detected, using keyboard only")

    return joystick


def reinitialize_joystick():
    """Completely reinitialize joystick after calibration."""
    if not pygame.get_init():
        pygame.init()

    pygame.joystick.quit()
    pygame.time.delay(100)
    pygame.joystick.init()

    joystick_count = pygame.joystick.get_count()
    joystick = None

    if joystick_count > 0:
        try:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            print(f"[INFO] Joystick reinitialized: {joystick.get_name()}")
        except pygame.error as e:
            print(f"[ERROR] Unable to reinitialize joystick: {e}")
            return None
    else:
        print("[INFO] No joystick detected after reinitialization")

    return joystick


def setup_unity_environment(unity_env_path, engine_config, project_root):
    """Configure and initialize Unity environment."""
    agent_config_path = os.path.join(
        project_root, "config", "agent_config.json"
    )
    print(f"[INFO] Agent configuration: {agent_config_path}")

    print("[INFO] Launching and connecting to Unity environment...")
    env = UnityEnvironment(
        file_name=unity_env_path,
        side_channels=[engine_config],
        base_port=5004,
        additional_args=[
            "--config-path", f"{agent_config_path}",
            "-logFile", "unity.log"
        ]
    )

    print("[INFO] Resetting Unity environment")
    env.reset()
    print("[INFO] Unity environment successfully initialized!")

    return env, agent_config_path


def get_num_rays_from_config(project_root):
    """Get number of rays from agent configuration."""
    num_rays = 10  # Default value
    agent_config_path = os.path.join(
        project_root, "config", "agent_config.json"
    )
    try:
        with open(agent_config_path, "r") as f:
            agent_config = json.load(f)
            if "agents" in agent_config and len(agent_config["agents"]) > 0:
                num_rays = agent_config["agents"][0].get("nbRay", 10)
                print(f"[INFO] Number of rays from config: {num_rays}")
    except Exception as e:
        print(f"[WARNING] Could not load ray count from config: {e}")
        print(f"[WARNING] Using default ray count: {num_rays}")

    return num_rays


def setup_data_collection(env, project_root):
    """Configure structures for data collection."""
    behavior_name = list(env.behavior_specs.keys())[0]
    print(f"[INFO] Detected behavior: {behavior_name}")

    behavior_spec = env.behavior_specs[behavior_name]
    print(f"[INFO] Number of observations: {len(behavior_spec.observation_specs)}")

    for i, obs_spec in enumerate(behavior_spec.observation_specs):
        print(
            f"[INFO] Observation {i}: shape={obs_spec.shape}, "
            f"type={obs_spec.observation_type}"
        )

    print(
        f"[INFO] Action type: Continuous with "
        f"{behavior_spec.action_spec.continuous_size} dimensions"
    )

    num_rays = get_num_rays_from_config(project_root)

    output_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"session_{int(time.time())}.csv")
    print(f"[INFO] Writing data to {output_file}")

    return behavior_name, behavior_spec, output_file, num_rays


def handle_joystick_calibration(joystick):
    """Handle joystick calibration process."""
    if not joystick:
        print("[ERROR] No joystick detected for calibration.")
        return joystick, False

    print("[INFO] Starting calibration...")
    calibrate_joystick(joystick)
    print("[INFO] Calibration completed, reinitializing joystick...")

    pygame.quit()
    pygame.init()

    joystick = reinitialize_joystick()
    print("[INFO] Resuming collection with reinitialized joystick.")

    return joystick, True


def process_observations(obs_array, num_rays):
    """Extract and process observations from observation array."""
    raycasts = obs_array[:num_rays].tolist()

    try:
        speed = float(obs_array[-5]) if len(obs_array) >= 5 else 0.0
        obs_steering = float(obs_array[-4]) if len(obs_array) >= 4 else 0.0
        position_x = float(obs_array[-3]) if len(obs_array) >= 3 else 0.0
        position_y = float(obs_array[-2]) if len(obs_array) >= 2 else 0.0
        position_z = float(obs_array[-1]) if len(obs_array) >= 1 else 0.0
    except IndexError:
        speed, obs_steering = 0.0, 0.0
        position_x, position_y, position_z = 0.0, 0.0, 0.0

    return raycasts, speed, obs_steering, position_x, position_y, position_z


def display_state_info(frame_count, steering, accel, speed, obs_steering,
                       position_x, position_y, position_z, raycasts):
    """Display current simulation state periodically."""
    if frame_count % 10 == 0:
        print("\n[INFO] Simulation state:")
        print(f"  User inputs: steering={steering:.2f}, "
              f"acceleration={accel:.2f}")
        
        # Add controller mode info
        from utils_collector import xbox_mode, single_stick_mode, wheel_mode
        mode_info = []
        if xbox_mode:
            mode_info.append("Xbox mode")
        if single_stick_mode:
            mode_info.append("Single stick")
        if wheel_mode:
            mode_info.append("Wheel mode")
        
        if mode_info:
            print(f"  Control mode: {', '.join(mode_info)}")
            
        print(f"  Observations:")
        print(f"    Speed: {speed:.2f}")
        print(f"    Car steering: {obs_steering:.2f}")
        print(f"    Position: ({position_x:.2f}, {position_y:.2f}, "
              f"{position_z:.2f})")
        print(f"    Raycasts: {len(raycasts)} rays, "
              f"range: {min(raycasts):.2f} to {max(raycasts):.2f}")


def collect_data_loop(env, behavior_name, output_file, joystick):
    """Execute main data collection loop."""
    project_root = get_project_root()
    num_rays = get_num_rays_from_config(project_root)

    fieldnames = [
        "timestamp", "steering_input", "acceleration_input",
        "raycasts", "speed", "steering", "position_x", "position_y", "position_z"
    ]

    with open(output_file, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        print("[INFO] Getting first observations...")
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        print("[INFO] Observations retrieved, starting main loop")
        print("[INFO] Data collection in progress. Press Ctrl+C to stop.")
        print("[INFO] Controls: Arrow keys or WASD/ZQSD")
        print("[INFO] Press 'c' to calibrate joystick")
        print("[INFO] Press X on Xbox controller to toggle Xbox mode")

        print("\n[INFO] Observation structure details:")
        for i, obs in enumerate(decision_steps.obs):
            print(f"  Observation {i}: shape={obs.shape}")

        print("\n[INFO] Agent information:")
        print(f"  Agent count: {len(decision_steps)}")

        frame_count = 0
        calibration_requested = False
        post_calibration = False
        post_calibration_counter = 0

        while True:
            if not pygame.get_init():
                pygame.init()
            if not pygame.display.get_init():
                pygame.display.init()

            pygame.event.pump()

            # Handle calibration
            if key_states['c'] and not calibration_requested:
                calibration_requested = True
                print("[INFO] Joystick calibration requested...")
                joystick, post_calibration = handle_joystick_calibration(joystick)
                post_calibration_counter = 0

            if not key_states['c']:
                calibration_requested = False

            # Post-calibration stabilization period
            if post_calibration:
                post_calibration_counter += 1
                if post_calibration_counter > 10:
                    post_calibration = False

            # Get user inputs
            if post_calibration:
                steering, accel = 0.0, 0.0
            else:
                steering, accel = parse_user_input(joystick)

            # Process observations
            obs_array = decision_steps.obs[0][0]
            raycasts, speed, obs_steering, position_x, position_y, position_z = (
                process_observations(obs_array, num_rays)
            )

            # Display information periodically
            frame_count += 1
            display_state_info(
                frame_count, steering, accel, speed, obs_steering,
                position_x, position_y, position_z, raycasts
            )

            # Record data
            writer.writerow({
                "timestamp": time.time(),
                "steering_input": steering,
                "acceleration_input": accel,
                "raycasts": str(raycasts),
                "speed": speed,
                "steering": obs_steering,
                "position_x": position_x,
                "position_y": position_y,
                "position_z": position_z
            })

            # Send actions to simulation
            continuous_actions = np.array([[accel, steering]], dtype=np.float32)
            action_tuple = ActionTuple(continuous=continuous_actions)
            env.set_actions(behavior_name, action_tuple)
            env.step()

            decision_steps, terminal_steps = env.get_steps(behavior_name)


def main():
    """Main function of the data collector."""
    args = parse_arguments()
    project_root = get_project_root()
    raycast_config = load_config(project_root)
    unity_env_path = check_unity_executable(project_root)

    # Configure Unity communication channel
    print("[INFO] Configuring communication channel with Unity")
    engine_config = EngineConfigurationChannel()
    engine_config.set_configuration_parameters(
        width=raycast_config["graphic_settings"]["width"],
        height=raycast_config["graphic_settings"]["height"],
        quality_level=raycast_config["graphic_settings"]["quality_level"],
        time_scale=raycast_config["time_scale"]
    )

    # Initialize keyboard listener
    print("[INFO] Initializing global keyboard listener")
    keyboard_listener = setup_keyboard_listener()
    keyboard_listener.start()
    print("[INFO] Keyboard listener started - arrows or WASD/ZQSD")

    # Initialize joystick
    joystick = init_joystick()

    # Calibrate at startup if requested
    if args.calibrate and joystick:
        print("[INFO] Joystick calibration at startup...")
        calibrate_joystick(joystick)

    try:
        # Configure Unity environment
        env, _ = setup_unity_environment(
            unity_env_path, engine_config, project_root
        )

        try:
            # Configure data collection
            behavior_name, _, output_file, num_rays = setup_data_collection(
                env, project_root
            )

            try:
                # Data collection loop
                collect_data_loop(env, behavior_name, output_file, joystick)
            except KeyboardInterrupt:
                print("[INFO] Collection interrupted by user.")
            except Exception as e:
                print(f"[ERROR] Exception during collection: {e}")
                traceback.print_exc()
        except Exception as e:
            print(f"[ERROR] Exception during configuration: {e}")
            traceback.print_exc()
        finally:
            print("[INFO] Closing Unity environment")
            env.close()
    except Exception as e:
        print(f"[ERROR] Exception during Unity initialization: {e}")
        traceback.print_exc()
    finally:
        print("[INFO] Stopping keyboard listener")
        keyboard_listener.stop()
        print("[INFO] Closing pygame")
        pygame.quit()
        print("[INFO] Program terminated.")


if __name__ == "__main__":
    main()
