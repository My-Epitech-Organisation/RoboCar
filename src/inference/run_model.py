"""
Autonomous driving inference script using trained neural network model.

This script:
1. Loads the trained model from model_checkpoint.pth
2. Connects to Unity simulation
3. Processes observations and generates control commands
4. Sends commands to control the car autonomously
"""

import os
import sys
import json
import time
import traceback

import numpy as np
import torch
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel
)

# Fix import path issues by adding parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Use direct import which works regardless of how the script is executed
import utils_inference


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


def load_model(project_root):
    """Load the trained neural network model."""
    model_path = os.path.join(project_root, "model_checkpoint.pth")
    print(f"[INFO] Loading model from {model_path}")

    try:
        # Define model architecture (must match training)
        class RegressionModel(torch.nn.Module):
            def __init__(self, input_size, output_size):
                super(RegressionModel, self).__init__()
                self.fc1 = torch.nn.Linear(input_size, 64)
                self.fc2 = torch.nn.Linear(64, 32)
                self.fc3 = torch.nn.Linear(32, output_size)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        # Input size matches features in train.py
        # [steering_input, acceleration_input, 10 raycasts, position_x, position_y, position_z]
        input_size = 15  
        output_size = 2  # [speed, steering]

        model = RegressionModel(input_size, output_size)
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set to evaluation mode
        print(f"[INFO] Model loaded successfully")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        traceback.print_exc()
        sys.exit(1)


def setup_unity_environment(unity_env_path, engine_config, project_root):
    """Configure and initialize Unity environment."""
    agent_config_path = os.path.join(
        project_root, "config", "agent_config.json"
    )
    print(f"[INFO] Agent configuration: {agent_config_path}")

    print("[INFO] Launching and connecting to Unity environment...")
    try:
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

        return env
    except Exception as e:
        print(f"[ERROR] Failed to initialize Unity environment: {e}")
        traceback.print_exc()
        sys.exit(1)


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


def run_inference_loop(env, model, num_rays):
    """Run the main inference loop."""
    behavior_name = list(env.behavior_specs.keys())[0]
    print(f"[INFO] Detected behavior: {behavior_name}")

    try:
        print("[INFO] Starting autonomous driving")
        frame_count = 0
        
        # Initialize exponential moving average for smoother control
        steering_ema = 0.0
        accel_ema = 0.3  # Start with moderate acceleration
        
        while True:
            # Get observations
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            # Skip if no agents
            if len(decision_steps) == 0:
                env.step()
                continue
                
            # Process for each agent (usually just one)
            for agent_id in decision_steps:
                # Extract observations
                obs_array = decision_steps[agent_id].obs[0]
                
                # Prepare inputs for the model
                input_data = utils_inference.preprocess_observation(obs_array, num_rays)
                
                # Generate predictions
                with torch.no_grad():
                    output = model(input_data)
                
                # Convert model output to actions
                raw_steering, raw_accel = utils_inference.postprocess_action(output)
                
                # Apply exponential moving average for smoother control
                alpha = 0.2  # Smoothing factor
                steering_ema = alpha * raw_steering + (1 - alpha) * steering_ema
                accel_ema = alpha * raw_accel + (1 - alpha) * accel_ema
                
                # Display info periodically
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"[INFO] Frame {frame_count}")
                    print(f"  Steering: raw={raw_steering:.2f}, smoothed={steering_ema:.2f}")
                    print(f"  Accel: raw={raw_accel:.2f}, smoothed={accel_ema:.2f}")
                
                # Send actions to simulation
                continuous_actions = np.array([[accel_ema, steering_ema]], dtype=np.float32)
                action_tuple = ActionTuple(continuous=continuous_actions)
                env.set_actions(behavior_name, action_tuple)
            
            env.step()
    
    except KeyboardInterrupt:
        print("[INFO] Inference interrupted by user")
    except Exception as e:
        print(f"[ERROR] Error during inference: {e}")
        traceback.print_exc()


def main():
    """Main function for autonomous driving inference."""
    try:
        # Setup
        project_root = get_project_root()
        raycast_config = load_config(project_root)
        unity_env_path = check_unity_executable(project_root)
        num_rays = get_num_rays_from_config(project_root)
        
        # Initialize normalization
        utils_inference.load_scaler_values()
        
        # Load model
        model = load_model(project_root)
        
        # Configure communication with Unity
        print("[INFO] Configuring communication channel with Unity")
        engine_config = EngineConfigurationChannel()
        engine_config.set_configuration_parameters(
            width=raycast_config["graphic_settings"]["width"],
            height=raycast_config["graphic_settings"]["height"],
            quality_level=raycast_config["graphic_settings"]["quality_level"],
            time_scale=raycast_config["time_scale"]
        )
        
        # Initialize environment
        env = setup_unity_environment(unity_env_path, engine_config, project_root)
        
        try:
            # Run main inference loop
            run_inference_loop(env, model, num_rays)
        finally:
            # Clean up
            print("[INFO] Closing Unity environment")
            env.close()
    
    except Exception as e:
        print(f"[ERROR] Unhandled exception: {e}")
        traceback.print_exc()
    finally:
        print("[INFO] Program terminated")


if __name__ == "__main__":
    main()
