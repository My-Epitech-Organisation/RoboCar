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
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Local imports
import src.inference.utils_inference as utils_inference


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
        # Load the model state and metadata
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Determine if it's a new format with metadata or old format
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # New format with metadata
                model_state = checkpoint['model_state_dict']
                
                # Get metadata if available
                input_size = checkpoint.get('input_size', None)
                num_rays = checkpoint.get('num_rays', None)
                
                print(f"[INFO] Checkpoint contains metadata:")
                print(f"[INFO] Input size: {input_size}")
                print(f"[INFO] Number of rays: {num_rays}")
                
                # Try to determine the model type
                model_type = checkpoint.get('model_type', 'hybrid')  # Default to hybrid if not specified
                
                # Import model builder dynamically
                sys.path.append(os.path.join(project_root, "src", "model"))
                from model_builder import create_model
                
                # Create the appropriate model type
                model = create_model(model_type, input_size=input_size, num_rays=num_rays)
                
                # Load the model state dictionary
                model.load_state_dict(model_state)
                print(f"[INFO] Loaded {model_type} model successfully with metadata")
                return model, model_type
            else:
                # It might be a direct state dict
                print("[INFO] Attempting to load as direct state dict")
                # Try to determine model type from keys
                if any("conv" in key for key in checkpoint.keys()):
                    model_type = "cnn"
                elif any("lstm" in key for key in checkpoint.keys()):
                    model_type = "lstm"
                else:
                    model_type = "simple"
                
                # Get number of rays from config
                config_path = os.path.join(project_root, "config", "agent_config.json")
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    num_rays = config["agents"][0]["nbRay"]
                
                print(f"[INFO] Determined model type: {model_type}")
                print(f"[INFO] Number of rays from config: {num_rays}")
                
                # Create model
                input_size = num_rays + 1  # rays + speed
                model = create_model(model_type, input_size=input_size, num_rays=num_rays)
                
                # Load the direct state dict
                model.load_state_dict(checkpoint)
                return model, model_type
        else:
            # Old direct model format - not supported
            print("[ERROR] Unsupported model format. Please retrain your model.")
            return None, None
    
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        traceback.print_exc()
        return None, None


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
            "-logFile", "unity_inference.log"
        ]
    )

    print("[INFO] Resetting Unity environment")
    env.reset()
    print("[INFO] Unity environment successfully initialized!")

    return env


def get_num_rays_from_config(project_root):
    """Get number of rays from agent configuration."""
    agent_config_path = os.path.join(
        project_root, "config", "agent_config.json"
    )
    try:
        with open(agent_config_path, "r") as f:
            agent_config = json.load(f)
            if "agents" in agent_config and len(agent_config["agents"]) > 0:
                num_rays = agent_config["agents"][0].get("nbRay", 10)
                print(f"[INFO] Number of rays from config: {num_rays}")
                return num_rays
    except Exception as e:
        print(f"[WARNING] Could not load ray count from config: {e}")
    
    # Default if we can't get from config
    print(f"[WARNING] Using default ray count: 10")
    return 10


def process_observations(obs_array, num_rays):
    """Extract and process observations from observation array."""
    # Extract raycasts (first part of the array)
    raycasts = obs_array[:num_rays]
    
    # Normalize raycasts to [0, 1] range
    max_raycast_value = 20.0  # Max value used in training
    raycasts_normalized = np.clip(raycasts / max_raycast_value, 0, 1)
    
    # Extract other observations
    try:
        speed = float(obs_array[-5]) if len(obs_array) >= 5 else 0.0
        steering = float(obs_array[-4]) if len(obs_array) >= 4 else 0.0
        position_x = float(obs_array[-3]) if len(obs_array) >= 3 else 0.0
        position_y = float(obs_array[-2]) if len(obs_array) >= 2 else 0.0
        position_z = float(obs_array[-1]) if len(obs_array) >= 1 else 0.0
    except IndexError:
        speed, steering = 0.0, 0.0
        position_x, position_y, position_z = 0.0, 0.0, 0.0
    
    # Normalize speed like in training
    speed_normalized = min(max(speed / 30.0, 0.0), 1.0)  # Assuming max speed is 30
    
    # Create feature array for model input (same format as training)
    features = np.concatenate([raycasts_normalized, [speed_normalized]])
    
    return features, speed, steering, position_x, position_y, position_z


def run_inference_loop(env, model, num_rays, model_type='simple'):
    """Run the main inference loop."""
    # Get behavior name
    behavior_name = list(env.behavior_specs.keys())[0]
    print(f"[INFO] Detected behavior: {behavior_name}")
    
    # Get initial observations
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    print("[INFO] Starting inference loop...")
    
    # Smoothing parameters for steering
    steering_history = []
    max_history = 3  # Number of previous steering values to consider
    
    # For hybrid model, keep a history of observations
    observation_history = []
    seq_length = 10  # Default sequence length for hybrid model
    hidden_state = None
    
    # Print model type being used
    print(f"[INFO] Using model type: {model_type}")
    
    # Create acceleration controller
    if hasattr(utils_inference, 'AccelerationController'):
        accel_controller = utils_inference.AccelerationController(max_speed=1.0)
    
    frame_count = 0
    try:
        while True:
            # Process observations
            obs_array = decision_steps.obs[0][0]
            features, speed, obs_steering, pos_x, pos_y, pos_z = process_observations(obs_array, num_rays)
            
            # Add to observation history for sequence-based models
            if model_type.lower() == 'hybrid' or model_type.lower() == 'lstm':
                observation_history.append(features)
                if len(observation_history) > seq_length:
                    observation_history.pop(0)
            
            # Convert features to tensor for model
            if model_type.lower() == 'hybrid' or model_type.lower() == 'lstm':
                # If we don't have enough history yet, duplicate the current observation
                while len(observation_history) < seq_length:
                    observation_history.append(features)
                
                # First convert to numpy array, then to tensor (more efficient)
                features_array = np.array(observation_history)
                features_tensor = torch.FloatTensor(features_array).unsqueeze(0)
            else:
                features_tensor = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension
            
            # Get model prediction
            with torch.no_grad():
                if model_type.lower() == 'hybrid':
                    prediction, hidden_state = model(features_tensor, hidden_state)
                    steering_pred = prediction[0, 0].item()  # First output is steering
                    accel_pred = prediction[0, 1].item()     # Second output is acceleration
                else:
                    prediction = model(features_tensor)
                    # Handle different output formats
                    if len(prediction.shape) > 1 and prediction.shape[1] > 1:
                        # Model outputs both steering and acceleration
                        steering_pred = prediction[0, 0].item()
                        accel_pred = prediction[0, 1].item()
                    else:
                        # Model outputs only steering
                        steering_pred = prediction.item()
                        accel_pred = 0.7  # Default acceleration
            
            # Apply smoothing to reduce oscillations
            steering_history.append(steering_pred)
            if len(steering_history) > max_history:
                steering_history.pop(0)
            
            smoothed_steering = sum(steering_history) / len(steering_history)
            
            # Calculate appropriate acceleration
            if hasattr(utils_inference, 'AccelerationController'):
                acceleration = accel_controller.compute_acceleration(
                    accel_pred, 
                    obs_array[:num_rays],
                    smoothed_steering
                )
            else:
                acceleration = 0.7  # Moderate acceleration if no controller
            
            # Create and send actions to the environment
            continuous_actions = np.array([[acceleration, smoothed_steering]], dtype=np.float32)
            action_tuple = ActionTuple(continuous=continuous_actions)
            env.set_actions(behavior_name, action_tuple)
            
            # Display information periodically
            frame_count += 1
            if frame_count % 10 == 0:
                print("\n[INFO] Inference state:")
                print(f"  Speed: {speed:.2f}")
                print(f"  Position: ({pos_x:.2f}, {pos_y:.2f}, {pos_z:.2f})")
                print(f"  Steering prediction: {steering_pred:.4f} (smoothed: {smoothed_steering:.4f})")
                print(f"  Acceleration: {acceleration:.4f}")
                # Show min/max raycast values
                min_ray = min(obs_array[:num_rays])
                max_ray = max(obs_array[:num_rays])
                print(f"  Raycasts range: {min_ray:.2f} to {max_ray:.2f}")
            
            # Step the environment
            env.step()
            
            # Get next observations
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
    except KeyboardInterrupt:
        print("[INFO] Inference interrupted by user")
    except Exception as e:
        print(f"[ERROR] Exception during inference: {e}")
        traceback.print_exc()


def main():
    """Main function for autonomous driving inference."""
    print("[INFO] Starting autonomous driving inference")
    
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
    
    try:
        # Load the trained model
        model, model_type = load_model(project_root)
        
        # Setup Unity environment
        env = setup_unity_environment(unity_env_path, engine_config, project_root)
        
        try:
            # Get number of rays
            num_rays = get_num_rays_from_config(project_root)
            
            # Run the inference loop
            run_inference_loop(env, model, num_rays, model_type)
            
        finally:
            print("[INFO] Closing Unity environment")
            env.close()
            
    except Exception as e:
        print(f"[ERROR] Exception during setup: {e}")
        traceback.print_exc()
        
    print("[INFO] Inference completed")


if __name__ == "__main__":
    main()
