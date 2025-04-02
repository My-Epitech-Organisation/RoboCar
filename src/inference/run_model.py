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


def load_model(model_path='model_checkpoint.pth'):
    """
    Load the trained neural network model.
    
    Args:
        model_path (str): Path to the model checkpoint file
        
    Returns:
        model: Loaded PyTorch model
        input_size (int): Input size of the model
        num_rays (int): Number of rays used in the model
        model_type (str): Type of model architecture
    """
    try:
        print(f"[INFO] Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Get model metadata
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            
            # Extract metadata
            input_size = checkpoint.get('input_size', None)
            num_rays = checkpoint.get('num_rays', None)
            model_type = checkpoint.get('model_type', None)
            
            # Check if metadata is in a nested dictionary
            if input_size is None and 'metadata' in checkpoint:
                metadata = checkpoint['metadata']
                input_size = metadata.get('input_size', None)
                num_rays = metadata.get('num_rays', None)
                model_type = metadata.get('model_type', None)
        else:
            # Old format where state_dict is directly stored
            model_state = checkpoint
            input_size = None
            num_rays = None
            model_type = None
            
        # Print available metadata
        print("[INFO] Checkpoint contains metadata:")
        if input_size:
            print(f"[INFO] Input size: {input_size}")
        if num_rays:
            print(f"[INFO] Number of rays: {num_rays}")
        if model_type:
            print(f"[INFO] Model type: {model_type}")
            
        # Detect model type from keys if not explicitly provided
        if model_type is None:
            # Look at the keys in the state dict to determine model type
            keys = list(model_state.keys())
            if any('raycast_branch' in key for key in keys):
                model_type = 'multi'
                print(f"[INFO] Detected model type from weights: {model_type}")
            elif any('lstm' in key for key in keys):
                model_type = 'lstm'
                print(f"[INFO] Detected model type from weights: {model_type}")
            elif any('cnn' in key for key in keys):
                if any('fc' in key for key in keys):
                    model_type = 'hybrid'
                else:
                    model_type = 'cnn'
                print(f"[INFO] Detected model type from weights: {model_type}")
            else:
                model_type = 'simple'
                print(f"[INFO] Detected model type from weights: {model_type}")
                
        # Check if the model was trained with only raycasts
        has_other_branch = any('other_branch' in k for k in model_state.keys())
        uses_only_raycasts = not has_other_branch
        
        if uses_only_raycasts:
            print("[INFO] Model was trained using only raycasts (no speed data)")
            
        # Create the appropriate model
        from src.model.model_builder import create_model
        
        # Default to 10 rays if not specified
        if num_rays is None:
            if input_size:
                num_rays = input_size if uses_only_raycasts else input_size - 1
            else:
                num_rays = 10
                input_size = 10 if uses_only_raycasts else 11
                
        print(f"[INFO] Creating model with input_size={input_size}, num_rays={num_rays}")
                
        # Create model based on detected or provided type
        model = create_model(
            model_type=model_type,
            input_size=input_size,
            num_rays=num_rays
        )
        
        # Load the weights
        model.load_state_dict(model_state, strict=False)
        model.eval()  # Set model to evaluation mode
        
        return model, input_size, num_rays, model_type
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


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


def process_observations(obs_array, num_rays, use_only_raycasts=False):
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
    
    # Create feature array for model input (same format as training)
    if use_only_raycasts:
        features = raycasts_normalized
    else:
        # Normalize speed like in training
        speed_normalized = min(max(speed / 30.0, 0.0), 1.0)  # Assuming max speed is 30
        features = np.concatenate([raycasts_normalized, [speed_normalized]])
    
    return features, speed, steering, position_x, position_y, position_z


def run_inference_loop(env, model, model_type, input_size, num_rays, behavior_name, 
                       max_speed=1.0, smooth_factor=0.3, debug=False):
    """
    Main inference loop for autonomous driving.
    """
    print("[INFO] Starting inference loop...")
    print(f"[INFO] Using model type: {model_type}")
    
    # Check if the model was trained with only raycasts
    uses_only_raycasts = input_size == num_rays
    print(f"[INFO] Model uses only raycasts (no speed): {uses_only_raycasts}")
    
    # Initialize variables
    done = False
    prev_steering = 0.0
    steering_history = []
    
    try:
        while not done:
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            obs_array = decision_steps.obs[0][0]
            features, speed, obs_steering, pos_x, pos_y, pos_z = process_observations(
                obs_array, num_rays, uses_only_raycasts
            )
            
            # Convert features to tensor for model
            features_tensor = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension
            
            # Different forward pass handling based on model type
            if model_type and (model_type.lower() == 'hybrid' or model_type.lower() == 'lstm'):
                # Handle models that return a tuple (output, hidden)
                predictions, _ = model(features_tensor)
            else:
                # Standard forward pass
                predictions = model(features_tensor)
                
            # Extract predictions
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                steering_pred = predictions[0, 0].item()
                accel_pred = predictions[0, 1].item()
            else:
                steering_pred = predictions.item()
                accel_pred = 0.7  # Default acceleration
            
            # Apply smoothing to reduce oscillations
            steering_history.append(steering_pred)
            if len(steering_history) > 3:
                steering_history.pop(0)
            
            smoothed_steering = sum(steering_history) / len(steering_history)
            
            # Calculate appropriate acceleration
            acceleration = accel_pred
            
            # Create and send actions to the environment
            continuous_actions = np.array([[acceleration, smoothed_steering]], dtype=np.float32)
            action_tuple = ActionTuple(continuous=continuous_actions)
            env.set_actions(behavior_name, action_tuple)
            
            # Step the environment
            env.step()
            
            # Display information periodically
            print("\n[INFO] Inference state:")
            print(f"  Speed: {speed:.2f}")
            print(f"  Position: ({pos_x:.2f}, {pos_y:.2f}, {pos_z:.2f})")
            print(f"  Steering prediction: {steering_pred:.4f} (smoothed: {smoothed_steering:.4f})")
            print(f"  Acceleration: {acceleration:.4f}")
            
    except Exception as e:
        print(f"[ERROR] Exception during inference: {e}")
        traceback.print_exc()
    finally:
        env.close()
        print("[INFO] Inference completed")


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
        model, input_size, num_rays, model_type = load_model(os.path.join(project_root, "model_checkpoint.pth"))
        
        if model is None:
            print("[ERROR] Failed to load model. Exiting.")
            return
        
        # Setup Unity environment
        env = setup_unity_environment(unity_env_path, engine_config, project_root)
        
        try:
            # Get number of rays
            num_rays = get_num_rays_from_config(project_root)
            
            # Get behavior name
            behavior_name = list(env.behavior_specs.keys())[0]
            
            # Run the inference loop
            run_inference_loop(
                env, 
                model, 
                model_type,
                input_size, 
                num_rays, 
                behavior_name
            )
            
        finally:
            print("[INFO] Closing Unity environment")
            env.close()
            
    except Exception as e:
        print(f"[ERROR] Exception during setup: {e}")
        traceback.print_exc()
        
    print("[INFO] Inference completed")


if __name__ == "__main__":
    main()
