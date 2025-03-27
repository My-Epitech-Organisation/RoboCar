"""
src/inference/run_model.py
Charge le modèle entraîné, lit la config raycast_config.json, et pilote en mode autonome.
"""

import os
import json
import torch
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from .utils_inference import preprocess_observation, postprocess_action
from ..model.model import MLPController

def main():
    # Charger la config Raycast
    config_path = os.path.join("..", "..", "..", "config", "raycast_config.json")
    with open(config_path, "r") as f:
        raycast_config = json.load(f)

    # Charger le modèle
    model_path = os.path.join("..", "..", "..", "data", "processed", "mlp_controller.pt")
    model = MLPController(input_size=12, hidden_size=64, output_size=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Configurer l'environnement
    unity_env_path = os.path.join("..", "..", "..", "RacingSimulatorLinux", "RacingSimulator.x86_64")
    engine_config = EngineConfigurationChannel()
    engine_config.set_configuration_parameters(
        width=raycast_config["graphic_settings"]["width"],
        height=raycast_config["graphic_settings"]["height"],
        quality_level=raycast_config["graphic_settings"]["quality_level"],
        time_scale=raycast_config["time_scale"]
    )

    env = UnityEnvironment(
        file_name=unity_env_path,
        side_channels=[engine_config],
        no_graphics=raycast_config["no_graphics"]
    )
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]

    try:
        while True:
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            for agent_id in decision_steps:
                obs = decision_steps[agent_id].obs[0]  # ex: raycasts
                input_tensor = preprocess_observation(obs)

                with torch.no_grad():
                    action_tensor = model(input_tensor)
                steering, accel = postprocess_action(action_tensor)

                env.set_actions(behavior_name, [[steering, accel]])
            env.step()
    except KeyboardInterrupt:
        print("Inference interrompue.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
