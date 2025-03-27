"""
src/collector/collect_data.py
Script principal pour la collecte de données depuis la simulation Unity.
Lit la config raycast_config.json, lance Unity, puis enregistre les actions utilisateurs.
"""

import os
import csv
import time
import json
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import pygame
from .utils_collector import parse_user_input

def main():
    # 1) Charger la config Raycast
    config_path = os.path.join("..", "..", "..", "config", "raycast_config.json")
    with open(config_path, "r") as f:
        raycast_config = json.load(f)

    # Paramètres de la simulation (file_name, etc.)
    unity_env_path = os.path.join("..", "..", "..", "RacingSimulatorLinux", "RacingSimulator.x86_64")
    
    # 2) Configurer l'environnement via EngineConfigurationChannel
    engine_config = EngineConfigurationChannel()
    engine_config.set_configuration_parameters(
        width=raycast_config["graphic_settings"]["width"],
        height=raycast_config["graphic_settings"]["height"],
        quality_level=raycast_config["graphic_settings"]["quality_level"],
        time_scale=raycast_config["time_scale"]
    )

    # 3) Lancement de l'environnement Unity
    env = UnityEnvironment(
        file_name=unity_env_path,
        side_channels=[engine_config],
        no_graphics=raycast_config["no_graphics"]
    )
    env.reset()

    # Nom du comportement
    behavior_name = list(env.behavior_specs.keys())[0]

    # Initialisation de pygame
    pygame.init()
    joystick_count = pygame.joystick.get_count()
    if joystick_count > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
    else:
        joystick = None

    # Création du fichier CSV de sortie
    output_file = os.path.join("..", "..", "..", "data", "raw", f"session_{int(time.time())}.csv")
    fieldnames = ["timestamp", "steering_input", "acceleration_input", "obs_values"]

    with open(output_file, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        try:
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            while True:
                pygame.event.pump()

                # Récupération des inputs
                steering, accel = parse_user_input(joystick)

                # Lecture des observations (ex: raycasts + speed)
                obs = decision_steps.obs[0][0]  # Hypothèse : le 1er tensor = raycasts
                # ...

                # Ecriture dans le CSV
                writer.writerow({
                    "timestamp": time.time(),
                    "steering_input": steering,
                    "acceleration_input": accel,
                    "obs_values": obs.tolist()
                })

                # Envoi de l'action
                action = [steering, accel]
                env.set_actions(behavior_name, [action])
                env.step()

                decision_steps, terminal_steps = env.get_steps(behavior_name)

        except KeyboardInterrupt:
            print("Collecte interrompue.")
        finally:
            env.close()
            pygame.quit()

if __name__ == "__main__":
    main()
