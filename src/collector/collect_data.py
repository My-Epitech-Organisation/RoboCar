"""
Script principal pour la collecte de données depuis la simulation Unity.

Ce script:
1. Lit la configuration de raycast_config.json
2. Lance la simulation Unity
3. Enregistre les actions utilisateur et les données capteurs
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
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from utils_collector import parse_user_input


def main():
    """Fonction principale du collecteur de données."""
    # Configuration des arguments de ligne de commande
    parser = argparse.ArgumentParser(description='Collecteur de données Robocar')
    parser.add_argument('--debug-joystick', action='store_true', 
                        help='Activer le débogage du joystick')
    args = parser.parse_args()

    # Définition du chemin racine du projet
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Chargement de la configuration Raycast
    config_path = os.path.join(project_root, "config", "raycast_config.json")
    print(f"[INFO] Chargement de la configuration depuis {config_path}")
    try:
        with open(config_path, "r") as f:
            raycast_config = json.load(f)
    except FileNotFoundError:
        print(f"[ERREUR] Fichier de configuration non trouvé: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[ERREUR] Fichier de configuration mal formaté: {config_path}")
        sys.exit(1)

    # Vérification de l'exécutable Unity
    unity_env_path = os.path.join(project_root, "RacingSimulatorLinux", 
                                 "RacingSimulator.x86_64")
    print(f"[INFO] Chemin vers l'environnement Unity: {unity_env_path}")

    if not os.path.exists(unity_env_path):
        print(f"[ERREUR] L'exécutable Unity n'existe pas: {unity_env_path}")
        sys.exit(1)

    # Ajout des permissions d'exécution si nécessaire
    if not os.access(unity_env_path, os.X_OK):
        print(f"[AVERTISSEMENT] Ajout des permissions d'exécution à {unity_env_path}")
        try:
            os.chmod(unity_env_path, 0o755)
        except Exception as e:
            print(f"[ERREUR] Impossible d'ajouter les permissions d'exécution: {e}")
            sys.exit(1)

    # Configuration du canal de communication Unity
    print("[INFO] Configuration du canal de communication avec Unity")
    engine_config = EngineConfigurationChannel()
    engine_config.set_configuration_parameters(
        width=raycast_config["graphic_settings"]["width"],
        height=raycast_config["graphic_settings"]["height"],
        quality_level=raycast_config["graphic_settings"]["quality_level"],
        time_scale=raycast_config["time_scale"]
    )

    # Initialisation de pygame pour les entrées utilisateur
    print("[INFO] Initialisation de pygame")
    pygame.init()
    joystick_count = pygame.joystick.get_count()
    joystick = None

    if joystick_count > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"[INFO] Joystick détecté: {joystick.get_name()}")

        if args.debug_joystick:
            print(f"[DEBUG] Nombre d'axes: {joystick.get_numaxes()}")
            print(f"[DEBUG] Nombre de boutons: {joystick.get_numbuttons()}")
            print(f"[DEBUG] Nombre de trackballs: {joystick.get_numballs()}")
            print(f"[DEBUG] Nombre de hats: {joystick.get_numhats()}")
    else:
        print("[INFO] Aucun joystick détecté, utilisation du clavier")

    # Chargement de la configuration des agents
    agent_config_path = os.path.join(project_root, "config", "agent_config.json")
    print(f"[INFO] Configuration des agents récupérée dans {agent_config_path}")

    # Connexion à l'environnement Unity
    print("[INFO] Lancement et connexion à l'environnement Unity...")
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

        print("[INFO] Réinitialisation de l'environnement Unity")
        env.reset()
        print("[INFO] Environnement Unity initialisé avec succès!")

        # Récupération des informations de comportement
        behavior_name = list(env.behavior_specs.keys())[0]
        print(f"[INFO] Comportement détecté: {behavior_name}")

        behavior_spec = env.behavior_specs[behavior_name]
        print(f"[INFO] Nombre d'observations: {len(behavior_spec.observation_specs)}")
        for i, obs_spec in enumerate(behavior_spec.observation_specs):
            print(f"[INFO] Observation {i}: forme={obs_spec.shape}, "
                  f"type={obs_spec.observation_type}")
        print(f"[INFO] Type d'action: Continue avec "
              f"{behavior_spec.action_spec.continuous_size} dimensions")

        # Préparation du fichier de sortie CSV
        output_dir = os.path.join(project_root, "data", "raw")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"session_{int(time.time())}.csv")
        print(f"[INFO] Écriture des données dans {output_file}")

        fieldnames = ["timestamp", "steering_input", "acceleration_input", 
                      "raycasts", "speed"]

        with open(output_file, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            try:
                print("[INFO] Récupération des premières observations...")
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                print("[INFO] Observations récupérées, démarrage de la boucle principale")
                print("[INFO] Collecte de données en cours. Appuyez sur Ctrl+C pour arrêter.")

                frame_count = 0
                debug_count = 0
                while True:
                    pygame.event.pump()

                    # Récupération des inputs utilisateur
                    steering, accel = parse_user_input(joystick)
        
                    # Affichage des valeurs du joystick en mode debug
                    if args.debug_joystick and joystick and debug_count % 30 == 0:
                        print("\n[DEBUG JOYSTICK] Valeurs brutes des axes:")
                        for i in range(joystick.get_numaxes()):
                            print(f"  Axe {i}: {joystick.get_axis(i):.3f}")
                        print(f"[DEBUG] Direction: {steering:.3f}, "
                              f"Accélération: {accel:.3f}")
                    debug_count += 1

                    # Lecture des observations
                    raycasts = decision_steps.obs[0][0].tolist()
                    speed = 0.0
        
                    if len(decision_steps.obs) > 1:
                        speed = float(decision_steps.obs[1][0][0])
        
                    # Affichage périodique des informations
                    frame_count += 1
                    if frame_count % 10 == 0:
                        print(f"Commandes: direction={steering:.2f}, "
                              f"accélération={accel:.2f}, vitesse={speed:.2f}")

                    # Enregistrement des données
                    writer.writerow({
                        "timestamp": time.time(),
                        "steering_input": steering,
                        "acceleration_input": accel,
                        "raycasts": str(raycasts),
                        "speed": speed
                    })

                    # Envoi des actions à la simulation
                    continuous_actions = np.array([[steering, accel]], dtype=np.float32)
                    action_tuple = ActionTuple(continuous=continuous_actions)
                    env.set_actions(behavior_name, action_tuple)
                    env.step()

                    decision_steps, terminal_steps = env.get_steps(behavior_name)

            except KeyboardInterrupt:
                print("[INFO] Collecte interrompue par l'utilisateur.")
            except Exception as e:
                print(f"[ERREUR] Exception pendant la collecte: {e}")
                traceback.print_exc()
            finally:
                print("[INFO] Fermeture de l'environnement Unity")
                env.close()
    except Exception as e:
        print(f"[ERREUR] Exception lors de l'initialisation de l'environnement Unity: {e}")
        traceback.print_exc()
    finally:
        print("[INFO] Fermeture de pygame")
        pygame.quit()
        print("[INFO] Programme terminé.")


if __name__ == "__main__":
    main()
