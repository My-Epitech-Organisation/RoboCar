"""
Script de collecte de données depuis la simulation Unity.

Ce script:
1. Lit la configuration depuis raycast_config.json
2. Lance la simulation Unity
3. Enregistre les actions utilisateur et les données des capteurs
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
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description='Collecteur de données Robocar')
    parser.add_argument(
        '--debug-joystick',
        action='store_true',
        help='Activer le débogage du joystick'
    )
    parser.add_argument(
        '--calibrate',
        action='store_true',
        help='Lancer la calibration du joystick au démarrage'
    )
    return parser.parse_args()


def get_project_root():
    """Retourne le chemin racine du projet."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_config(project_root):
    """Charge la configuration depuis le fichier JSON."""
    config_path = os.path.join(project_root, "config", "raycast_config.json")
    print(f"[INFO] Chargement de la configuration depuis {config_path}")

    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERREUR] Fichier de configuration non trouvé: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[ERREUR] Fichier de configuration mal formaté: {config_path}")
        sys.exit(1)


def check_unity_executable(project_root):
    """Vérifie l'existence et les permissions de l'exécutable Unity."""
    unity_env_path = os.path.join(
        project_root,
        "RacingSimulatorLinux",
        "RacingSimulator.x86_64"
    )
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

    return unity_env_path


def init_joystick(debug_mode=False):
    """Initialise et configure le joystick."""
    pygame.init()
    joystick_count = pygame.joystick.get_count()
    joystick = None

    if joystick_count > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"[INFO] Joystick détecté: {joystick.get_name()}")

        if debug_mode:
            print(f"[DEBUG] Nombre d'axes: {joystick.get_numaxes()}")
            print(f"[DEBUG] Nombre de boutons: {joystick.get_numbuttons()}")
            print(f"[DEBUG] Nombre de trackballs: {joystick.get_numballs()}")
            print(f"[DEBUG] Nombre de hats: {joystick.get_numhats()}")
    else:
        print("[INFO] Aucun joystick détecté, utilisation du clavier uniquement")

    return joystick


def setup_unity_environment(unity_env_path, engine_config, project_root):
    """Configure et initialise l'environnement Unity."""
    agent_config_path = os.path.join(project_root, "config", "agent_config.json")
    print(f"[INFO] Configuration des agents récupérée dans {agent_config_path}")

    print("[INFO] Lancement et connexion à l'environnement Unity...")
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

    return env, agent_config_path


def setup_data_collection(env, project_root):
    """Configure les structures pour la collecte de données."""
    # Récupération des informations de comportement
    behavior_name = list(env.behavior_specs.keys())[0]
    print(f"[INFO] Comportement détecté: {behavior_name}")

    behavior_spec = env.behavior_specs[behavior_name]
    print(f"[INFO] Nombre d'observations: {len(behavior_spec.observation_specs)}")

    for i, obs_spec in enumerate(behavior_spec.observation_specs):
        print(
            f"[INFO] Observation {i}: forme={obs_spec.shape}, "
            f"type={obs_spec.observation_type}"
        )

    print(
        f"[INFO] Type d'action: Continue avec "
        f"{behavior_spec.action_spec.continuous_size} dimensions"
    )

    # Préparation du fichier de sortie CSV
    output_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"session_{int(time.time())}.csv")
    print(f"[INFO] Écriture des données dans {output_file}")

    return behavior_name, behavior_spec, output_file


def collect_data_loop(env, behavior_name, output_file, joystick, debug_joystick=False):
    """Exécute la boucle principale de collecte de données."""
    fieldnames = [
        "timestamp", "steering_input", "acceleration_input", "raycasts", "speed"
    ]

    with open(output_file, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        print("[INFO] Récupération des premières observations...")
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        print("[INFO] Observations récupérées, démarrage de la boucle principale")
        print("[INFO] Collecte de données en cours. Appuyez sur Ctrl+C pour arrêter.")
        print("[INFO] Contrôles: Flèches directionnelles ou ZQSD pour diriger la voiture")
        print("[INFO] Appuyez sur 'c' pour calibrer le joystick")

        frame_count = 0
        debug_count = 0
        
        # Variable pour suivre l'état de la touche 'c'
        calibration_requested = False

        while True:
            # Vérifier si pygame est toujours initialisé, sinon le réinitialiser
            if not pygame.get_init():
                print("[INFO] Réinitialisation de pygame après la calibration...")
                pygame.init()
                if joystick and not joystick.get_init():
                    joystick.init()

            # Traitement des événements Pygame pour le joystick
            pygame.event.pump()

            # Vérifier si la touche 'c' est pressée pour la calibration
            if key_states['c'] and not calibration_requested:
                calibration_requested = True
                print("[INFO] Calibration du joystick demandée...")
                
                # Pause de la collecte de données
                if joystick:
                    print("[INFO] Lancement de la calibration...")
                    calibrate_joystick(joystick)
                    print("[INFO] Calibration terminée, reprise de la collecte.")
                    
                    # S'assurer que pygame est toujours initialisé après la calibration
                    if not pygame.get_init():
                        print("[INFO] Réinitialisation de pygame après la calibration...")
                        pygame.init()
                        if not joystick.get_init():
                            joystick.init()
                else:
                    print("[ERREUR] Aucun joystick détecté pour la calibration.")
                
            # Réinitialiser l'état de la demande de calibration quand 'c' est relâché
            if not key_states['c']:
                calibration_requested = False

            # Récupération des inputs utilisateur
            steering, accel = parse_user_input(joystick)

            # Affichage des valeurs du joystick en mode debug
            if debug_joystick and joystick and debug_count % 30 == 0:
                print("\n[DEBUG JOYSTICK] Valeurs brutes des axes:")
                for i in range(joystick.get_numaxes()):
                    print(f"  Axe {i}: {joystick.get_axis(i):.3f}")
                print(
                    f"[DEBUG] Direction: {steering:.3f}, "
                    f"Accélération: {accel:.3f}"
                )
            debug_count += 1

            # Lecture des observations
            raycasts = decision_steps.obs[0][0].tolist()
            speed = 0.0

            if len(decision_steps.obs) > 1:
                speed = float(decision_steps.obs[1][0][0])

            # Affichage périodique des informations
            frame_count += 1
            if frame_count % 10 == 0:
                print(
                    f"Commandes: direction={steering:.2f}, "
                    f"accélération={accel:.2f}, vitesse={speed:.2f}"
                )

            # Enregistrement des données
            writer.writerow({
                "timestamp": time.time(),
                "steering_input": steering,
                "acceleration_input": accel,
                "raycasts": str(raycasts),
                "speed": speed
            })

            # Envoi des actions à la simulation
            continuous_actions = np.array([[accel, steering]], dtype=np.float32)
            action_tuple = ActionTuple(continuous=continuous_actions)
            env.set_actions(behavior_name, action_tuple)
            env.step()

            decision_steps, terminal_steps = env.get_steps(behavior_name)


def main():
    """Fonction principale du collecteur de données."""
    args = parse_arguments()
    project_root = get_project_root()
    raycast_config = load_config(project_root)
    unity_env_path = check_unity_executable(project_root)

    # Configuration du canal de communication Unity
    print("[INFO] Configuration du canal de communication avec Unity")
    engine_config = EngineConfigurationChannel()
    engine_config.set_configuration_parameters(
        width=raycast_config["graphic_settings"]["width"],
        height=raycast_config["graphic_settings"]["height"],
        quality_level=raycast_config["graphic_settings"]["quality_level"],
        time_scale=raycast_config["time_scale"]
    )

    # Initialisation de l'écouteur de clavier global
    print("[INFO] Initialisation de l'écouteur de clavier global")
    keyboard_listener = setup_keyboard_listener()
    keyboard_listener.start()
    print("[INFO] Écouteur de clavier démarré - vous pouvez utiliser les flèches ou ZQSD")

    # Initialisation du joystick
    joystick = init_joystick(args.debug_joystick)
    
    # Lancer la calibration au démarrage si demandé
    if args.calibrate and joystick:
        print("[INFO] Calibration du joystick au démarrage...")
        calibrate_joystick(joystick)

    try:
        # Configuration de l'environnement Unity
        env, _ = setup_unity_environment(unity_env_path, engine_config, project_root)

        try:
            # Configuration de la collecte de données
            behavior_name, _, output_file = setup_data_collection(env, project_root)

            try:
                # Boucle de collecte de données
                collect_data_loop(
                    env, behavior_name, output_file, joystick, args.debug_joystick
                )
            except KeyboardInterrupt:
                print("[INFO] Collecte interrompue par l'utilisateur.")
            except Exception as e:
                print(f"[ERREUR] Exception pendant la collecte: {e}")
                traceback.print_exc()
        except Exception as e:
            print(f"[ERREUR] Exception lors de la configuration: {e}")
            traceback.print_exc()
        finally:
            print("[INFO] Fermeture de l'environnement Unity")
            env.close()
    except Exception as e:
        print(f"[ERREUR] Exception lors de l'initialisation de l'environnement Unity: {e}")
        traceback.print_exc()
    finally:
        print("[INFO] Arrêt de l'écouteur de clavier")
        keyboard_listener.stop()
        print("[INFO] Fermeture de pygame")
        pygame.quit()
        print("[INFO] Programme terminé.")


if __name__ == "__main__":
    main()
