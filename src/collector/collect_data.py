"""
src/collector/collect_data.py
Script principal pour la collecte de données depuis la simulation Unity.
Lit la config raycast_config.json, lance Unity, puis enregistre les actions utilisateurs.
"""

import os
import csv
import time
import json
import sys
import subprocess
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import pygame
import utils_collector
from utils_collector import parse_user_input

def main():
    # Définir le chemin du projet racine
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # 1) Charger la config Raycast (chemin absolu)
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

    # Paramètres de la simulation (file_name, etc.)
    unity_env_path = os.path.join(project_root, "RacingSimulatorLinux", "RacingSimulator.x86_64")
    print(f"[INFO] Chemin vers l'environnement Unity: {unity_env_path}")
    
    if not os.path.exists(unity_env_path):
        print(f"[ERREUR] L'exécutable Unity n'existe pas: {unity_env_path}")
        sys.exit(1)

    # Vérifier si l'exécutable a les permissions d'exécution
    if not os.access(unity_env_path, os.X_OK):
        print(f"[AVERTISSEMENT] Ajout des permissions d'exécution à {unity_env_path}")
        try:
            os.chmod(unity_env_path, 0o755)
        except Exception as e:
            print(f"[ERREUR] Impossible d'ajouter les permissions d'exécution: {e}")
            sys.exit(1)

    # 2) Configurer l'environnement via EngineConfigurationChannel
    print("[INFO] Configuration du canal de communication avec Unity")
    engine_config = EngineConfigurationChannel()
    engine_config.set_configuration_parameters(
        width=raycast_config["graphic_settings"]["width"],
        height=raycast_config["graphic_settings"]["height"],
        quality_level=raycast_config["graphic_settings"]["quality_level"],
        time_scale=raycast_config["time_scale"]
    )

    # Initialisation de pygame avant le lancement de Unity
    print("[INFO] Initialisation de pygame")
    pygame.init()
    joystick_count = pygame.joystick.get_count()
    if joystick_count > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"[INFO] Joystick détecté: {joystick.get_name()}")
    else:
        joystick = None
        print("[INFO] Aucun joystick détecté, utilisation du clavier")

    # Créer un fichier de configuration pour les agents
    agent_config_path = os.path.join(project_root, "config", "agent_config.json")
    agent_config = {
        "agents": [
            {
                "fov": 180,
                "nbRay": 10
            },
            {
                "fov": 48,
                "nbRay": 36
            }
        ]
    }
    
    with open(agent_config_path, 'w') as f:
        json.dump(agent_config, f, indent=2)
    
    print(f"[INFO] Configuration des agents écrite dans {agent_config_path}")

    # Option pour lancer Unity manuellement ou via le script
    print("\nChoisissez une option:")
    print("1. Lancer Unity automatiquement via le script")
    print("2. Je vais lancer Unity manuellement")
    option = input("Votre choix (1/2): ")
    
    if option == "2":
        print("\n[INFO] Lancez Unity manuellement avec la commande suivante:")
        unity_cmd = f"{unity_env_path} --mlagents-port 5004 --config-path={agent_config_path}"
        print(f"    {unity_cmd}")
        print("[INFO] Attendez que Unity soit complètement chargé puis appuyez sur Entrée...")
        input()
    else:
        # Lancer Unity en arrière-plan
        print("[INFO] Lancement de Unity en arrière-plan...")
        unity_process = subprocess.Popen([
            unity_env_path,
            "--mlagents-port", "5004",
            "--config-path", agent_config_path
        ])
        print("[INFO] Attente du démarrage de Unity (15 secondes)...")
        time.sleep(15)  # Attendre que Unity démarre complètement
    
    # 3) Connexion à l'environnement Unity
    print("[INFO] Tentative de connexion à l'environnement Unity...")
    try:
        # Au lieu de lancer Unity, on se connecte à l'environnement existant
        env = UnityEnvironment(
            file_name=None,  # Connexion à un environnement existant
            side_channels=[engine_config],
            base_port=5004,
            worker_id=0,
            timeout_wait=60
        )
        
        print("[INFO] Réinitialisation de l'environnement Unity")
        env.reset()
        print("[INFO] Environnement Unity initialisé avec succès!")

        # Nom du comportement
        behavior_name = list(env.behavior_specs.keys())[0]
        print(f"[INFO] Comportement détecté: {behavior_name}")
        
        # Affichage des spécifications du comportement pour le débogage
        behavior_spec = env.behavior_specs[behavior_name]
        print(f"[INFO] Nombre d'observations: {len(behavior_spec.observation_specs)}")
        for i, obs_spec in enumerate(behavior_spec.observation_specs):
            print(f"[INFO] Observation {i}: forme={obs_spec.shape}, type={obs_spec.observation_type}")
        print(f"[INFO] Type d'action: {behavior_spec.action_spec.name}")
        print(f"[INFO] Taille d'action continue: {behavior_spec.action_spec.continuous_size}")

        # Création du fichier CSV de sortie
        output_dir = os.path.join(project_root, "data", "raw")
        os.makedirs(output_dir, exist_ok=True)  # Créer le répertoire s'il n'existe pas
        output_file = os.path.join(output_dir, f"session_{int(time.time())}.csv")
        print(f"[INFO] Écriture des données dans {output_file}")
        
        fieldnames = ["timestamp", "steering_input", "acceleration_input", "raycasts", "speed"]

        with open(output_file, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            try:
                print("[INFO] Récupération des premières observations...")
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                print("[INFO] Observations récupérées, démarrage de la boucle principale")
                print("[INFO] Collecte de données en cours. Appuyez sur Ctrl+C pour arrêter.")

                frame_count = 0
                while True:
                    pygame.event.pump()

                    # Récupération des inputs
                    steering, accel = parse_user_input(joystick)

                    # Lecture des observations (raycasts + speed)
                    raycasts = decision_steps.obs[0][0].tolist()  # 1er tensor = raycasts
                    speed = 0.0
                    
                    if len(decision_steps.obs) > 1:  # S'il y a un deuxième tensor, c'est probablement la vitesse
                        speed = float(decision_steps.obs[1][0][0])
                    
                    # Afficher les informations toutes les 10 frames pour ne pas surcharger la console
                    frame_count += 1
                    if frame_count % 10 == 0:
                        print(f"Commandes: direction={steering:.2f}, accélération={accel:.2f}, vitesse={speed:.2f}")

                    # Ecriture dans le CSV
                    writer.writerow({
                        "timestamp": time.time(),
                        "steering_input": steering,
                        "acceleration_input": accel,
                        "raycasts": str(raycasts),  # Convertir en string pour le CSV
                        "speed": speed
                    })

                    # Envoi de l'action
                    action = [steering, accel]
                    env.set_actions(behavior_name, [action])
                    env.step()

                    decision_steps, terminal_steps = env.get_steps(behavior_name)

            except KeyboardInterrupt:
                print("[INFO] Collecte interrompue par l'utilisateur.")
            except Exception as e:
                print(f"[ERREUR] Exception pendant la collecte: {e}")
                import traceback
                traceback.print_exc()
            finally:
                print("[INFO] Fermeture de l'environnement Unity")
                env.close()
    except Exception as e:
        print(f"[ERREUR] Exception lors de l'initialisation de l'environnement Unity: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[INFO] Fermeture de pygame")
        pygame.quit()
        
        # Terminer le processus Unity si on l'a démarré
        if option != "2" and 'unity_process' in locals():
            print("[INFO] Arrêt de Unity...")
            try:
                unity_process.terminate()
                unity_process.wait(timeout=5)
            except:
                unity_process.kill()
                
        print("[INFO] Programme terminé.")

if __name__ == "__main__":
    main()
