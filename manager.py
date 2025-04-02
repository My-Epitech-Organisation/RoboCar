#!/usr/bin/env python3
"""
manager.py - Script de gestion avancé du projet RoboCar

Ce script gère:
1) La création et mise à jour d'un environnement virtuel (.venv)
2) L'installation des dépendances (requirements.txt)
3) L'interface complète pour tous les processus:
   - Collecte de données avec options avancées
   - Entraînement de modèles avec différentes configurations
   - Inférence avec paramètres personnalisables
   - Visualisation et analyse des données et résultats
   - Gestion et maintenance du projet
"""

import os
import subprocess
import sys
import platform
import json
import glob
import shutil
from datetime import datetime


# ===================== CONFIGURATION ET SETUP =====================

def check_python_version():
    """
    Vérifie la disponibilité de Python 3.10.12 sur le système.

    Returns:
        str: Chemin vers l'exécutable Python 3.10.12 si disponible.

    Exits:
        Programme si la version requise n'est pas disponible.
    """
    try:
        result = subprocess.run(
            ["python3.10", "--version"],
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode == 0 and "3.10" in result.stdout:
            if "3.10.12" in result.stdout:
                print("[INFO] Python 3.10.12 trouvé.")
                return "python3.10"
            else:
                version = result.stdout.strip()
                print(f"[ATTENTION] Python 3.10 trouvé mais pas 3.10.12 ({version})")
                choice = input("Continuer avec cette version ? (o/n): ").lower()
                if choice == 'o':
                    return "python3.10"
                else:
                    print("[ERREUR] Python 3.10.12 requis pour continuer.")
                    sys.exit(1)
        else:
            print("[ERREUR] Python 3.10 n'est pas disponible sur ce système.")
            print("Veuillez installer Python 3.10.12 avant de continuer.")
            sys.exit(1)
    except Exception as e:
        print(f"[ERREUR] Lors de la vérification de Python: {e}")
        sys.exit(1)


def check_and_setup_venv():
    """
    Vérifie et configure l'environnement virtuel Python.

    Crée un nouvel environnement si nécessaire et installe les dépendances
    depuis requirements.txt.
    """
    venv_dir = ".venv"

    python_exec = check_python_version()

    is_windows = platform.system() == "Windows"
    bin_dir = "Scripts" if is_windows else "bin"
    activate_path = os.path.join(venv_dir, bin_dir, "activate")
    activate_cmd = f"call {activate_path}" if is_windows else f"source {activate_path}"

    if not os.path.isdir(venv_dir):
        print("[INFO] Environnement virtuel (.venv) introuvable, création en cours...")
        subprocess.run([python_exec, "-m", "venv", venv_dir], check=True)
    else:
        print("[INFO] Environnement virtuel déjà présent.")
        print("[INFO] Vérification de la version Python dans l'environnement...")

        py_path = os.path.join(venv_dir, bin_dir,
                              "python.exe" if is_windows else "python")

        if os.path.exists(py_path):
            result = subprocess.run(
                [py_path, "--version"],
                capture_output=True,
                text=True,
                check=False
            )

            if "3.10" not in result.stdout:
                version = result.stdout.strip()
                print(f"[ATTENTION] Version Python incorrecte: {version}")
                choice = input("Recréer l'environnement ? (o/n): ").lower()

                if choice == 'o':
                    print("[INFO] Suppression de l'ancien environnement...")
                    import shutil
                    shutil.rmtree(venv_dir)
                    print("[INFO] Création du nouvel environnement...")
                    subprocess.run([python_exec, "-m", "venv", venv_dir], check=True)

    print("[INFO] Installation/mise à jour des dépendances...")
    cmd = f"{activate_cmd} && pip install -r requirements.txt"
    subprocess.run(cmd, shell=True, check=True)
    print("[INFO] Dépendances installées/à jour.")


def get_activate_command():
    """
    Retourne la commande d'activation de l'environnement virtuel.

    Returns:
        tuple: (chemin d'activation, commande d'activation complète)
    """
    is_windows = platform.system() == "Windows"
    bin_dir = "Scripts" if is_windows else "bin"
    activate_path = os.path.join(".venv", bin_dir, "activate")
    activate_cmd = f"call {activate_path}" if is_windows else f"source {activate_path}"
    return activate_path, activate_cmd


def run_script(script_path, args=None, description=None):
    """
    Lance un script Python dans l'environnement virtuel avec des arguments optionnels.

    Args:
        script_path (str): Chemin vers le script à exécuter
        args (list): Liste d'arguments à passer au script
        description (str): Description pour le message d'information
    """
    if description:
        print(f"[INFO] Lancement de {description}...")

    _, activate_cmd = get_activate_command()
    cmd = f"{activate_cmd} && python {script_path}"

    if args:
        cmd += " " + " ".join(args)

    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"[ERREUR] Échec de l'exécution de {script_path}")
        return False


def check_system_resources():
    """
    Vérifie et affiche les ressources système disponibles.
    """
    print("\n=== Ressources système ===")

    # Vérifier l'espace disque
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        print(f"Espace disque: {free // (2**30)} GB libre sur {total // (2**30)} GB")
    except Exception as e:
        print(f"Impossible de vérifier l'espace disque: {e}")

    # Vérifier la disponibilité du GPU
    try:
        _, activate_cmd = get_activate_command()
        cmd = f"{activate_cmd} && python -c \"import torch; print('CUDA disponible:', torch.cuda.is_available()); print('Appareils GPU:', torch.cuda.device_count())\""
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout.strip())
    except Exception as e:
        print(f"Impossible de vérifier le GPU: {e}")

    # Vérifier la RAM
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"Mémoire RAM: {ram.available // (2**20)} MB libre sur {ram.total // (2**20)} MB")
    except ImportError:
        print("Module psutil non disponible pour vérifier la RAM")
    except Exception as e:
        print(f"Impossible de vérifier la RAM: {e}")

    print("")


# ===================== COLLECTE DE DONNÉES =====================

def update_agent_config(fov=None, nb_ray=None):
    """
    Met à jour la configuration des agents (FOV, nombre de raycasts).

    Args:
        fov (int): Champ de vision en degrés (1-180)
        nb_ray (int): Nombre de raycasts (1-50)
    """
    config_path = "config/agent_config.json"

    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # Charger la configuration existante ou créer une nouvelle
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {"agents": [{"fov": 180, "nbRay": 10}]}

    # Mettre à jour les valeurs si spécifiées
    if fov is not None:
        config["agents"][0]["fov"] = min(max(1, fov), 180)

    if nb_ray is not None:
        config["agents"][0]["nbRay"] = min(max(1, nb_ray), 50)

    # Enregistrer la configuration
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"[INFO] Configuration mise à jour: FOV={config['agents'][0]['fov']}, nbRay={config['agents'][0]['nbRay']}")


def collect_data_menu():
    """Affiche le menu de collecte de données avec options avancées."""
    while True:
        print("\n=== MENU COLLECTE DE DONNÉES ===")
        print("1) Collecte standard")
        print("2) Collecte avec calibration du joystick")
        print("3) Configurer les capteurs (FOV, nombre de raycasts)")
        print("4) Collecter données de récupération")
        print("5) Afficher les sessions existantes")
        print("6) Retour au menu principal")

        choice = input("Votre choix: ").strip()

        if choice == "1":
            # Collecte standard
            session_name = input("Nom de la session (laissez vide pour auto): ").strip()
            args = ["--session_name", session_name] if session_name else []
            run_script("src/collector/collect_data.py", args, "la collecte de données")

        elif choice == "2":
            # Collecte avec calibration
            run_script("src/collector/collect_data.py", ["--calibrate"],
                      "la collecte avec calibration du joystick")

        elif choice == "3":
            # Configuration des capteurs
            try:
                fov = int(input("Champ de vision (1-180 degrés): ").strip())
                nb_ray = int(input("Nombre de raycasts (1-50): ").strip())
                update_agent_config(fov, nb_ray)
            except ValueError:
                print("[ERREUR] Veuillez entrer des valeurs numériques valides.")

        elif choice == "4":
            # Collecte de données de récupération
            print("[INFO] Mode collecte de récupération")
            print("Ce mode est conçu pour capturer des données de correction après des erreurs.")
            print("Placez la voiture près des bords et corrigez la trajectoire.")
            run_script("src/collector/collect_data.py", ["--session_name", "recovery_training"],
                      "la collecte de données de récupération")

        elif choice == "5":
            # Afficher les sessions existantes
            list_data_sessions()

        elif choice == "6":
            # Retour
            break

        else:
            print("[ERREUR] Choix invalide. Réessayez.")


def list_data_sessions():
    """Affiche la liste des sessions de données disponibles."""
    data_dir = "data/raw"
    if not os.path.exists(data_dir):
        print("[INFO] Aucune donnée collectée pour l'instant.")
        return

    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        print("[INFO] Aucune donnée collectée pour l'instant.")
        return

    print("\n=== SESSIONS DE DONNÉES DISPONIBLES ===")
    total_size = 0
    for i, file_path in enumerate(sorted(files), 1):
        file_name = os.path.basename(file_path)
        size_kb = os.path.getsize(file_path) / 1024
        total_size += size_kb

        # Compter les lignes (échantillons)
        try:
            with open(file_path, 'r') as f:
                lines = sum(1 for _ in f) - 1  # -1 pour l'en-tête
            print(f"{i}. {file_name} - {size_kb:.1f} KB - {lines} échantillons")
        except Exception:
            print(f"{i}. {file_name} - {size_kb:.1f} KB")

    print(f"\nTotal: {len(files)} sessions, {total_size/1024:.2f} MB")


# ===================== ENTRAÎNEMENT DE MODÈLES =====================

def training_menu():
    """Affiche le menu d'entraînement avec options avancées."""
    while True:
        print("\n=== MENU ENTRAÎNEMENT ===")
        print("1) Entraînement rapide (valeurs par défaut)")
        print("2) Entraînement personnalisé")
        print("3) Analyser les résultats d'entraînement")
        print("4) Gérer les modèles entraînés")
        print("5) Visualiser les données d'entraînement")
        print("6) Retour au menu principal")

        choice = input("Votre choix: ").strip()

        if choice == "1":
            # Entraînement rapide
            run_script("src/model/train.py", description="l'entraînement rapide")

        elif choice == "2":
            # Entraînement personnalisé
            custom_training()

        elif choice == "3":
            # Analyser les résultats
            analyze_training_results()

        elif choice == "4":
            # Gérer les modèles
            manage_models()

        elif choice == "5":
            # Visualiser les données
            visualize_training_data()

        elif choice == "6":
            # Retour
            break

        else:
            print("[ERREUR] Choix invalide. Réessayez.")


def custom_training():
    """Interface pour configurer un entraînement personnalisé."""
    print("\n=== CONFIGURATION D'ENTRAÎNEMENT PERSONNALISÉ ===")

    # Type de modèle
    print("\nChoix du type de modèle:")
    print("1) Simple (MLP basique)")
    print("2) CNN (Réseau convolutionnel)")
    print("3) LSTM (Réseau récurrent)")
    print("4) Hybrid (Architecture combinée CNN+LSTM)")
    print("5) Multi (Multi-entrées avec CNN pour raycasts) - Recommandé")

    model_types = {
        "1": "simple",
        "2": "cnn",
        "3": "lstm",
        "4": "hybrid",
        "5": "multi"
    }

    model_choice = input("Choix du modèle [5]: ").strip() or "5"
    model_type = model_types.get(model_choice, "multi")

    # Nombre d'époques
    epochs = input("Nombre d'époques [100]: ").strip() or "100"

    # Taille des batchs
    batch_size = input("Taille des batchs [64]: ").strip() or "64"

    # Taux d'apprentissage
    learning_rate = input("Taux d'apprentissage [0.001]: ").strip() or "0.001"

    # Utiliser uniquement les raycasts
    raycast_only_choice = input("Utiliser uniquement les raycasts? (o/n) [o]: ").strip().lower() or "o"
    raycast_flag = "--use_only_raycasts" if raycast_only_choice == "o" else "--use_all_inputs"

    # Augmentation des données
    augment_choice = input("Activer l'augmentation des données? (o/n) [o]: ").strip().lower() or "o"
    augment_flag = "--augment" if augment_choice == "o" else "--no_augment"

    # Construire la commande
    args = [
        "--model_type", model_type,
        "--epochs", epochs,
        "--batch_size", batch_size,
        "--learning_rate", learning_rate,
        raycast_flag,
        augment_flag
    ]

    # Confirmation
    print("\nConfiguration d'entraînement:")
    print(f"- Modèle: {model_type}")
    print(f"- Époques: {epochs}")
    print(f"- Batch size: {batch_size}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Utiliser uniquement les raycasts: {'Oui' if raycast_only_choice == 'o' else 'Non'}")
    print(f"- Augmentation: {'Activée' if augment_choice == 'o' else 'Désactivée'}")

    confirm = input("\nLancer l'entraînement? (o/n): ").lower().strip()
    if confirm == "o":
        run_script("src/model/train.py", args,
                  f"l'entraînement personnalisé (modèle {model_type})")


def analyze_training_results():
    """Analyse les résultats d'entraînement."""
    model_dir = "models"
    if not os.path.exists(model_dir):
        print("[INFO] Aucun résultat d'entraînement disponible.")
        return

    # Trouver tous les dossiers d'entraînement
    training_dirs = [d for d in os.listdir(model_dir)
                    if os.path.isdir(os.path.join(model_dir, d))]

    if not training_dirs:
        print("[INFO] Aucun résultat d'entraînement disponible.")
        return

    print("\n=== SESSIONS D'ENTRAÎNEMENT DISPONIBLES ===")
    for i, dir_name in enumerate(sorted(training_dirs), 1):
        metrics_path = os.path.join(model_dir, dir_name, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            steer_mae = metrics["steering"]["mae"]
            accel_mae = metrics["acceleration"]["mae"]
            print(f"{i}. {dir_name} - Direction MAE: {steer_mae:.4f}, Accélération MAE: {accel_mae:.4f}")
        else:
            print(f"{i}. {dir_name} - (Métriques non disponibles)")

    choice = input("\nSélectionnez une session à analyser (n° ou r pour retour): ").strip()
    if choice.lower() == "r":
        return

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(training_dirs):
            session_dir = training_dirs[idx]
            # Ouvrir les graphiques d'analyse
            graph_files = [
                os.path.join(model_dir, session_dir, "predictions_comparison.png"),
                os.path.join(model_dir, session_dir, "error_distribution.png"),
                os.path.join(model_dir, session_dir, "training_history.png")
            ]

            for graph in graph_files:
                if os.path.exists(graph):
                    platform_open(graph)

            # Afficher les métriques détaillées
            metrics_path = os.path.join(model_dir, session_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                print("\n=== MÉTRIQUES DÉTAILLÉES ===")
                print("\nDirection:")
                for k, v in metrics["steering"].items():
                    print(f"- {k}: {v:.6f}")
                print("\nAccélération:")
                for k, v in metrics["acceleration"].items():
                    print(f"- {k}: {v:.6f}")
        else:
            print("[ERREUR] Choix invalide.")
    except (ValueError, IndexError):
        print("[ERREUR] Choix invalide.")


def platform_open(file_path):
    """Ouvre un fichier avec l'application par défaut selon la plateforme."""
    try:
        if platform.system() == "Windows":
            os.startfile(file_path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", file_path], check=True)
        else:  # Linux
            subprocess.run(["xdg-open", file_path], check=True)
    except Exception as e:
        print(f"[ERREUR] Impossible d'ouvrir {file_path}: {e}")


def manage_models():
    """Interface pour gérer les modèles entraînés."""
    print("\n=== GESTION DES MODÈLES ===")

    # Lister les modèles disponibles
    model_checkpoint = "model_checkpoint.pth"
    if os.path.exists(model_checkpoint):
        size_mb = os.path.getsize(model_checkpoint) / (1024 * 1024)
        print(f"\nModèle principal actif: {model_checkpoint} ({size_mb:.2f} MB)")
    else:
        print("\n[ATTENTION] Aucun modèle principal trouvé!")

    # Options de gestion
    print("\nOptions:")
    print("1) Utiliser un modèle entraîné comme modèle principal")
    print("2) Supprimer d'anciens modèles")
    print("3) Exporter le modèle principal (pour déploiement)")
    print("4) Afficher les informations sur le modèle principal")
    print("5) Retour")

    choice = input("\nVotre choix: ").strip()

    if choice == "1":
        # Utiliser un modèle comme principal
        select_model_as_main()
    elif choice == "2":
        # Supprimer des modèles
        delete_models()
    elif choice == "3":
        # Exporter le modèle
        export_model()
    elif choice == "4":
        # Informations sur le modèle
        display_model_info()
    elif choice == "5":
        return
    else:
        print("[ERREUR] Choix invalide.")


def select_model_as_main():
    """Sélectionne un modèle entraîné comme modèle principal."""
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("[INFO] Aucun modèle entraîné disponible.")
        return

    # Trouver tous les fichiers modèles dans les sous-répertoires
    model_files = []
    for root, _, files in os.walk(models_dir):
        for file in files:
            if file.endswith(".pth"):
                model_files.append(os.path.join(root, file))

    if not model_files:
        print("[INFO] Aucun modèle (.pth) trouvé.")
        return

    print("\n=== MODÈLES DISPONIBLES ===")
    for i, path in enumerate(model_files, 1):
        rel_path = os.path.relpath(path, ".")
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"{i}. {rel_path} ({size_mb:.2f} MB)")

    choice = input("\nSélectionnez un modèle (n° ou r pour retour): ").strip()
    if choice.lower() == "r":
        return

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(model_files):
            selected_model = model_files[idx]
            # Copier comme modèle principal
            shutil.copy2(selected_model, "model_checkpoint.pth")
            print(f"[INFO] {os.path.basename(selected_model)} défini comme modèle principal.")
        else:
            print("[ERREUR] Choix invalide.")
    except (ValueError, IndexError):
        print("[ERREUR] Choix invalide.")


def delete_models():
    """Interface pour supprimer d'anciens modèles."""
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("[INFO] Aucun modèle à supprimer.")
        return

    # Lister les dossiers de sessions d'entraînement
    training_dirs = [d for d in os.listdir(models_dir)
                    if os.path.isdir(os.path.join(models_dir, d))]

    if not training_dirs:
        print("[INFO] Aucune session d'entraînement à supprimer.")
        return

    print("\n=== SESSIONS D'ENTRAÎNEMENT ===")
    for i, dir_name in enumerate(sorted(training_dirs), 1):
        dir_path = os.path.join(models_dir, dir_name)
        dir_size = get_dir_size(dir_path) / (1024 * 1024)  # MB
        print(f"{i}. {dir_name} ({dir_size:.2f} MB)")

    choice = input("\nSélectionnez une session à supprimer (n° ou r pour retour): ").strip()
    if choice.lower() == "r":
        return

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(training_dirs):
            to_delete = os.path.join(models_dir, training_dirs[idx])
            confirm = input(f"Confirmer la suppression de {training_dirs[idx]}? (o/n): ").lower()
            if confirm == "o":
                shutil.rmtree(to_delete)
                print(f"[INFO] Session {training_dirs[idx]} supprimée.")
        else:
            print("[ERREUR] Choix invalide.")
    except (ValueError, IndexError):
        print("[ERREUR] Choix invalide.")


def get_dir_size(path):
    """Calcule la taille totale d'un répertoire en octets."""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def export_model():
    """Exporte le modèle principal pour le déploiement."""
    model_path = "model_checkpoint.pth"
    if not os.path.exists(model_path):
        print("[ERREUR] Aucun modèle principal trouvé!")
        return

    export_dir = "export"
    os.makedirs(export_dir, exist_ok=True)

    # Créer un nom de fichier avec horodatage
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_name = f"robocar_model_{timestamp}.pth"
    export_path = os.path.join(export_dir, export_name)

    # Copier le modèle
    shutil.copy2(model_path, export_path)
    print(f"[INFO] Modèle exporté vers: {export_path}")


def display_model_info():
    """Affiche les informations sur le modèle principal."""
    model_path = "model_checkpoint.pth"
    if not os.path.exists(model_path):
        print("[ERREUR] Aucun modèle principal trouvé!")
        return

    try:
        _, activate_cmd = get_activate_command()
        cmd = f"{activate_cmd} && python -c \"import torch; m = torch.load('{model_path}'); print('=== INFORMATIONS MODÈLE ==='); [print(f'{k}: {v}') for k, v in m.items() if k != 'model_state_dict']\""
        subprocess.run(cmd, shell=True)
    except Exception as e:
        print(f"[ERREUR] Impossible d'afficher les informations du modèle: {e}")


def visualize_training_data():
    """Visualise les données d'entraînement."""
    print("\n=== VISUALISATION DES DONNÉES ===")

    # Vérifier si les données existent
    data_dir = "data/raw"
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print("[INFO] Aucune donnée disponible à visualiser.")
        return

    print("Cette fonction lancera une visualisation de vos données d'entraînement.")
    print("Options:")
    print("1) Distribution des commandes de direction")
    print("2) Distribution des commandes d'accélération")
    print("3) Visualisation des trajectoires")
    print("4) Statistiques des données")
    print("5) Retour")

    choice = input("\nVotre choix: ").strip()

    script_path = "src/utils/data_visualizer.py"
    if choice == "1":
        args = ["--type", "steering"]
        run_script(script_path, args, "la visualisation des commandes de direction")
    elif choice == "2":
        args = ["--type", "acceleration"]
        run_script(script_path, args, "la visualisation des commandes d'accélération")
    elif choice == "3":
        args = ["--type", "trajectory"]
        run_script(script_path, args, "la visualisation des trajectoires")
    elif choice == "4":
        args = ["--type", "stats"]
        run_script(script_path, args, "les statistiques des données")
    elif choice == "5":
        return
    else:
        print("[ERREUR] Choix invalide.")


# ===================== INFÉRENCE ET TESTS =====================

def inference_menu():
    """Affiche le menu d'inférence avec options avancées."""
    while True:
        print("\n=== MENU INFÉRENCE ===")
        print("1) Lancer l'inférence standard")
        print("2) Inférence avec options avancées")
        print("3) Test de performance")
        print("4) Mode démonstration (vitesse réduite)")
        print("5) Mode évaluation (métriques)")
        print("6) Retour au menu principal")

        choice = input("Votre choix: ").strip()

        if choice == "1":
            # Inférence standard
            run_script("src/inference/run_model.py", description="l'inférence")

        elif choice == "2":
            # Inférence avec options
            custom_inference()

        elif choice == "3":
            # Test de performance
            performance_test()

        elif choice == "4":
            # Mode démonstration
            run_script("src/inference/run_model.py", ["--mode", "demo"],
                      "le mode démonstration")

        elif choice == "5":
            # Mode évaluation
            run_script("src/inference/run_model.py", ["--mode", "eval"],
                      "le mode évaluation")

        elif choice == "6":
            # Retour
            break

        else:
            print("[ERREUR] Choix invalide. Réessayez.")


def custom_inference():
    """Interface pour configurer l'inférence avec des options avancées."""
    print("\n=== CONFIGURATION D'INFÉRENCE AVANCÉE ===")

    # Utilisation d'un modèle spécifique
    use_custom_model = input("Utiliser un modèle spécifique? (o/n): ").lower() == "o"
    model_path = None

    if use_custom_model:
        # Lister les modèles disponibles
        models_dir = "models"
        if os.path.exists(models_dir):
            all_models = []
            for root, _, files in os.walk(models_dir):
                for file in files:
                    if file.endswith(".pth"):
                        all_models.append(os.path.join(root, file))

            if all_models:
                print("\nModèles disponibles:")
                for i, path in enumerate(all_models, 1):
                    rel_path = os.path.relpath(path, ".")
                    print(f"{i}. {rel_path}")

                try:
                    choice = int(input("\nChoisissez un modèle (n°): ").strip())
                    if 1 <= choice <= len(all_models):
                        model_path = all_models[choice - 1]
                    else:
                        print("[ERREUR] Choix invalide. Utilisation du modèle par défaut.")
                except ValueError:
                    print("[ERREUR] Entrée invalide. Utilisation du modèle par défaut.")

    # Options de lissage
    smooth_factor = input("Facteur de lissage (0.0-1.0) [0.3]: ").strip() or "0.3"

    # Vitesse maximale
    max_speed = input("Vitesse maximale (0.0-1.0) [1.0]: ").strip() or "1.0"

    # Mode debug
    debug_mode = input("Activer le mode debug? (o/n) [n]: ").lower() == "o"

    # Construire les arguments
    args = []
    if model_path:
        args.extend(["--model", model_path])
    args.extend(["--smooth", smooth_factor])
    args.extend(["--max_speed", max_speed])
    if debug_mode:
        args.append("--debug")

    # Lancer l'inférence
    run_script("src/inference/run_model.py", args, "l'inférence personnalisée")


def performance_test():
    """Lance un test de performance pour l'inférence."""
    print("\n=== TEST DE PERFORMANCE ===")
    print("Ce test évaluera les performances d'inférence du modèle:")
    print("- Temps moyen d'inférence par image")
    print("- FPS (images par seconde)")
    print("- Utilisation mémoire")
    print("- Stabilité des prédictions")

    duration = input("Durée du test en secondes [30]: ").strip() or "30"

    run_script("src/inference/performance_test.py", ["--duration", duration],
              "le test de performance")


# ===================== MAINTENANCE =====================

def maintenance_menu():
    """Affiche le menu de maintenance du projet."""
    while True:
        print("\n=== MENU MAINTENANCE ===")
        print("1) Vérifier les ressources système")
        print("2) Nettoyer l'espace disque")
        print("3) Mettre à jour les dépendances")
        print("4) Créer une sauvegarde du projet")
        print("5) Options de déploiement")
        print("6) Retour au menu principal")

        choice = input("Votre choix: ").strip()

        if choice == "1":
            # Vérifier les ressources
            check_system_resources()

        elif choice == "2":
            # Nettoyer
            clean_disk_space()

        elif choice == "3":
            # Mettre à jour
            update_dependencies()

        elif choice == "4":
            # Sauvegarde
            create_backup()

        elif choice == "5":
            # Déploiement
            deployment_options()

        elif choice == "6":
            # Retour
            break

        else:
            print("[ERREUR] Choix invalide. Réessayez.")


def clean_disk_space():
    """Interface pour nettoyer l'espace disque."""
    print("\n=== NETTOYAGE D'ESPACE DISQUE ===")
    print("Options de nettoyage:")
    print("1) Nettoyer les fichiers temporaires")
    print("2) Archiver d'anciennes sessions de données")
    print("3) Supprimer les anciennes sessions d'entraînement")
    print("4) Supprimer tous les modèles (sauf principal)")
    print("5) Retour")

    choice = input("\nVotre choix: ").strip()

    if choice == "1":
        # Nettoyage fichiers temporaires
        clean_temp_files()
    elif choice == "2":
        # Archiver données
        archive_data()
    elif choice == "3":
        # Supprimer anciennes sessions
        delete_models()
    elif choice == "4":
        # Supprimer modèles
        confirm = input("Confirmer la suppression de tous les modèles (sauf principal)? (o/n): ").lower()
        if confirm == "o":
            remove_all_models_except_main()
    elif choice == "5":
        return
    else:
        print("[ERREUR] Choix invalide.")


def clean_temp_files():
    """Nettoie les fichiers temporaires."""
    temp_dirs = ["__pycache__", ".pytest_cache", ".ipynb_checkpoints"]
    deleted_count = 0

    for root, dirs, files in os.walk("."):
        for d in dirs:
            if d in temp_dirs:
                path = os.path.join(root, d)
                try:
                    shutil.rmtree(path)
                    deleted_count += 1
                    print(f"[INFO] Supprimé: {path}")
                except Exception as e:
                    print(f"[ERREUR] Impossible de supprimer {path}: {e}")

    print(f"[INFO] {deleted_count} répertoires temporaires supprimés.")


def archive_data():
    """Archive d'anciennes sessions de données."""
    data_dir = "data/raw"
    if not os.path.exists(data_dir):
        print("[INFO] Aucune donnée à archiver.")
        return

    # Lister les sessions de données
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        print("[INFO] Aucune donnée à archiver.")
        return

    print("\n=== SESSIONS DE DONNÉES DISPONIBLES ===")
    for i, file_path in enumerate(sorted(files), 1):
        file_name = os.path.basename(file_path)
        size_kb = os.path.getsize(file_path) / 1024
        mtime = os.path.getmtime(file_path)
        date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        print(f"{i}. {file_name} - {size_kb:.1f} KB - {date_str}")

    # Créer un répertoire d'archive
    archive_dir = "data/archive"
    os.makedirs(archive_dir, exist_ok=True)

    # Archiver les fichiers sélectionnés
    selection = input("\nSélectionnez les fichiers à archiver (ex: 1,3,5 ou 'all' pour tous): ").strip()

    if selection.lower() == "all":
        indices = range(len(files))
    else:
        try:
            indices = [int(idx.strip()) - 1 for idx in selection.split(",")]
        except ValueError:
            print("[ERREUR] Format de sélection invalide.")
            return

    # Vérifier les indices
    valid_indices = [idx for idx in indices if 0 <= idx < len(files)]
    if not valid_indices:
        print("[ERREUR] Aucun fichier valide sélectionné.")
        return

    # Créer une archive zip
    archive_name = f"data_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    archive_path = os.path.join(archive_dir, archive_name)

    try:
        import zipfile
        with zipfile.ZipFile(archive_path, 'w') as zipf:
            for idx in valid_indices:
                file_path = files[idx]
                zipf.write(file_path, os.path.basename(file_path))
                print(f"[INFO] Archivé: {os.path.basename(file_path)}")

        # Demander si l'utilisateur veut supprimer les originaux
        delete_orig = input("\nSupprimer les fichiers originaux? (o/n): ").lower() == "o"
        if delete_orig:
            for idx in valid_indices:
                os.remove(files[idx])
                print(f"[INFO] Supprimé: {os.path.basename(files[idx])}")

        print(f"[INFO] Archive créée: {archive_path}")
    except Exception as e:
        print(f"[ERREUR] Échec de l'archivage: {e}")


def remove_all_models_except_main():
    """Supprime tous les modèles sauf le modèle principal."""
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("[INFO] Aucun modèle à supprimer.")
        return

    # Compter avant
    model_count = 0
    for root, _, files in os.walk(models_dir):
        for file in files:
            if file.endswith(".pth"):
                model_count += 1

    # Supprimer les dossiers de modèles
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"[INFO] Supprimé: {item_path}")

    print(f"[INFO] {model_count} modèles supprimés.")


def update_dependencies():
    """Met à jour les dépendances du projet."""
    print("\n=== MISE À JOUR DES DÉPENDANCES ===")

    _, activate_cmd = get_activate_command()
    cmd = f"{activate_cmd} && pip install --upgrade -r requirements.txt"

    try:
        subprocess.run(cmd, shell=True, check=True)
        print("[INFO] Dépendances mises à jour avec succès.")
    except subprocess.CalledProcessError:
        print("[ERREUR] Échec de la mise à jour des dépendances.")


def create_backup():
    """Crée une sauvegarde du projet."""
    print("\n=== CRÉATION D'UNE SAUVEGARDE ===")

    # Répertoire de sauvegarde
    backup_dir = "backups"
    os.makedirs(backup_dir, exist_ok=True)

    # Nom du fichier de sauvegarde
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"robocar_backup_{timestamp}.zip"
    backup_path = os.path.join(backup_dir, backup_name)

    # Éléments à inclure/exclure
    include_dirs = ["src", "config", "data", "models"]
    include_files = ["manager.py", "requirements.txt"]
    exclude_patterns = ["__pycache__", ".venv", "*.pyc"]

    print("Éléments à sauvegarder:")
    for d in include_dirs:
        print(f"- {d}/")
    for f in include_files:
        print(f"- {f}")

    confirm = input("\nCréer la sauvegarde? (o/n): ").lower()
    if confirm != "o":
        return

    try:
        import zipfile

        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Ajouter les fichiers individuels
            for file in include_files:
                if os.path.exists(file):
                    zipf.write(file)
                    print(f"[INFO] Ajouté: {file}")

            # Ajouter les répertoires
            for directory in include_dirs:
                if os.path.exists(directory):
                    for root, dirs, files in os.walk(directory):
                        # Filtrer les répertoires exclus
                        dirs[:] = [d for d in dirs if not any(
                            pattern.rstrip('/') == d or
                            (pattern.endswith('*') and d.startswith(pattern[:-1]))
                            for pattern in exclude_patterns
                        )]

                        for file in files:
                            # Filtrer les fichiers exclus
                            if not any(
                                pattern == file or
                                (pattern.startswith('*') and file.endswith(pattern[1:])) or
                                (pattern.endswith('*') and file.startswith(pattern[:-1]))
                                for pattern in exclude_patterns
                            ):
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, '.')
                                zipf.write(file_path, arcname)
                                print(f"[INFO] Ajouté: {arcname}")

        backup_size = os.path.getsize(backup_path) / (1024 * 1024)  # MB
        print(f"[INFO] Sauvegarde créée: {backup_path} ({backup_size:.2f} MB)")
    except Exception as e:
        print(f"[ERREUR] Échec de la sauvegarde: {e}")


def deployment_options():
    """Options de déploiement du projet."""
    print("\n=== OPTIONS DE DÉPLOIEMENT ===")
    print("1) Préparer modèle pour Jetson Nano (quantization)")
    print("2) Exporter modèle optimisé")
    print("3) Générer package de déploiement")
    print("4) Retour")

    choice = input("\nVotre choix: ").strip()

    if choice == "1":
        run_script("src/deploy/quantize_model.py", description="la préparation du modèle pour Jetson Nano")
    elif choice == "2":
        run_script("src/deploy/optimize_model.py", description="l'export du modèle optimisé")
    elif choice == "3":
        run_script("src/deploy/generate_package.py", description="la génération du package de déploiement")
    elif choice == "4":
        return
    else:
        print("[ERREUR] Choix invalide.")


# ===================== MENU PRINCIPAL =====================

def display_welcome():
    """Affiche un message de bienvenue avec logo ASCII."""
    print("\n" + "="*60)
    print("""
    ____       _           ____
   / __ \____  (_)_________/ __ \_________
  / /_/ / __ \/ / ___/ ___/ /_/ / ___/ __ \\
 / _, _/ /_/ / / /__/ /  / _, _/ /__/ /_/ /
/_/ |_|\____/_/\___/_/  /_/ |_|\___/\____/
    
     Système de gestion du projet RoboCar
    """)
    print("="*60 + "\n")


def main_menu():
    """Affiche le menu principal amélioré."""
    while True:
        display_welcome()
        print("MENU PRINCIPAL:")
        print("1) Collecte de données")
        print("2) Entraînement des modèles")
        print("3) Inférence et tests")
        print("4) Maintenance du projet")
        print("5) Afficher les ressources système")
        print("6) Quitter")

        choice = input("\nVotre choix: ").strip()

        if choice == "1":
            collect_data_menu()
        elif choice == "2":
            training_menu()
        elif choice == "3":
            inference_menu()
        elif choice == "4":
            maintenance_menu()
        elif choice == "5":
            check_system_resources()
            input("\nAppuyez sur Entrée pour continuer...")
        elif choice == "6":
            print("[INFO] Fin du programme. Au revoir!")
            break
        else:
            print("[ERREUR] Choix invalide. Réessayez.")


def main():
    """
    Point d'entrée principal du programme.

    1) Vérifie/installe l'environnement virtuel et les packages
    2) Lance le menu interactif amélioré
    """
    check_and_setup_venv()
    main_menu()


if __name__ == "__main__":
    main()
