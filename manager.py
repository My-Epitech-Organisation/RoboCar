#!/usr/bin/env python3
"""
manager.py - Script de gestion du projet RoboCar

Ce script gère:
1) La création et mise à jour d'un environnement virtuel (.venv)
2) L'installation des dépendances (requirements.txt)
3) L'interface pour lancer les différents processus (collecte, entraînement, inférence)
"""

import os
import subprocess
import sys
import platform


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


def run_script(script_path, description):
    """
    Lance un script Python dans l'environnement virtuel.

    Args:
        script_path (str): Chemin vers le script à exécuter
        description (str): Description pour le message d'information
    """
    print(f"[INFO] Lancement de {description}...")
    _, activate_cmd = get_activate_command()
    cmd = f"{activate_cmd} && python {script_path}"
    subprocess.run(cmd, shell=True, check=True)


def launch_collect_data():
    """Lance le script de collecte de données."""
    run_script("src/collector/collect_data.py", "la collecte de données")


def launch_training():
    """Lance le script d'entraînement du modèle."""
    run_script("src/model/train.py", "l'entraînement")


def launch_inference():
    """Lance le script d'inférence du modèle."""
    run_script("src/inference/run_model.py", "l'inférence")


def main_menu():
    """
    Affiche un menu interactif pour l'utilisateur.
    """
    while True:
        print("\n=== MENU ROBOCAR ===")
        print("1) Collecte de données")
        print("2) Entraîner le modèle")
        print("3) Lancer l'inférence")
        print("4) Quitter")

        choice = input("Votre choix: ").strip()

        if choice == "1":
            launch_collect_data()
        elif choice == "2":
            launch_training()
        elif choice == "3":
            launch_inference()
        elif choice == "4":
            print("[INFO] Fin du script.")
            break
        else:
            print("[ERREUR] Choix invalide. Réessayez.")


def main():
    """
    Point d'entrée principal du programme.

    1) Vérifie/installe l'environnement virtuel et les packages
    2) Lance le menu interactif
    """
    check_and_setup_venv()
    main_menu()


if __name__ == "__main__":
    main()
