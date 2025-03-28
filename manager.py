#!/usr/bin/env python3
"""
manager.py
Script qui gère:
1) La création/MAJ d'un virtualenv (.venv)
2) L'installation des packages (requirements.txt)
3) Un menu pour lancer les différents process (collecte, entraînement, inférence)
"""

import os
import subprocess
import sys
import platform

def check_python_version():
    """
    Vérifie si Python 3.10.12 est disponible sur le système.
    Retourne le chemin vers l'exécutable Python 3.10.12 s'il est disponible.
    """
    try:
        # Vérifie si python3.10 est disponible
        result = subprocess.run(
            ["python3.10", "--version"], 
            capture_output=True, 
            text=True, 
            check=False
        )
        if result.returncode == 0 and "3.10" in result.stdout:
            # Vérifie si c'est exactement 3.10.12
            if "3.10.12" in result.stdout:
                print("[INFO] Python 3.10.12 trouvé.")
                return "python3.10"
            else:
                print(f"[ATTENTION] Python 3.10 trouvé mais pas exactement 3.10.12 ({result.stdout.strip()})")
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
    Vérifie si le dossier .venv existe.
    Si non, le crée et installe les dépendances via requirements.txt.
    """
    venv_dir = ".venv"

    # Obtenir l'exécutable Python 3.10.12
    python_exec = check_python_version()

    # Déterminer le séparateur et les chemins selon l'OS
    is_windows = platform.system() == "Windows"
    activate_path = os.path.join(venv_dir, "Scripts", "activate") if is_windows else os.path.join(venv_dir, "bin", "activate")
    activate_cmd = f"call {activate_path}" if is_windows else f"source {activate_path}"

    # 1) Vérifier si .venv existe déjà
    if not os.path.isdir(venv_dir):
        print("[INFO] Environnement virtuel (.venv) introuvable, création en cours...")
        # Création de l'environnement virtuel avec Python 3.10.12
        subprocess.run([python_exec, "-m", "venv", venv_dir], check=True)
    else:
        print("[INFO] Environnement virtuel déjà présent.")
        print("[INFO] Vérification de la version Python dans l'environnement...")
        # Vérifier la version Python dans le venv existant
        py_path = os.path.join(venv_dir, "bin", "python") if not is_windows else os.path.join(venv_dir, "Scripts", "python.exe")
        if os.path.exists(py_path):
            result = subprocess.run([py_path, "--version"], capture_output=True, text=True, check=False)
            if "3.10" not in result.stdout:
                print(f"[ATTENTION] Version Python incorrecte dans le venv: {result.stdout.strip()}")
                choice = input("Recréer l'environnement avec Python 3.10.12 ? (o/n): ").lower()
                if choice == 'o':
                    print("[INFO] Suppression de l'ancien environnement...")
                    import shutil
                    shutil.rmtree(venv_dir)
                    print("[INFO] Création du nouvel environnement...")
                    subprocess.run([python_exec, "-m", "venv", venv_dir], check=True)

    # 2) Installation / mise à jour des dépendances
    print("[INFO] Installation / Mise à jour des dépendances...")
    # Utilisation de l'activation du venv
    cmd = f"{activate_cmd} && pip install -r requirements.txt"
    subprocess.run(cmd, shell=True, check=True)
    print("[INFO] Les dépendances sont installées / à jour.")

def launch_collect_data():
    """
    Lance le script de collecte de données depuis le venv.
    """
    print("[INFO] Lancement de la collecte de données...")
    is_windows = platform.system() == "Windows"
    activate_path = os.path.join(".venv", "Scripts", "activate") if is_windows else os.path.join(".venv", "bin", "activate")
    activate_cmd = f"call {activate_path}" if is_windows else f"source {activate_path}"

    cmd = f"{activate_cmd} && python src/collector/collect_data.py"
    subprocess.run(cmd, shell=True, check=True)

def launch_training():
    """
    Lance le script d'entraînement depuis le venv.
    """
    print("[INFO] Lancement de l'entraînement...")
    is_windows = platform.system() == "Windows"
    activate_path = os.path.join(".venv", "Scripts", "activate") if is_windows else os.path.join(".venv", "bin", "activate")
    activate_cmd = f"call {activate_path}" if is_windows else f"source {activate_path}"

    cmd = f"{activate_cmd} && python src/model/train.py"
    subprocess.run(cmd, shell=True, check=True)

def launch_inference():
    """
    Lance le script d'inférence depuis le venv.
    """
    print("[INFO] Lancement de l'inférence...")
    is_windows = platform.system() == "Windows"
    activate_path = os.path.join(".venv", "Scripts", "activate") if is_windows else os.path.join(".venv", "bin", "activate")
    activate_cmd = f"call {activate_path}" if is_windows else f"source {activate_path}"

    cmd = f"{activate_cmd} && python src/inference/run_model.py"
    subprocess.run(cmd, shell=True, check=True)

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
    Point d'entrée principal:
    1) Vérifie / installe le venv et les packages
    2) Lance le menu interactif
    """
    check_and_setup_venv()
    main_menu()

if __name__ == "__main__":
    main()
