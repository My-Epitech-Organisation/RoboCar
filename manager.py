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

def check_and_setup_venv():
    """
    Vérifie si le dossier .venv existe.
    Si non, le crée et installe les dépendances via requirements.txt.
    """
    venv_dir = ".venv"
    
    # 1) Vérifier si .venv existe déjà
    if not os.path.isdir(venv_dir):
        print("[INFO] Environnement virtuel (.venv) introuvable, création en cours...")
        # Création de l'environnement virtuel
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
    else:
        print("[INFO] Environnement virtuel déjà présent.")

    # 2) Installation / mise à jour des dépendances
    print("[INFO] Installation / Mise à jour des dépendances...")
    # Sur Linux/Mac, l'exécutable Python du venv est dans .venv/bin/python
    venv_python = os.path.join(venv_dir, "bin", "python")
    
    # Sous Windows, remplacer la ligne ci-dessus par:
    # venv_python = os.path.join(venv_dir, "Scripts", "python.exe")

    subprocess.run([venv_python, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    print("[INFO] Les dépendances sont installées / à jour.")

def launch_collect_data():
    """
    Lance le script de collecte de données depuis le venv.
    """
    print("[INFO] Lancement de la collecte de données...")
    venv_python = os.path.join(".venv", "bin", "python")
    # (Sous Windows, adapter le chemin.)
    subprocess.run([venv_python, "src/collector/collect_data.py"], check=True)

def launch_training():
    """
    Lance le script d'entraînement depuis le venv.
    """
    print("[INFO] Lancement de l'entraînement...")
    venv_python = os.path.join(".venv", "bin", "python")
    subprocess.run([venv_python, "src/model/train.py"], check=True)

def launch_inference():
    """
    Lance le script d'inférence depuis le venv.
    """
    print("[INFO] Lancement de l'inférence...")
    venv_python = os.path.join(".venv", "bin", "python")
    subprocess.run([venv_python, "src/inference/run_model.py"], check=True)

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
