# RoboCar Data Collector

Ce module collecte des données d'entraînement à partir du simulateur de course RoboCar pour les modèles d'apprentissage supervisé. Il gère la connexion à la simulation Unity, enregistre les données des capteurs et les entrées utilisateur, et sauvegarde le tout dans un format adapté à l'entraînement des modèles d'IA.

## Vue d'ensemble

Le collecteur de données :
1. Se connecte au simulateur de course Unity
2. Capture les entrées utilisateur (clavier ou joystick)
3. Enregistre les données de simulation (raycasts, vitesse, angles de direction, position)
4. Sauvegarde les données synchronisées dans des fichiers CSV pour l'entraînement

## Fonctionnalités

- **Intégration avec le simulateur** : Connexion directe au simulateur de course Unity via ML-Agents
- **Méthodes d'entrée** : Support pour le contrôle au clavier et au joystick
- **Outil de calibration** : Utilitaire graphique intégré pour la calibration du joystick
- **Capteurs configurables** : Nombre de raycasts et champ de vision ajustables
- **Retour en temps réel** : Affichage terminal de toutes les lectures de capteurs et entrées
- **Persistance des données** : Génération automatique de CSV avec des noms de fichiers horodatés
- **Gestion robuste des erreurs** : Récupération élégante des déconnexions et des erreurs

## Prérequis

- Python 3.7+
- Simulateur de course Unity RoboCar
- Package ML-Agents
- pygame
- numpy
- pynput

## Installation

1. Clonez le dépôt :
   ```bash
   git clone <url-du-dépôt>
   cd RoboCar
   ```

2. Installez les dépendances :
   ```bash
   pip install mlagents pygame numpy pynput
   ```

3. Assurez-vous que l'exécutable du simulateur de course Unity a les permissions d'exécution :
   ```bash
   chmod +x RacingSimulatorLinux/RacingSimulator.x86_64
   ```

## Utilisation

### Démarrer la collecte de données

Usage basique :
```bash
python src/collector/collect_data.py
```

Avec calibration du joystick au démarrage :
```bash
python src/collector/collect_data.py --calibrate
```

### Contrôles

- **Clavier** :
  - **Touches fléchées** ou **WASD/ZQSD** : Contrôler la voiture (direction et accélération)
  - **C** : Lancer la calibration du joystick pendant l'exécution
  - **Ctrl+C** : Arrêter la collecte de données

- **Joystick** :
  - **Mode Dual Stick (par défaut)** :
    - **Stick gauche** : Contrôler la direction (gauche/droite)
    - **Stick droit** : Contrôler l'accélération (haut/bas)
    - **Bouton Y** : Basculer entre les modes de contrôle
  - **Mode Single Stick (traditionnel)** :
    - **Stick gauche** : Contrôler la direction et l'accélération
    - **Bouton Y** : Basculer entre les modes de contrôle
  - Le joystick est automatiquement détecté s'il est présent

### Configuration

#### Paramètres de simulation

Configurez les graphismes et la physique dans `config/raycast_config.json` :
```json
{
  "graphic_settings": {
    "width": 1280,
    "height": 720,
    "quality_level": 3
  },
  "time_scale": 1.0
}
```

#### Configuration de l'agent

Configurez les capteurs de l'agent dans `config/agent_config.json` :
```json
{
  "agents": [
    {
      "fov": 180,
      "nbRay": 10
    }
  ]
}
```

Paramètres :
- `fov` : Champ de vision en degrés (1-180)
- `nbRay` : Nombre de raycasts (1-50)

> **Important pour l'entraînement du réseau neuronal** : Le nombre de raycasts (`nbRay`) défini dans ce fichier de configuration détermine la dimension d'entrée de votre réseau neuronal. L'architecture de votre modèle doit être compatible avec cette valeur. Lors de l'entraînement d'un réseau neuronal, assurez-vous que la couche d'entrée peut accepter exactement ce nombre de valeurs de raycast.

## Sortie de données

### Emplacement des fichiers

Les données sont enregistrées au format CSV dans le répertoire `data/raw/` avec des noms de fichiers basés sur l'horodatage :
```
data/raw/session_1234567890.csv
```

### Format des données

Chaque fichier CSV contient les colonnes suivantes :

| Colonne | Description |
|--------|-------------|
| timestamp | Horodatage Unix au moment de l'enregistrement |
| steering_input | Entrée de direction utilisateur (-1.0 à 1.0) |
| acceleration_input | Entrée d'accélération utilisateur (-1.0 à 1.0) |
| raycasts | Tableau des distances de la voiture aux obstacles |
| speed | Vitesse actuelle de la voiture |
| steering | Angle de direction actuel de la voiture |
| position_x | Coordonnée X de la voiture |
| position_y | Coordonnée Y de la voiture |
| position_z | Coordonnée Z de la voiture |

### Extraction des observations

Le collecteur de données extrait les observations de la simulation Unity en utilisant ce modèle :
- Raycasts : `obs_array[:num_rays]`
- Vitesse : `obs_array[-5]`
- Direction : `obs_array[-4]`
- Position : `[obs_array[-3], obs_array[-2], obs_array[-1]]`

## Calibration du joystick

### Calibration automatique

L'outil de calibration vous guidera pour déplacer le joystick à ses extrêmes afin de déterminer la plage maximale de mouvement. Les données de calibration sont enregistrées et appliquées pour normaliser les entrées.

### Configuration manuelle

Les données de calibration sont stockées dans `src/collector/joystick_calibration.json` et peuvent être modifiées manuellement si nécessaire :
```json
{
  "steering": {"min": -1.0, "max": 1.0},
  "acceleration": {"min": -1.0, "max": 1.0}
}
```

## Dépannage

### Problèmes courants

1. **Exécutable Unity introuvable** :
   - Assurez-vous que le simulateur est au bon emplacement : `RacingSimulatorLinux/RacingSimulator.x86_64`

2. **Joystick non détecté** :
   - Connectez le joystick avant de démarrer le programme
   - Exécutez `pygame.joystick.Joystick(0).get_name()` pour vérifier la détection

3. **Port déjà en utilisation** :
   - Si vous voyez des erreurs "port already in use", assurez-vous qu'aucune autre instance n'est en cours d'exécution
   - Le port par défaut est 5004, peut être modifié dans le code si nécessaire

4. **Observations manquantes** :
   - Vérifiez que `nbRay` dans la configuration correspond aux paramètres du simulateur
   - Consultez les logs Unity pour toute erreur

## Structure du projet

- `collect_data.py` : Script principal pour la collecte de données
- `utils_collector.py` : Utilitaires de traitement des entrées
- `joystick_calibrator.py` : Fonctions de calibration du joystick
- `ui_components.py` : Interface utilisateur pour la calibration du joystick
- `config/` : Fichiers de configuration pour le simulateur et l'agent

## Intégration avec les autres modules

- **Module de modèle** : Les données collectées servent à entraîner les réseaux de neurones dans `src/model`
- **Module d'inférence** : Les modèles entraînés sur ces données sont utilisés dans `src/inference` pour la conduite autonome
