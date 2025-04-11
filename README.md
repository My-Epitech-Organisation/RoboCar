# RoboCar - Projet d'IA pour Voiture Autonome

RoboCar est un projet de conduite autonome basé sur l'apprentissage supervisé. Le système collecte des données depuis un simulateur Unity, entraîne des modèles de réseaux neuronaux, et utilise ces modèles pour piloter une voiture virtuelle de manière autonome.

## Vue d'ensemble

Le projet est divisé en trois modules principaux :

1. **Collecte de données** : Enregistre les données de conduite depuis le simulateur
2. **Entraînement de modèles** : Crée et entraîne des réseaux neuronaux
3. **Inférence** : Utilise les modèles entraînés pour la conduite autonome

## Prérequis

- Système d'exploitation Linux (testé sur Fedora)
- Simulateur Unity RoboCar
- Python 3.10.12

## Installation

### 1. Installation de Python 3.10.12

```bash
wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz
sudo tar xvf Python-3.10.12.tgz
cd Python-3.10.12
./configure --enable-optimizations
make -j$(nproc)
sudo make altinstall
```

### 2. Configuration du projet

1. Clonez le dépôt :
   ```bash
   git clone <URL_DU_REPO>
   cd RoboCar
   ```

2. Exécutez le script de gestion :
   ```bash
   python3.10 manager.py
   ```

Le script configurera automatiquement l'environnement virtuel et installera toutes les dépendances nécessaires.

## Utilisation avec le script de gestion (manager.py)

Le projet propose un script de gestion complet qui simplifie toutes les opérations. Pour une utilisation simple, suivez ce guide en utilisant uniquement les options "1" dans chaque menu.

### Démarrer le script de gestion

```bash
python3.10 manager.py
```

### 1. Collecte de données

1. Dans le menu principal, sélectionnez `1` (Collecte de données)
2. Dans le sous-menu, sélectionnez `1` (Collecte standard)
3. Appuyez sur Enter pour un nom de session auto-généré ou entrez un nom personnalisé
4. Utilisez les contrôles suivants dans le simulateur :
   - Touches fléchées ou WASD/ZQSD : Contrôler la voiture
   - Ctrl+C : Arrêter la collecte

### 2. Entraînement du modèle

1. Dans le menu principal, sélectionnez `2` (Entraînement des modèles)
2. Dans le sous-menu, sélectionnez `1` (Entraînement rapide)
3. Attendez que l'entraînement se termine

L'entraînement utilisera automatiquement les données collectées précédemment avec les paramètres optimaux par défaut.

### 3. Test du modèle (Inférence)

1. Dans le menu principal, sélectionnez `3` (Inférence et tests)
2. Dans le sous-menu, sélectionnez `1` (Lancer l'inférence standard)
3. Observez la voiture conduire de manière autonome
4. Utilisez Ctrl+C pour arrêter l'inférence

### 4. Maintenance (si nécessaire)

1. Dans le menu principal, sélectionnez `4` (Maintenance du projet)
2. Dans le sous-menu, sélectionnez `1` (Vérifier les ressources système)

## Comprendre les métriques d'entraînement

Pendant l'entraînement, vous verrez des lignes comme :

### Contributors

- [@Arkteus](https://github.com/Arkteus)
- [@nduboi](https://github.com/nduboi)
- [@albanrss](https://github.com/albanrss)