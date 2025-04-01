# RoboCar Autonomous Driving Inference

Ce module charge les modèles de réseaux neuronaux entraînés et les utilise pour la conduite autonome dans le simulateur de course RoboCar. Il se connecte à la simulation Unity, traite les données des capteurs et utilise le modèle pour générer des commandes de conduite.

## Vue d'ensemble

Le module d'inférence :
1. Charge le modèle de réseau neuronal entraîné
2. Se connecte au simulateur de course Unity
3. Traite les lectures des capteurs (raycasts, vitesse, position)
4. Génère les commandes de direction et d'accélération
5. Envoie les commandes pour conduire la voiture de manière autonome

## Fonctionnalités

- **Chargement de modèle** : Charge les modèles PyTorch entraînés avec métadonnées
- **Inférence en temps réel** : Traitement rapide des données des capteurs
- **Lissage de direction** : Réduit les oscillations avec lissage temporel
- **Surveillance des performances** : Suit les FPS et les temps d'inférence
- **Compatibilité de configuration** : S'assure que le modèle correspond à la configuration des raycasts

## Prérequis

- Python 3.7+
- Simulateur de course Unity RoboCar
- PyTorch
- Package ML-Agents
- NumPy

## Utilisation

### Usage de base

Démarrez la conduite autonome avec le modèle par défaut :

```bash
python src/inference/run_model.py
```

### Contrôles

- **Ctrl+C** : Arrêter le mode de conduite autonome

### Sélection du modèle

Le système charge automatiquement le fichier modèle `model_checkpoint.pth` à partir de la racine du projet. C'est le chemin de sortie par défaut lors de l'entraînement d'un modèle avec le module d'entraînement.

## Détails d'implémentation

### Traitement des observations

Le module d'inférence traite les observations de la simulation Unity de la même manière qu'elles ont été traitées pendant l'entraînement :

1. Les raycasts sont extraits et normalisés dans la plage [0,1]
2. La vitesse du véhicule est normalisée
3. Les caractéristiques sont combinées en un tenseur d'entrée pour le modèle

### Lissage de direction

Pour éviter un comportement de direction erratique, le module implémente un lissage :

1. Maintient un historique des prédictions de direction récentes
2. Applique une moyenne mobile pour lisser les transitions
3. Limite le changement maximal de direction par image

### Optimisation des performances

Pour un contrôle en temps réel, le module d'inférence :

1. Minimise les frais généraux de prétraitement
2. Utilise l'inférence CPU pour éviter les délais de transfert GPU
3. Surveille les taux d'images pour assurer un contrôle réactif

## Classes et méthodes principales

### Dans `run_model.py`
- `load_model()` : Charge le modèle entraîné avec ses métadonnées
- `process_observations()` : Traite et normalise les observations du simulateur
- `run_inference_loop()` : Boucle principale qui exécute l'inférence en temps réel

### Dans `utils_inference.py`
- `smooth_steering()` : Applique un algorithme de lissage aux prédictions de direction
- `normalize_observations()` : Normalise les valeurs d'observation pour correspondre à l'entraînement
- `PerformanceMonitor` : Classe pour surveiller et enregistrer les performances d'inférence

## Dépannage

### Problèmes courants

1. **Erreurs de chargement de modèle** :
   - Assurez-vous que le fichier modèle existe à la racine du projet
   - Vérifiez que le modèle a été entraîné avec des données compatibles

2. **Problèmes de connexion au simulateur** :
   - Vérifiez que le simulateur est en cours d'exécution et sur le port 5004
   - Vérifiez les permissions sur l'exécutable du simulateur

3. **Comportement de conduite erratique** :
   - Essayez d'augmenter les paramètres de lissage de direction
   - Collectez plus de données d'entraînement dans des scénarios problématiques
   - Réentraînez le modèle avec des données plus diverses

## Architecture

### Compatibilité du modèle

La taille d'entrée du modèle doit correspondre au nombre de raycasts dans votre configuration :

```
model_input_size = num_rays + 1  # Rayons + vitesse
```

Si vous changez le nombre de raycasts dans votre configuration, vous devez réentraîner votre modèle.

## Intégration avec les autres modules

- **Module collecteur** : Utilise la même structure de simulateur et de données que `src/collector`
- **Module de modèle** : Charge les modèles entraînés créés par `src/model`
