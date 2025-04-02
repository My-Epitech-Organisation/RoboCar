# RoboCar Neural Network Training

Ce module entraîne des réseaux de neurones avec les données collectées à partir du simulateur de course RoboCar pour la conduite autonome. Il fournit des outils pour le prétraitement des données, la création de modèles, l'entraînement, l'évaluation et le déploiement.

## Vue d'ensemble

Le pipeline d'entraînement du réseau neuronal :
1. Charge et prétraite les données collectées depuis la simulation
2. Construit et configure des architectures de réseaux neuronaux
3. Entraîne les modèles sur les données collectées
4. Évalue les performances du modèle à l'aide de plusieurs métriques
5. Déploie les modèles entraînés pour la conduite autonome

## Fonctionnalités

- **Prétraitement des données** : Outils pour normaliser, augmenter et préparer les données collectées
- **Architectures de modèles** : Plusieurs architectures de réseaux neuronaux préconfigurées optimisées pour la tâche
- **Pipeline d'entraînement** : Processus d'entraînement complet avec checkpointing et arrêt anticipé
- **Outils d'évaluation** : Métriques complètes pour évaluer les performances du modèle
- **Visualisation** : Outils pour visualiser la progression de l'entraînement et les prédictions du modèle
- **Déploiement** : Méthodes pour exporter et utiliser les modèles entraînés dans le simulateur

## Prérequis

- Python 3.7+
- PyTorch
- NumPy, Pandas
- scikit-learn
- Matplotlib (optionnel, pour la visualisation)

## Installation

Installez les dépendances requises :
```bash
pip install torch numpy pandas scikit-learn matplotlib
```

## Utilisation

### Préparation des données

```python
from data_preprocessor import load_session, preprocess_data, split_data

# Charger les données collectées
data = load_session('data/raw/session_1234567890.csv')

# Prétraiter les données
X, y = preprocess_data(data, normalize=True, augment=True)

# Diviser en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2)
```

### Entraînement du modèle

```python
from model_builder import create_model
from trainer import train_model

# Créer un modèle
model = create_model("cnn", input_size=11, num_rays=10)  # Pour 10 raycasts + vitesse

# Entraîner le modèle
trained_model, history = train_model(
    model,
    X_train, y_train,
    X_val, y_val,
    epochs=100,
    batch_size=32,
    project_root="chemin/vers/projet"
)
```

### Lancement de l'entraînement via le script principal

```bash
# Entraînement de base avec paramètres par défaut
python src/model/train.py

# Avec des paramètres personnalisés
python src/model/train.py --model_type cnn --epochs 200 --batch_size 64
```

### Options du script d'entraînement

- `--data_dir` : Répertoire contenant les fichiers CSV (défaut: "data")
- `--model_type` : Type de modèle à créer (défaut: "simple", options: "simple", "cnn", "lstm", "multi")
- `--epochs` : Nombre d'époques d'entraînement (défaut: 100)
- `--batch_size` : Taille du batch (défaut: 32)
- `--learning_rate` : Taux d'apprentissage (défaut: 0.001)
- `--no_augment` : Désactiver l'augmentation des données
- `--no_viz` : Désactiver la visualisation (utile si matplotlib n'est pas disponible)

## Préparation des données

### Chargement des données

Le module collecteur enregistre les données au format CSV avec les colonnes suivantes :
- `timestamp` : Horodatage Unix
- `steering_input` : Entrée de direction utilisateur (-1.0 à 1.0)
- `acceleration_input` : Entrée d'accélération utilisateur (-1.0 à 1.0)
- `raycasts` : Tableau des distances de la voiture aux obstacles
- `speed` : Vitesse actuelle de la voiture
- `steering` : Angle de direction actuel
- `position_x`, `position_y`, `position_z` : Coordonnées de position de la voiture

> **Important** : Le nombre de raycasts dans vos données est déterminé par le paramètre `nbRay` dans `config/agent_config.json`. La couche d'entrée de votre réseau neuronal doit être compatible avec cette valeur. Si vous modifiez le nombre de rayons dans la configuration, vous devrez ajuster l'architecture de votre modèle en conséquence.

### Étapes de prétraitement

1. **Parsing** : Conversion des chaînes de raycasts en tableaux numériques
2. **Normalisation** : Mise à l'échelle des valeurs de raycast dans la plage [0, 1]
3. **Ingénierie des caractéristiques** : Calcul de caractéristiques supplémentaires
4. **Augmentation** : Génération d'échantillons supplémentaires par miroir et ajout de bruit
5. **Création de séquences** : Pour les modèles récurrents, création d'échantillons de séquences temporelles

## Architectures de modèles

### Modèle Simple (SimpleModel)

Réseau neuronal entièrement connecté pour le traitement de base des raycasts :

```python
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        # ...
```

### Modèle CNN (CNNModel)

Réseau neuronal convolutif pour le traitement spatial des données de raycast :

```python
class CNNModel(nn.Module):
    def __init__(self, input_size, num_rays):
        # ...
```

### Modèle LSTM (LSTMModel)

Pour la prise de décision séquentielle en tenant compte des observations passées :

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        # ...
```

### Modèle Multi-entrées (MultiInputModel)

Pour combiner les données de raycast avec d'autres informations comme la vitesse :

```python
class MultiInputModel(nn.Module):
    def __init__(self, raycast_size, other_feature_size):
        # ...
```

## Bonnes pratiques d'entraînement

### Recommandations pour la collecte de données

1. **Quantité** : Collectez au moins 30 minutes de données de conduite
2. **Diversité** : Incluez :
   - Plusieurs tours sur différentes pistes
   - Vitesses variables
   - Récupération à partir de cas limites
   - Conduite fluide dans les virages
3. **Qualité** : Assurez-vous d'avoir des modèles de conduite cohérents et fluides

### Paramètres d'entraînement

- **Taille de batch** : 32-64 (plus petit pour des mises à jour de gradient plus précises)
- **Taux d'apprentissage** : Commencez avec 0.001, réduisez si l'entraînement est instable
- **Époques** : Utilisez un arrêt anticipé avec une patience de 10-20 époques
- **Division de validation** : Réservez 20% des données pour la validation

### Astuces pour un meilleur entraînement

1. **Commencez simplement** : Débutez avec un petit réseau et augmentez progressivement la complexité
2. **Régularisation** : Appliquez du dropout (0.2-0.3) pour éviter le surapprentissage
3. **Équilibre des données** : Assurez une représentation équilibrée des virages et des segments droits
4. **Transfer Learning** : Utilisez les poids de modèles réussis comme points de départ pour de nouveaux

## Métriques d'évaluation

### Métriques numériques

- **Mean Squared Error (MSE)** : Erreur de prédiction globale
- **Mean Absolute Error (MAE)** : Écart moyen de l'angle de direction
- **Maximum Error** : Plus grande erreur de prédiction de direction

### Performance en simulation

- **Taux de complétion de piste** : Pourcentage de réussite de complétion de piste
- **Maintien de voie** : Distance moyenne par rapport au centre de la piste
- **Fluidité** : Variation des commandes de direction (plus bas est meilleur)
- **Capacité de récupération** : Taux de réussite dans la récupération à partir de positions limites

## Structure du projet

- `data_preprocessor.py` : Fonctions pour charger et prétraiter les données
- `model_builder.py` : Définitions d'architectures de réseaux neuronaux
- `trainer.py` : Pipeline d'entraînement et utilitaires
- `evaluator.py` : Métriques d'évaluation et visualisation
- `train.py` : Script principal pour lancer l'entraînement

## Intégration avec les autres modules

- **Module collecteur** : Utilise les données collectées par `src/collector` pour l'entraînement
- **Module d'inférence** : Les modèles entraînés sont chargés par `src/inference` pour la conduite autonome
