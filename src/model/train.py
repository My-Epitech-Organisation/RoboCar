"""
Main training script for RoboCar neural network.

This script:
1. Loads and preprocesses data
2. Configures and creates a neural network model
3. Trains the model on collected data
4. Evaluates performance and saves the model
"""

import os
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from data_preprocessor import load_session, preprocess_data, split_data, augment_data
from model_builder import create_model
from trainer import train_model
from evaluator import ModelEvaluator

def main():
    # Configuration des arguments
    parser = argparse.ArgumentParser(description='Entraînement de modèle RoboCar')
    # parser.add_argument('--data_dir', type=str, default='data/raw',
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Répertoire des données brutes')
    parser.add_argument('--model_type', type=str, default='hybrid',
                        choices=['simple', 'cnn', 'lstm', 'hybrid', 'multi'],
                        help='Type de modèle à entraîner')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Nombre d\'époques d\'entraînement')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Taille des batchs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Taux d\'apprentissage')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion des données pour la validation')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Activer l\'augmentation des données')
    parser.add_argument('--no_augment', action='store_false', dest='augment',
                        help='Désactiver l\'augmentation des données')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed pour la reproductibilité')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Répertoire de sortie pour les modèles')
    
    args = parser.parse_args()
    
    # Garantir la reproductibilité
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Timestamp pour l'identifiant unique de cette session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"{args.model_type}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # 1. Charger et préparer les données
    print("Chargement des données...")
    csv_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"Aucun fichier CSV trouvé dans {args.data_dir}")
        return
    
    # Charger et combiner les données
    all_data = []
    for csv_file in csv_files:
        print(f"  Traitement de {os.path.basename(csv_file)}")
        data = load_session(csv_file)
        all_data.append(data)
    
    data = pd.concat(all_data, ignore_index=True)
    print(f"Total de {len(data)} échantillons chargés")
    
    # 2. Prétraiter les données
    print("Prétraitement des données...")
    X, y = preprocess_data(data, normalize=True)
    
    # 3. Augmenter les données si demandé
    if args.augment:
        print("Augmentation des données...")
        X, y = augment_data(X, y)
        print(f"Données augmentées à {len(X)} échantillons")
    
    # 4. Diviser en ensembles d'entraînement et de validation
    X_train, X_val, y_train, y_val = split_data(X, y, test_size=args.test_size)
    print(f"Ensemble d'entraînement: {len(X_train)} échantillons")
    print(f"Ensemble de validation: {len(X_val)} échantillons")
    
    # 5. Créer le modèle
    print(f"Création du modèle {args.model_type}...")
    input_size = X_train.shape[1]
    num_rays = input_size - 1  # Supposant que le dernier élément est la vitesse
    
    model = create_model(
        args.model_type, 
        input_size=input_size, 
        num_rays=num_rays
    )
    print(model)
    
    # 6. Entraîner le modèle
    print("Démarrage de l'entraînement...")
    model, history = train_model(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        project_root=".",  # Enregistre le meilleur modèle à la racine
        use_scheduler=True,
        early_stopping_patience=15
    )
    
    # 7. Évaluer le modèle
    print("Évaluation du modèle...")
    evaluator = ModelEvaluator(model, X_val, y_val)
    metrics = evaluator.calculate_metrics()
    
    # Afficher les métriques principales
    print("\nMétriques d'évaluation:")
    print(f"  Direction - MAE: {metrics['steering']['mae']:.4f}, MSE: {metrics['steering']['mse']:.4f}")
    print(f"  Accélération - MAE: {metrics['acceleration']['mae']:.4f}, MSE: {metrics['acceleration']['mse']:.4f}")
    
    # 8. Sauvegarder le rapport d'évaluation
    print("Génération du rapport d'évaluation...")
    evaluator.export_report(run_dir)
    
    # 9. Visualiser l'historique d'entraînement
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Perte d\'entraînement')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_steering_mae'], label='Direction')
    plt.plot(history['val_accel_mae'], label='Accélération')
    plt.title('MAE de validation')
    plt.xlabel('Époque')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'training_history.png'))
    
    print(f"\nEntraînement terminé. Tous les résultats sauvegardés dans {run_dir}")
    print(f"Le meilleur modèle a été sauvegardé à model_checkpoint.pth")

if __name__ == "__main__":
    main()
