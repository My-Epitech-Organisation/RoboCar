"""
Evaluation module for RoboCar neural networks.

This module:
1. Calculates performance metrics
2. Visualizes predictions vs. actual values
3. Analyzes model behavior in different scenarios
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import os

class ModelEvaluator:
    """
    Évaluateur complet de modèle qui analyse les performances sous différents angles.
    """
    def __init__(self, model, X_test, y_test, device='cpu'):
        self.model = model.to(device)
        self.X_test = torch.FloatTensor(X_test).to(device)
        self.y_test = y_test
        self.device = device
        self.predictions = None
        
    def predict(self):
        """Génère des prédictions pour l'ensemble de test"""
        self.model.eval()
        with torch.no_grad():
            self.predictions = self.model(self.X_test).cpu().numpy()
        return self.predictions
    
    def calculate_metrics(self):
        """Calcule les métriques d'évaluation"""
        if self.predictions is None:
            self.predict()
            
        # Séparer direction et accélération
        y_steering = self.y_test[:, 0]
        y_accel = self.y_test[:, 1]
        pred_steering = self.predictions[:, 0]
        pred_accel = self.predictions[:, 1]
        
        # Métriques pour la direction
        steering_metrics = {
            'mse': mean_squared_error(y_steering, pred_steering),
            'mae': mean_absolute_error(y_steering, pred_steering),
            'max_error': np.max(np.abs(y_steering - pred_steering)),
            'correlation': np.corrcoef(y_steering, pred_steering)[0, 1]
        }
        
        # Métriques pour l'accélération
        accel_metrics = {
            'mse': mean_squared_error(y_accel, pred_accel),
            'mae': mean_absolute_error(y_accel, pred_accel),
            'max_error': np.max(np.abs(y_accel - pred_accel)),
            'correlation': np.corrcoef(y_accel, pred_accel)[0, 1]
        }
        
        return {
            'steering': steering_metrics,
            'acceleration': accel_metrics
        }
    
    def visualize_predictions(self, sample_indices=None, save_path=None):
        """
        Visualise les prédictions vs les valeurs réelles
        
        Args:
            sample_indices: Indices à visualiser, si None utilise les 100 premiers
            save_path: Chemin où sauvegarder la figure
        """
        if self.predictions is None:
            self.predict()
            
        if sample_indices is None:
            sample_indices = range(min(100, len(self.y_test)))
            
        # Extraire les données à visualiser
        y_steering = self.y_test[sample_indices, 0]
        y_accel = self.y_test[sample_indices, 1]
        pred_steering = self.predictions[sample_indices, 0]
        pred_accel = self.predictions[sample_indices, 1]
        
        # Créer la figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Tracer les prédictions de direction
        ax1.plot(y_steering, label='Direction réelle', color='blue')
        ax1.plot(pred_steering, label='Direction prédite', color='red', linestyle='--')
        ax1.set_title('Prédictions de direction')
        ax1.set_ylabel('Angle de direction')
        ax1.legend()
        ax1.grid(True)
        
        # Tracer les prédictions d'accélération
        ax2.plot(y_accel, label='Accélération réelle', color='blue')
        ax2.plot(pred_accel, label='Accélération prédite', color='red', linestyle='--')
        ax2.set_title('Prédictions d\'accélération')
        ax2.set_xlabel('Échantillons')
        ax2.set_ylabel('Commande d\'accélération')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Figure sauvegardée à {save_path}")
        
        plt.show()
        
    def export_report(self, output_dir):
        """
        Exporte un rapport complet d'évaluation
        """
        # Créer le répertoire si nécessaire
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculer les métriques
        metrics = self.calculate_metrics()
        
        # Sauvegarder les métriques au format JSON
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Générer et sauvegarder les visualisations
        viz_path = os.path.join(output_dir, 'predictions_comparison.png')
        self.visualize_predictions(save_path=viz_path)
        
        # Analyser les erreurs
        self.analyze_errors(output_dir)
        
        print(f"Rapport d'évaluation complet exporté dans {output_dir}")
        
    def analyze_errors(self, output_dir):
        """
        Analyse détaillée des erreurs de prédiction
        """
        if self.predictions is None:
            self.predict()
            
        # Calculer les erreurs
        steering_errors = np.abs(self.y_test[:, 0] - self.predictions[:, 0])
        accel_errors = np.abs(self.y_test[:, 1] - self.predictions[:, 1])
        
        # Trouver les cas les plus problématiques
        worst_steering_idx = np.argsort(steering_errors)[-10:]
        worst_accel_idx = np.argsort(accel_errors)[-10:]
        
        # Sauvegarder les résultats
        error_analysis = {
            'worst_steering_cases': [
                {
                    'index': int(idx),
                    'actual': float(self.y_test[idx, 0]),
                    'predicted': float(self.predictions[idx, 0]),
                    'error': float(steering_errors[idx])
                }
                for idx in worst_steering_idx
            ],
            'worst_acceleration_cases': [
                {
                    'index': int(idx),
                    'actual': float(self.y_test[idx, 1]),
                    'predicted': float(self.predictions[idx, 1]),
                    'error': float(accel_errors[idx])
                }
                for idx in worst_accel_idx
            ]
        }
        
        # Sauvegarder l'analyse
        error_path = os.path.join(output_dir, 'error_analysis.json')
        with open(error_path, 'w') as f:
            json.dump(error_analysis, f, indent=4)
        
        # Visualiser la distribution des erreurs
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(steering_errors, bins=30)
        plt.title('Distribution des erreurs de direction')
        plt.xlabel('Erreur absolue')
        plt.ylabel('Fréquence')
        
        plt.subplot(1, 2, 2)
        plt.hist(accel_errors, bins=30)
        plt.title('Distribution des erreurs d\'accélération')
        plt.xlabel('Erreur absolue')
        plt.ylabel('Fréquence')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
