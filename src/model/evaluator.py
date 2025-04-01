"""
Evaluation module for RoboCar neural networks.

This module:
1. Calculates performance metrics
2. Visualizes predictions vs. actual values
3. Analyzes model behavior in different scenarios
"""

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARNING] Matplotlib not available in evaluator module. Visualization functions disabled.")

try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARNING] scikit-learn not available. Using numpy implementations for metrics.")


def evaluate_model(model, X_test, y_test, device=None):
    """
    Evaluate model performance on test data.
    
    Args:
        model (nn.Module): Trained PyTorch model
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test targets
        device (torch.device): Device for computation
        
    Returns:
        dict: Evaluation metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert data to tensors
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # Set model to evaluation mode
    model.eval()
    model = model.to(device)
    
    # Make predictions
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        y_pred = y_pred_tensor.cpu().numpy().flatten()
    
    # Calculate metrics
    if SKLEARN_AVAILABLE:
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    else:
        mse = ((y_test - y_pred) ** 2).mean()
        mae = np.abs(y_test - y_pred).mean()
        # Simple R² implementation
        y_mean = y_test.mean()
        ss_total = ((y_test - y_mean) ** 2).sum()
        ss_residual = ((y_test - y_pred) ** 2).sum()
        r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    
    max_error = np.max(np.abs(y_test - y_pred))
    
    # Print metrics
    print(f"Evaluation Metrics:")
    print(f"  Mean Squared Error (MSE): {mse:.6f}")
    print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    print(f"  R² Score: {r2:.6f}")
    print(f"  Maximum Error: {max_error:.6f}")
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'max_error': max_error,
        'predictions': y_pred,
        'actual': y_test
    }


def visualize_predictions(y_true, y_pred, title="Model Predictions vs. Actual Values"):
    """
    Visualize model predictions against actual values.
    
    Args:
        y_true (np.ndarray): Actual values
        y_pred (np.ndarray): Predicted values
        title (str): Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[WARNING] Cannot visualize predictions: matplotlib is not available")
        return
        
    plt.figure(figsize=(12, 6))
    
    # Plot sample indices vs values
    indices = np.arange(len(y_true))
    plt.subplot(2, 1, 1)
    plt.plot(indices, y_true, 'b-', label='Actual', alpha=0.7)
    plt.plot(indices, y_pred, 'r-', label='Predicted', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Steering Value')
    plt.title(f"{title} - Time Series View")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot scatter of predicted vs actual
    plt.subplot(2, 1, 2)
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Plot perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction Scatter Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """
    Plot training and validation loss over epochs.
    
    Args:
        history (dict): Training history dictionary
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[WARNING] Cannot plot training history: matplotlib is not available")
        return
        
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], 'b-', label='Training Loss')
    plt.plot(history['val_loss'], 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot epoch times
    plt.subplot(1, 2, 2)
    plt.plot(history['epoch_times'], 'g-')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Epoch Training Time')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_error_distribution(y_true, y_pred):
    """
    Analyze the distribution of prediction errors.
    
    Args:
        y_true (np.ndarray): Actual values
        y_pred (np.ndarray): Predicted values
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[WARNING] Cannot analyze error distribution: matplotlib is not available")
        return
        
    errors = y_true - y_pred
    
    plt.figure(figsize=(12, 5))
    
    # Error histogram
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=30, alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # Error vs. actual value
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xlabel('Actual Value')
    plt.ylabel('Error')
    plt.title('Error vs. Actual Value')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
