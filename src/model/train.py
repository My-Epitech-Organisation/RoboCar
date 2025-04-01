"""
Main training script for RoboCar neural network.

This script:
1. Loads and preprocesses data
2. Configures and creates a neural network model
3. Trains the model on collected data
4. Evaluates performance and saves the model
"""

import os
import sys
import argparse
import numpy as np

# Check for required dependencies
missing_deps = []

try:
    import torch
except ImportError:
    missing_deps.append("torch")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    missing_deps.append("matplotlib")

if missing_deps:
    print("\n[ERROR] Missing required dependencies:")
    print("  The following packages are required but not installed:")
    for dep in missing_deps:
        print(f"    - {dep}")
    print("\nPlease install them using pip:")
    print(f"  pip install {' '.join(missing_deps)}")
    print("\nOr with your virtual environment:")
    print("  source .venv/bin/activate")
    print(f"  pip install {' '.join(missing_deps)}")
    sys.exit(1)

# Fix import path issues
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import local modules
import data_preprocessor
import model_builder
import trainer
if MATPLOTLIB_AVAILABLE:
    import evaluator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RoboCar Neural Network Training')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory with CSV data files')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable data augmentation')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='simple',
                        choices=['simple', 'cnn', 'lstm', 'multi'],
                        help='Type of model architecture')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Size of hidden layers')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save model and results (default: project root)')
    parser.add_argument('--model_name', type=str, default='model_checkpoint.pth',
                        help='Filename for saved model')
    
    # Visualization parameters
    parser.add_argument('--no_viz', action='store_true',
                        help='Disable visualization (useful if matplotlib is not available)')
    
    return parser.parse_args()


def get_project_root():
    """Return the project root path."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def evaluate_without_viz(model, X_test, y_test):
    """
    Simple evaluation without matplotlib visualization.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        dict: Evaluation metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # Set model to evaluation mode
    model.eval()
    model = model.to(device)
    
    # Make predictions
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        y_pred = y_pred_tensor.cpu().numpy().flatten()
    
    # Calculate basic metrics
    mse = ((y_test - y_pred) ** 2).mean()
    mae = np.abs(y_test - y_pred).mean()
    
    # Print metrics
    print(f"Evaluation Metrics:")
    print(f"  Mean Squared Error (MSE): {mse:.6f}")
    print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    
    return {
        'mse': mse,
        'mae': mae,
        'predictions': y_pred,
        'actual': y_test
    }


def main():
    """Main training function."""
    args = parse_arguments()
    project_root = get_project_root()
    
    # Check if visualization is disabled by argument or missing matplotlib
    visualization_enabled = not args.no_viz and MATPLOTLIB_AVAILABLE
    if not MATPLOTLIB_AVAILABLE and not args.no_viz:
        print("[WARNING] Matplotlib not available. Visualizations disabled.")
        print("[WARNING] Install matplotlib for visualizations: pip install matplotlib")
    
    print(f"RoboCar Neural Network Training")
    print(f"Project root: {project_root}")
    
    # Set paths
    data_dir = os.path.join(project_root, args.data_dir)
    output_dir = args.output_dir if args.output_dir else project_root
    model_path = os.path.join(output_dir, args.model_name)
    
    print(f"Data directory: {data_dir}")
    print(f"Output model path: {model_path}")
    
    # Get number of rays from config
    num_rays = data_preprocessor.get_num_rays_from_config(project_root)
    
    # Load and preprocess data
    print("\n[1] Loading and preprocessing data...")
    try:
        data = data_preprocessor.load_all_sessions(data_dir)
        X, y = data_preprocessor.preprocess_data(data, augment=not args.no_augment)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessor.split_data(X, y)
        
        print(f"Data shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_val: {X_val.shape}")
        print(f"  X_test: {X_test.shape}")
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        sys.exit(1)
    
    # Create model
    print("\n[2] Creating model...")
    try:
        input_size = X_train.shape[1]
        print(f"Input size: {input_size}")
        
        # Create model based on type
        model = model_builder.create_model(
            args.model_type, 
            input_size, 
            num_rays=num_rays
        )
        
        print(f"Created {args.model_type.upper()} model")
    except Exception as e:
        print(f"Error creating model: {e}")
        sys.exit(1)
    
    # Train model
    print("\n[3] Training model...")
    try:
        trained_model, history = trainer.train_model(
            model,
            X_train, y_train,
            X_val, y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            project_root=project_root
        )
        
        # Plot training history if visualization is enabled
        if visualization_enabled:
            evaluator.plot_training_history(history)
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)
    
    # Evaluate model
    print("\n[4] Evaluating model...")
    try:
        if visualization_enabled:
            metrics = evaluator.evaluate_model(trained_model, X_test, y_test)
            
            # Visualize predictions
            evaluator.visualize_predictions(
                metrics['actual'], 
                metrics['predictions'], 
                title=f"{args.model_type.upper()} Model Predictions"
            )
            
            # Analyze errors
            evaluator.analyze_error_distribution(
                metrics['actual'], 
                metrics['predictions']
            )
        else:
            # Use simplified evaluation without visualization
            metrics = evaluate_without_viz(trained_model, X_test, y_test)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)
    
    # Save model
    print("\n[5] Saving model...")
    try:
        # Save metadata with the model
        metadata = {
            'model_type': args.model_type,
            'num_rays': num_rays,
            'input_size': input_size,
            'metrics': {
                'mse': float(metrics['mse']),
                'mae': float(metrics['mae']),
            },
            'training_params': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate
            }
        }
        
        # Add r2 score if available
        if 'r2' in metrics:
            metadata['metrics']['r2'] = float(metrics['r2'])
        
        trainer.save_model(
            trained_model, 
            model_path, 
            input_size=input_size,
            model_type=args.model_type,
            metadata=metadata
        )
        
        print(f"Model successfully saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
        sys.exit(1)
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
