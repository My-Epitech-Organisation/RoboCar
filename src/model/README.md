# RoboCar Neural Network Training

This module trains neural networks with data collected from the RoboCar racing simulator for autonomous driving. It provides tools for data preprocessing, model creation, training, evaluation, and deployment.

## Overview

The neural network training pipeline:
1. Loads and preprocesses data collected from the simulation
2. Builds and configures neural network architectures
3. Trains models on the collected data
4. Evaluates model performance using multiple metrics
5. Deploys trained models for autonomous driving

## Features

- **Data Preprocessing**: Tools for normalizing, augmenting, and preparing collected data
- **Model Architectures**: Multiple pre-configured neural network architectures optimized for the task
- **Training Pipeline**: End-to-end training process with checkpointing and early stopping
- **Evaluation Tools**: Comprehensive metrics for assessing model performance
- **Visualization**: Tools to visualize training progress and model predictions
- **Deployment**: Methods to export and use trained models in the simulator

## Requirements

- Python 3.7+
- TensorFlow 2.x or PyTorch
- NumPy, Pandas, Matplotlib
- scikit-learn

## Installation

Install the required dependencies:
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

## Usage

### Data Preparation

```python
from data_preprocessor import load_session, preprocess_data, split_data

# Load collected data
data = load_session('data/raw/session_1234567890.csv')

# Preprocess data
X, y = preprocess_data(data, normalize=True, augment=True)

# Split into training and validation sets
X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2)
```

### Model Training

```python
from model_builder import create_cnn_model
from trainer import train_model

# Create model
model = create_cnn_model(input_shape=(10,))  # For 10 raycasts

# Train model
history = train_model(
    model,
    X_train, y_train,
    X_val, y_val,
    epochs=100,
    batch_size=32,
    save_path='models/cnn_model'
)
```

### Model Evaluation

```python
from evaluator import evaluate_model, visualize_predictions

# Evaluate model on test data
metrics = evaluate_model(model, X_test, y_test)
print(f"MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}")

# Visualize predictions
visualize_predictions(model, X_test, y_test)
```

### Model Deployment

```python
from deployer import load_model, prepare_for_inference

# Load saved model
model = load_model('models/cnn_model')

# Prepare for real-time inference
inference_model = prepare_for_inference(model)

# Example inference
steering_angle = inference_model.predict(observation)
```

## Data Preparation

### Loading Data

The collector module saves data in CSV format with the following columns:
- `timestamp`: Unix timestamp
- `steering_input`: User steering input (-1.0 to 1.0)
- `acceleration_input`: User acceleration input (-1.0 to 1.0)
- `raycasts`: Array of distances from car to obstacles
- `speed`: Current car speed
- `steering`: Current steering angle
- `position_x`, `position_y`, `position_z`: Car position coordinates

> **Important**: The number of raycasts in your data is determined by the `nbRay` parameter in `config/agent_config.json`. Your neural network's input layer must be compatible with this value. If you change the number of rays in the configuration, you will need to adjust your model architecture accordingly.

### Preprocessing Steps

1. **Parsing**: Convert raycast strings to numerical arrays
2. **Normalization**: Scale raycast values to [0, 1] range
3. **Feature Engineering**: Calculate additional features like distance to track center
4. **Augmentation**: Generate additional samples through mirroring and noise addition
5. **Sequence Creation**: For recurrent models, create time sequence samples

## Model Architectures

When building your models, always consider the input dimensions defined by your configuration:

```python
# Read the number of rays from config
import json
import os

def get_num_rays():
    config_path = os.path.join("config", "agent_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
        return config["agents"][0]["nbRay"]

# Use this value when creating your model
num_rays = get_num_rays()
model = create_cnn_model(input_shape=(num_rays,))
```

### Convolutional Neural Network (CNN)

Suitable for spatial processing of raycast data:

```python
def create_cnn_model(input_shape, learning_rate=0.001):
    model = Sequential([
        Reshape((input_shape[0], 1), input_shape=input_shape),
        Conv1D(16, 3, activation='relu', padding='same'),
        Conv1D(32, 3, activation='relu', padding='same'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1)  # Steering angle output
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse'
    )
    return model
```

### Recurrent Neural Network (RNN/LSTM)

For sequential decision making considering past observations:

```python
def create_lstm_model(sequence_length, features, learning_rate=0.001):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(sequence_length, features)),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1)  # Steering angle output
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse'
    )
    return model
```

### Multi-Input Model

For combining raycast data with other information like speed:

```python
def create_multi_input_model(raycast_shape, learning_rate=0.001):
    # Raycast input branch
    raycast_input = Input(shape=(raycast_shape,))
    x1 = Dense(64, activation='relu')(raycast_input)
    
    # Speed input branch
    speed_input = Input(shape=(1,))
    x2 = Dense(8, activation='relu')(speed_input)
    
    # Merge branches
    merged = Concatenate()([x1, x2])
    merged = Dense(32, activation='relu')(merged)
    output = Dense(1)(merged)  # Steering angle
    
    model = Model(inputs=[raycast_input, speed_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse'
    )
    return model
```

## Training Best Practices

### Data Collection Guidelines

1. **Quantity**: Collect at least 30 minutes of driving data
2. **Diversity**: Include:
   - Multiple laps on different tracks
   - Varying speeds
   - Recovery from edge cases
   - Smooth driving through corners
3. **Quality**: Ensure consistent, smooth driving patterns

### Training Parameters

- **Batch Size**: 32-64 (smaller for more precise gradient updates)
- **Learning Rate**: Start with 0.001, reduce if training is unstable
- **Epochs**: Use early stopping with patience of 10-20 epochs
- **Validation Split**: Reserve 20% of data for validation

### Tips for Better Training

1. **Start Simple**: Begin with a small network and gradually increase complexity
2. **Regularization**: Apply dropout (0.2-0.3) to prevent overfitting
3. **Data Balance**: Ensure balanced representation of turns and straight segments
4. **Transfer Learning**: Use weights from successful models as starting points for new ones

## Evaluation Metrics

### Numerical Metrics

- **Mean Squared Error (MSE)**: Overall prediction error
- **Mean Absolute Error (MAE)**: Average steering angle deviation
- **Maximum Error**: Largest steering prediction error

### Simulation Performance

- **Track Completion Rate**: Percentage of successful track completions
- **Lane Keeping**: Average distance from track center
- **Smoothness**: Variation in steering commands (lower is better)
- **Recovery Ability**: Success rate in recovering from edge positions

## Common Issues and Solutions

1. **Overfitting**:
   - **Symptoms**: Low training error, high validation error
   - **Solutions**: More data, dropout layers, data augmentation

2. **Underfitting**:
   - **Symptoms**: High training and validation error
   - **Solutions**: Larger model, longer training, feature engineering

3. **Oscillating Steering**:
   - **Symptoms**: Car zigzags on straight segments
   - **Solutions**: Smooth labels, add penalty for rapid steering changes

4. **Failure in Specific Scenarios**:
   - **Symptoms**: Model performs well except in certain cases
   - **Solutions**: Collect more data for those specific scenarios

## Tips for Better Autonomous Driving

1. **Consistency Over Speed**: Prioritize staying on track over driving fast
2. **Smooth Predictions**: Consider applying a moving average to steering predictions
3. **Safety Margins**: Train the model to maintain a safe distance from track edges
4. **Combined Approaches**: Consider ensemble methods combining multiple models

## Project Structure

- `data_preprocessor.py`: Functions for loading and preprocessing data
- `model_builder.py`: Neural network architecture definitions
- `trainer.py`: Training pipeline and utilities
- `evaluator.py`: Evaluation metrics and visualization
- `deployer.py`: Model export and inference utilities
