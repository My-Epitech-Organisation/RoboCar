# RoboCar Autonomous Driving Inference

This module loads trained neural network models and uses them for autonomous driving in the RoboCar racing simulator. It connects to the Unity simulation, processes sensor data, and uses the model to generate driving commands.

## Overview

The inference module:
1. Loads the trained neural network model
2. Connects to the Unity Racing Simulator
3. Processes sensor readings (raycasts, speed, position)
4. Generates steering and acceleration commands
5. Sends commands to drive the car autonomously

## Features

- **Model Loading**: Loads trained PyTorch models with metadata
- **Real-time Inference**: Fast processing of sensor data
- **Steering Smoothing**: Reduces oscillations with temporal smoothing
- **Performance Monitoring**: Tracks FPS and inference times
- **Configuration Compatibility**: Ensures model matches raycast configuration

## Requirements

- Python 3.7+
- Unity Racing Simulator
- PyTorch
- ML-Agents package
- NumPy

## Usage

### Basic Usage

Start autonomous driving with the default model:

```bash
python src/inference/run_model.py
```

### Controls

- **Ctrl+C**: Stop the autonomous driving mode

### Model Selection

The system automatically loads the model file `model_checkpoint.pth` from the project root. This is the default output path when training a model with the training module.

## Implementation Details

### Observation Processing

The inference module processes observations from the Unity simulation in the same way they were processed during training:

1. Raycasts are extracted and normalized to [0,1] range
2. Vehicle speed is normalized
3. Features are combined into an input tensor for the model

### Steering Smoothing

To prevent erratic steering behavior, the module implements smoothing:

1. Maintains a history of recent steering predictions
2. Applies a moving average to smooth transitions
3. Limits the maximum steering change per frame

### Performance Optimization

For real-time control, the inference module:

1. Minimizes preprocessing overhead
2. Uses CPU inference to avoid GPU transfer delays
3. Monitors frame rates to ensure responsive control

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure the model file exists at the project root
   - Verify the model was trained with compatible data

2. **Simulator Connection Issues**:
   - Check that the simulator is running and on port 5004
   - Verify permissions on the simulator executable

3. **Erratic Driving Behavior**:
   - Try increasing the steering smoothing parameters
   - Collect more training data in problematic scenarios
   - Retrain the model with more diverse data

## Architecture

### Model Compatibility

The model input size must match the number of raycasts in your configuration:

```
model_input_size = num_rays + 1  # Rays + speed
```

If you change the number of raycasts in your configuration, you must retrain your model.
