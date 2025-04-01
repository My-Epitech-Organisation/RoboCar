# RoboCar Data Collector

This module collects training data from the RoboCar racing simulator for supervised learning models. It manages the connection to the Unity simulation, records sensor data and user inputs, and saves everything in a format suitable for training AI models.

## Overview

The data collector:
1. Connects to the Unity Racing Simulator
2. Captures user inputs (keyboard or joystick)
3. Records simulation data (raycasts, speed, steering angles, position)
4. Saves synchronized data to CSV files for training

## Features

- **Simulator Integration**: Direct connection to Unity Racing Simulator via ML-Agents
- **Input Methods**: Support for both keyboard and joystick control
- **Calibration Tool**: Built-in graphical joystick calibration utility
- **Configurable Sensors**: Adjustable raycast count and field of view
- **Real-time Feedback**: Terminal display of all sensor readings and inputs
- **Data Persistence**: Automatic CSV generation with timestamped filenames
- **Robust Error Handling**: Graceful recovery from disconnections and errors

## Requirements

- Python 3.7+
- Unity Racing Simulator
- ML-Agents package
- pygame
- numpy
- pynput

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd RoboCar
   ```

2. Install dependencies:
   ```bash
   pip install mlagents pygame numpy pynput
   ```

3. Ensure the Unity Racing Simulator executable has execution permissions:
   ```bash
   chmod +x RacingSimulatorLinux/RacingSimulator.x86_64
   ```

## Usage

### Starting Data Collection

Basic usage:
```bash
python src/collector/collect_data.py
```

With joystick calibration at startup:
```bash
python src/collector/collect_data.py --calibrate
```

### Controls

- **Keyboard**:
  - **Arrow keys** or **WASD/ZQSD**: Control the car (steering and acceleration)
  - **C**: Launch joystick calibration during runtime
  - **Ctrl+C**: Stop data collection

- **Joystick**:
  - **Left stick**: Control steering and acceleration
  - Joystick is automatically detected if present

### Configuration

#### Simulation Settings

Configure graphics and physics in `config/raycast_config.json`:
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

#### Agent Configuration

Configure the agent's sensors in `config/agent_config.json`:
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

Parameters:
- `fov`: Field of view in degrees (1-180)
- `nbRay`: Number of raycasts (1-50)

> **Important for Neural Network Training**: The number of raycasts (`nbRay`) defined in this configuration file determines the input dimension of your neural network. Your model architecture must be compatible with this value. When training a neural network, ensure that the input layer can accept exactly this number of raycast values.

## Data Output

### File Location

Data is saved in CSV format in the `data/raw/` directory with filenames based on timestamp:
```
data/raw/session_1234567890.csv
```

### Data Format

Each CSV file contains the following columns:

| Column | Description |
|--------|-------------|
| timestamp | Unix timestamp when the data was recorded |
| steering_input | User steering input (-1.0 to 1.0) |
| acceleration_input | User acceleration input (-1.0 to 1.0) |
| raycasts | Array of distances from car to obstacles |
| speed | Current speed of the car |
| steering | Current steering angle of the car |
| position_x | X coordinate of the car |
| position_y | Y coordinate of the car |
| position_z | Z coordinate of the car |

### Observation Extraction

The data collector extracts observations from the Unity simulation using this pattern:
- Raycasts: `obs_array[:num_rays]`
- Speed: `obs_array[-5]`
- Steering: `obs_array[-4]`
- Position: `[obs_array[-3], obs_array[-2], obs_array[-1]]`

## Joystick Calibration

### Automatic Calibration

The calibration tool will guide you to move the joystick to its extremes to determine the maximum range of motion. The calibration data is saved and applied to normalize inputs.

### Manual Configuration

Calibration data is stored in `src/collector/joystick_calibration.json` and can be manually edited if needed:
```json
{
  "steering": {"min": -1.0, "max": 1.0},
  "acceleration": {"min": -1.0, "max": 1.0}
}
```

## Troubleshooting

### Common Issues

1. **Unity executable not found**:
   - Ensure the simulator is in the correct location: `RacingSimulatorLinux/RacingSimulator.x86_64`

2. **Joystick not detected**:
   - Connect the joystick before starting the program
   - Run `pygame.joystick.Joystick(0).get_name()` to verify detection

3. **Port already in use**:
   - If you see "port already in use" errors, ensure no other instances are running
   - Default port is 5004, can be changed in the code if needed

4. **Missing observations**:
   - Verify that `nbRay` in config matches the simulator settings
   - Check Unity logs for any errors

## Project Structure

- `collect_data.py`: Main script for data collection
- `utils_collector.py`: Input processing utilities
- `joystick_calibrator.py`: Joystick calibration functions
- `ui_components.py`: UI for joystick calibration
- `config/`: Configuration files for the simulator and agent

## License

[Include license information here]
