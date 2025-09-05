# Predictive DWA Simulation GUI

This GUI provides an interactive interface for running and tuning the Predictive DWA (Dynamic Window Approach) simulation.

## Features

### Simulation Controls
- **Run**: Start the simulation
- **Stop**: Pause the simulation
- **Restart**: Reset and restart the simulation with current parameters

### DWA Parameters
The GUI allows real-time tuning of the following DWA parameters:

#### Basic Parameters
- **Max Speed**: Maximum linear velocity (0.5 - 5.0 m/s)
- **Max Rotation**: Maximum angular velocity (1.0 - 6.0 rad/s)
- **Max Accel**: Maximum linear acceleration (1.0 - 10.0 m/s²)
- **Max Angular Accel**: Maximum angular acceleration (2.0 - 12.0 rad/s²)
- **Predict Time**: How far ahead to predict trajectories (0.5 - 5.0 seconds)

#### Sampling Parameters
- **V Samples**: Number of linear velocity samples (1 - 20)
- **W Samples**: Number of angular velocity samples (1 - 20)

#### Scoring Weights
- **Goal Weight**: Weight for goal-seeking behavior (0.0 - 1.0)
- **Clearance Weight**: Weight for obstacle avoidance (0.0 - 1.0)
- **Velocity Weight**: Weight for maintaining speed (0.0 - 1.0)

### Simulation Parameters
- **Corridor Width**: Width of the corridor (2.0 - 8.0 meters)
- **Num People**: Number of people in the simulation (1 - 10)
- **Spawn Interval**: Time between spawning new people (0.5 - 3.0 seconds)
- **Door Side**: Which side of the corridor the door is on ("left" or "right")

## Usage

### Running the GUI

There are three different GUI versions available to handle different compatibility issues:

#### 1. **Threaded Version (Default)**
```bash
python3 run_gui.py
```
- Uses threading to run simulation in background
- May have issues on some systems (especially macOS)

#### 2. **Simple Version (Main Thread)**
```bash
python3 run_gui_simple.py
```
- Runs simulation in main thread
- GUI becomes unresponsive while simulation runs
- Most compatible but blocks GUI interaction

#### 3. **Subprocess Version (Recommended)**
```bash
python3 run_gui_subprocess.py
```
- Launches simulation in separate process
- GUI remains responsive
- Best compatibility across different systems
- **Recommended for most users**

#### 4. **macOS Version (macOS-specific)**
```bash
python3 run_gui_macos.py
```
- Specifically designed for macOS compatibility
- Includes console output for debugging
- Handles macOS-specific pygame issues
- **Recommended for macOS users**

#### 5. **Direct from src/main directory:**
```bash
cd src/main
python3 gui.py              # Threaded version
python3 gui_simple.py       # Simple version  
python3 gui_subprocess.py   # Subprocess version
python3 gui_macos.py        # macOS version
```

### Using the GUI

1. **Start the simulation:**
   - Click the "Run" button to start the simulation
   - The simulation will open in a separate pygame window

2. **Tune parameters:**
   - Adjust DWA parameters using the sliders and spinboxes
   - Click "Apply DWA Parameters" to apply changes
   - Adjust simulation parameters as needed
   - Click "Apply Simulation Parameters" to restart with new settings

3. **Control the simulation:**
   - Use "Stop" to pause the simulation
   - Use "Restart" to reset and restart with current parameters

## Tips for Parameter Tuning

### Goal Weight vs Clearance Weight
- Higher **Goal Weight** makes the robot more focused on reaching the goal
- Higher **Clearance Weight** makes the robot more cautious around obstacles
- These should typically sum to around 0.9, with the remaining 0.1 for velocity

### Speed and Acceleration
- **Max Speed** affects how fast the robot can move
- **Max Accel** affects how quickly the robot can change speed
- Higher values may cause more aggressive behavior

### Prediction Time
- Longer **Predict Time** allows the robot to plan further ahead
- Shorter values make the robot more reactive to immediate obstacles

### Sampling
- More **V Samples** and **W Samples** provide finer control but increase computation
- 8 samples each is usually a good balance

## Troubleshooting

- If the GUI doesn't start, ensure pygame is installed: `pip install pygame`
- If the simulation window doesn't appear, check that pygame is working correctly
- Parameter changes are applied immediately when you click the apply buttons 