# PredictiveDWA

A Predictive Dynamic Window Approach (DWA) implementation for robot navigation in corridors with moving people, featuring reinforcement learning for dynamic parameter tuning.

## Installation

### Prerequisites
- Python 3.11.x
- pip

### Setup

1. **Install Python**
   - Download from https://www.python.org/downloads/
   - Ensure "Add Python to PATH" is checked during installation

2. **Verify Installation**
   ```bash
   python --version
   ```

3. **Create Virtual Environment**
   ```bash
   python -m venv venv
   ```

4. **Activate Virtual Environment**
   ```bash
   # macOS/Linux
   source venv/bin/activate
   
   # Windows Command Prompt
   venv\Scripts\activate.bat
   
   # Windows PowerShell
   venv\Scripts\Activate.ps1
   ```

5. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Running the Modules

### GUI Module

The GUI provides an interactive interface for running and tuning the simulation with real-time parameter adjustment.

**Run the GUI:**
```bash
python run_gui.py
```

**Alternative GUI Versions:**
```bash
# Simple version (main thread)
python src/main/gui_simple.py

# Subprocess version (recommended for compatibility)
python src/main/gui_subprocess.py

# macOS-specific version
python src/main/gui_macos.py
```

### Basic Simulation

Run the simulation directly without GUI:

```bash
# With visualization
python src/main/run.py

# Headless mode (faster, no visualization)
python src/main/run.py --no-render
```

### Learning Module

#### Training

Train the theta-range Q-network:
```bash
python src/learning/train_theta_range.py
```

Trained models are saved to `checkpoints/theta_qnet.pt` with hyperparameters in `checkpoints/hyperparameters.json`.

#### Evaluation

Test a trained model:
```bash
# Without visualization
python src/learning/test_theta_eval.py

# With visualization
python src/learning/test_theta_eval.py --render

# Custom model path and number of episodes
python src/learning/test_theta_eval.py --model checkpoints/custom_model.pt --episodes 5
```

### Monte Carlo Simulations

Run multiple headless simulations for statistical analysis:

```bash
# Basic usage (10 runs)
python src/learning/monte_carlo.py

# Custom parameters
python src/learning/monte_carlo.py --runs 50 --max-time 30 --export-summary

# All available options
python src/learning/monte_carlo.py --help
```

**Common Options:**
- `--runs N`: Number of simulation runs (default: 10)
- `--corridor-width W`: Corridor width in meters (default: 4.0)
- `--door-side {left,right}`: Door side (default: right)
- `--num-people N`: Number of people (default: 3)
- `--people-speed-min X`: Minimum people speed (default: 0.6)
- `--people-speed-max X`: Maximum people speed (default: 1.2)
- `--max-time T`: Maximum simulation time per run (default: 60.0)
- `--export-summary`: Export summary statistics to CSV

**Output:**
- Individual run CSVs: `mc_run_XXX_YYYYMMDD_HHMMSS.csv`
- Summary CSV (with `--export-summary`): `monte_carlo_summary_YYYYMMDD_HHMMSS.csv`

## Project Structure

```
PredictiveDWA/
├── src/
│   ├── main/           # Main simulation and GUI runners
│   │   ├── run.py
│   │   ├── gui.py
│   │   └── gui_simple.py
│   ├── sim/            # Simulation engine
│   │   ├── sim.py
│   │   ├── robot.py
│   │   └── person.py
│   ├── algo/           # Navigation algorithms
│   │   ├── dwa.py
│   │   └── ts_dwa.py
│   └── learning/       # RL training and evaluation
│       ├── rl_theta_net.py
│       ├── train_theta_range.py
│       ├── test_theta_eval.py
│       └── monte_carlo.py
├── checkpoints/        # Saved models
├── logs/              # Training logs
├── simulation_data/   # Monte Carlo output
├── requirements.txt
└── README.md
```

## Troubleshooting

### Python not found
- Ensure Python is added to PATH during installation
- Restart your terminal after installation

### Permission errors
- Run terminal/command prompt as Administrator (Windows)
- Use `sudo` if needed (macOS/Linux)

### Virtual environment issues
- Delete the `venv` folder and recreate it
- Ensure you're using the correct activation script for your shell

### pygame/GUI issues
- Verify pygame installation: `pip install pygame`
- On macOS, try the macOS-specific GUI version
- If GUI becomes unresponsive, try the subprocess version

### PIL/Pillow not found
```bash
pip install Pillow
```

Notes:

- In file "run.py"
    The function 'draw' (from sim.py) draws the simulation's frame.
    'draw_v0.py' will do the same but with printing:
        1) pepole number
        2) Robot's speed
        3) Robot position


Test Commit after merge