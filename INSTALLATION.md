# PredictiveDWA Installation Guide

This guide will help you set up the Python environment and install all required dependencies for the PredictiveDWA project.

## Prerequisites

- Windows 10/11
- Internet connection for downloading packages

## Step 1: Install Python

### Option A: Official Python Installer (Recommended)
1. Go to https://www.python.org/downloads/
2. Download Python 3.11.x (latest stable version)
3. **IMPORTANT**: During installation, check "Add Python to PATH"
4. Complete the installation

### Option B: Microsoft Store
1. Open Microsoft Store
2. Search for "Python 3.11"
3. Install the official Python 3.11 app

## Step 2: Verify Python Installation

Open Command Prompt or PowerShell and run:
```bash
python --version
```

You should see something like `Python 3.11.x`

## Step 3: Set Up Project Environment

### Option A: Automated Setup (Recommended)

**Using Command Prompt:**
```bash
setup.bat
```

**Using PowerShell:**
```powershell
.\setup.ps1
```

### Option B: Manual Setup

1. **Create Virtual Environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate Virtual Environment:**
   ```bash
   # Command Prompt
   venv\Scripts\activate.bat
   
   # PowerShell
   venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Step 4: Run the Simulation

1. **Activate the environment:**
   ```bash
   # Command Prompt
   venv\Scripts\activate.bat
   
   # PowerShell
   venv\Scripts\Activate.ps1
   ```

2. **Run with visualization:**
   ```bash
   python src\main\run.py
   ```

3. **Run in headless mode (faster):**
   ```bash
   python src\main\run.py --no-render
   ```

## Project Dependencies

The project requires these Python packages:
- **pygame** (>=2.0.0): For visualization and simulation
- **numpy** (>=1.21.0): For numerical computations
- **matplotlib** (>=3.5.0): For plotting (if needed)

## Troubleshooting

### Python not found
- Make sure Python is added to PATH during installation
- Try restarting your terminal after installation

### Permission errors
- Run Command Prompt or PowerShell as Administrator
- Make sure you're in the project directory

### Virtual environment issues
- Delete the `venv` folder and recreate it
- Make sure you're using the correct activation script for your shell

## Project Structure

```
PredictiveDWA/
├── src/
│   ├── main/
│   │   └── run.py          # Main simulation runner
│   ├── sim/
│   │   ├── sim.py          # Simulation engine
│   │   ├── robot.py        # Robot implementation
│   │   └── person.py       # Person/agent implementation
│   └── algo/
│       ├── dwa.py          # Dynamic Window Approach
│       └── simple.py       # Simple navigation
├── requirements.txt         # Python dependencies
├── setup.bat              # Windows setup script
├── setup.ps1              # PowerShell setup script
└── INSTALLATION.md        # This file
```

## What the Simulation Does

This project implements a Predictive Dynamic Window Approach (DWA) for robot navigation in a corridor with moving people. The robot must navigate from one end of the corridor to the other while avoiding collisions with people moving through the corridor.

The simulation features:
- Real-time visualization using pygame
- Dynamic Window Approach algorithm for robot navigation
- Person simulation with realistic movement patterns
- Collision avoidance and path planning
- Configurable parameters for corridor width, number of people, etc. 