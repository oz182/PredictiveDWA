#!/usr/bin/env python3
"""
Launcher script for the Subprocess Predictive DWA Simulation GUI
This version launches the simulation in a separate subprocess to avoid threading issues
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the subprocess GUI
from main.gui_subprocess import main

if __name__ == "__main__":
    main() 