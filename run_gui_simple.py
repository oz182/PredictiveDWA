#!/usr/bin/env python3
"""
Launcher script for the Simple Predictive DWA Simulation GUI
This version runs the simulation in the main thread to avoid threading issues
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the simple GUI
from main.gui_simple import main

if __name__ == "__main__":
    main() 