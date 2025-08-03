#!/usr/bin/env python3
"""
Launcher script for the Embedded Predictive DWA Simulation GUI
This version runs the simulation within the GUI canvas instead of a separate window
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the embedded GUI
from main.gui import main

if __name__ == "__main__":
    main() 