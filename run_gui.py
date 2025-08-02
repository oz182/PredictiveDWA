#!/usr/bin/env python3
"""
Launcher script for the Predictive DWA Simulation GUI
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the GUI
from main.gui import main

if __name__ == "__main__":
    main() 