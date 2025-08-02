#!/usr/bin/env python3
"""
Launcher script for the macOS Fixed Predictive DWA Simulation GUI
This version includes specific fixes for the black screen issue on macOS
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the macOS fixed GUI
from main.gui_macos_fixed import main

if __name__ == "__main__":
    main() 