#!/usr/bin/env python3
"""
Launcher script for the macOS Predictive DWA Simulation GUI
This version is specifically designed to work on macOS systems
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the macOS GUI
from main.gui_macos import main

if __name__ == "__main__":
    main() 