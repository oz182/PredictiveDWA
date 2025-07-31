#!/usr/bin/env python3
"""
Main entry point for the OOP DWA simulation.
This file demonstrates how to use the refactored simulation classes.
"""

from simulation import Simulation

def main():
    """Main function to run the simulation."""
    print("Starting OOP DWA Simulation...")
    print("Press ESC to quit")
    
    # Create and run simulation
    sim = Simulation()
    
    # Optional: print initial obstacle information
    # sim.print_obstacles()
    
    # Run the simulation
    sim.run()

if __name__ == "__main__":
    main() 