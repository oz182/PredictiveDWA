#!/usr/bin/env python3
"""
Simple test script to verify the simulation works on macOS
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_simulation():
    """Test if the simulation can be imported and run"""
    try:
        print("Testing simulation import...")
        from sim.sim import Simulation
        print("✓ Simulation imported successfully")
        
        print("Creating simulation instance...")
        sim = Simulation(
            corridor_width=4.0,
            door_side="right",
            num_people=3,
            people_speeds=[1.2, 1.0, 1.5]
        )
        print("✓ Simulation instance created successfully")
        
        print("Testing simulation step...")
        state, reward, done = sim.step(0.016)  # 60 FPS
        print(f"✓ Simulation step completed - State: {type(state)}, Reward: {reward}, Done: {done}")
        
        print("Testing pygame import...")
        import pygame
        print("✓ Pygame imported successfully")
        
        print("Testing pygame initialization...")
        pygame.init()
        print("✓ Pygame initialized successfully")
        
        print("Testing pygame display creation...")
        screen = pygame.display.set_mode((800, 400))
        print("✓ Pygame display created successfully")
        
        print("Testing simulation rendering...")
        sim.draw_v0(screen)
        pygame.display.flip()
        print("✓ Simulation rendered successfully")
        
        pygame.quit()
        print("✓ Pygame quit successfully")
        
        print("\n🎉 All tests passed! The simulation should work on your system.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Predictive DWA Simulation on macOS...")
    print("=" * 50)
    
    success = test_simulation()
    
    if success:
        print("\nYou can now try running the GUI:")
        print("python3 run_gui_macos.py")
    else:
        print("\nThere seems to be an issue with the simulation setup.")
        print("Please check the error message above.") 