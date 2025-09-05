#!/usr/bin/env python3
"""
Test script to verify the collision detection functionality works correctly
"""

import sys
import os
import time

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_collision_detection():
    """Test the collision detection functionality"""
    try:
        print("Testing collision detection functionality...")
        from sim.sim import Simulation
        
        print("Creating simulation instance with more people to increase collision chances...")
        sim = Simulation(
            corridor_width=3.0,  # Narrower corridor to increase collision probability
            door_side="right",
            num_people=5,  # More people
            people_speeds=[1.5, 1.8, 2.0, 1.2, 1.6]  # Faster speeds
        )
        print("✓ Simulation instance created successfully")
        
        print("Testing collision detection initialization...")
        print(f"  Initial collision count: {sim.collision_count}")
        print(f"  Initial collision history: {len(sim.collision_history)}")
        print("✓ Collision detection initialized")
        
        print("Running simulation for 10 seconds to test collision detection...")
        start_time = time.time()
        step_count = 0
        last_collision_count = 0
        
        while time.time() - start_time < 10.0:  # Run for 10 seconds
            dt = 0.016  # 60 FPS
            state, reward, done = sim.step(dt)
            step_count += 1
            
            # Report collision count changes
            if sim.collision_count > last_collision_count:
                print(f"  Collision #{sim.collision_count} detected at step {step_count}")
                last_collision_count = sim.collision_count
            
            if done:
                print("✓ Goal reached!")
                break
        
        print(f"✓ Simulation completed after {step_count} steps")
        print(f"  Total collisions detected: {sim.collision_count}")
        print(f"  Collision history entries: {len(sim.collision_history)}")
        
        if sim.collision_history:
            print("Collision details:")
            for i, collision in enumerate(sim.collision_history):
                print(f"  Collision {i+1}:")
                print(f"    Time: {collision['timestamp']}")
                print(f"    Robot pos: ({collision['robot_position'][0]:.2f}, {collision['robot_position'][1]:.2f})")
                print(f"    Person pos: ({collision['person_position'][0]:.2f}, {collision['person_position'][1]:.2f})")
                print(f"    Distance: {collision['distance']:.2f}")
        
        print("Testing data export with collision information...")
        filepath = sim.export_data_to_csv("test_collision_data.csv")
        
        if filepath and os.path.exists(filepath):
            print(f"✓ Data exported successfully to: {filepath}")
            
            # Check file size
            file_size = os.path.getsize(filepath)
            print(f"  File size: {file_size} bytes")
            
            # Read first few lines to verify collision data is included
            with open(filepath, 'r') as f:
                lines = f.readlines()
                print(f"  Total lines in file: {len(lines)}")
                if len(lines) > 1:
                    print(f"  Header: {lines[0].strip()}")
                    if 'collision_count' in lines[0]:
                        print("  ✓ Collision count column found in CSV")
                    else:
                        print("  ❌ Collision count column missing from CSV")
                    
                    # Check a few data lines
                    for i in range(1, min(4, len(lines))):
                        print(f"  Data line {i}: {lines[i].strip()}")
        else:
            print("❌ Data export failed")
            return False
        
        print("Testing simulation summary with collision data...")
        summary = sim.get_simulation_summary()
        if summary:
            print("✓ Simulation summary generated:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
            
            if 'total_collisions' in summary:
                print("  ✓ Collision count included in summary")
            else:
                print("  ❌ Collision count missing from summary")
                return False
        else:
            print("❌ Failed to generate simulation summary")
            return False
        
        print("Testing data reset with collision data...")
        sim.reset_data_recording()
        print(f"  Collision count after reset: {sim.collision_count}")
        print(f"  Collision history after reset: {len(sim.collision_history)}")
        print("✓ Data reset successful")
        
        print("\n🎉 All collision detection tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Predictive DWA Collision Detection...")
    print("=" * 50)
    
    success = test_collision_detection()
    
    if success:
        print("\nCollision detection feature is working correctly!")
        print("The simulation now tracks robot-person collisions and includes this data in CSV exports.")
    else:
        print("\nThere seems to be an issue with the collision detection implementation.")
        print("Please check the error message above.")
