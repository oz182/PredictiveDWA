#!/usr/bin/env python3
"""
Quick Start Guide for Curriculum Learning Training
===================================================

This script demonstrates how to start training with the new curriculum learning implementation.
"""

import os
import sys

def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print('='*70)

def main():
    print_header("CURRICULUM LEARNING - QUICK START")
    
    print("\n📚 WHAT'S NEW:")
    print("  1. Door inflation radius varies: 0.8m - 2.5m")
    print("  2. Door position varies: 25% - 65% of corridor length")
    print("  3. Door side varies: left or right")
    print("  4. Curriculum learning: gradually increases difficulty")
    print("  5. Agent learns generalizable avoidance behaviors")
    
    print_header("CURRICULUM STAGES")
    
    stages = [
        ("Easy (0-20%)", "Radius: 1.6-2.0m | Position: 7-9m | Side: Right only"),
        ("Medium (20-50%)", "Radius: 1.2-2.3m | Position: 6-11m | Side: Right only"),
        ("Hard (50-80%)", "Radius: 0.8-2.5m | Position: 5-13m | Side: Left or Right"),
        ("Expert (80-100%)", "Radius: 0.8-2.5m | Position: 5-13m | Side: Left or Right"),
    ]
    
    for stage, params in stages:
        print(f"\n  {stage:20s}: {params}")
    
    print_header("HOW TO START TRAINING")
    
    print("\n  Option 1: Basic Training (500 episodes)")
    print("  " + "-"*66)
    print("    cd src/learning")
    print("    python train.py --episodes 500")
    
    print("\n  Option 2: Training with WandB Logging")
    print("  " + "-"*66)
    print("    cd src/learning")
    print("    python train.py --episodes 500 --use-wandb")
    
    print("\n  Option 3: Custom Configuration")
    print("  " + "-"*66)
    print("    cd src/learning")
    print("    python train.py --episodes 1000 \\")
    print("                    --lr-actor 0.0005 \\")
    print("                    --lr-critic 0.001 \\")
    print("                    --use-wandb")
    
    print_header("WHAT TO MONITOR")
    
    print("\n  During Training, watch for:")
    print("    • curriculum_stage: Should progress from easy → expert")
    print("    • door_halo_radius: Should show increasing variance")
    print("    • overlap_door_pct: Should stay low (<15%)")
    print("    • return: May dip at stage transitions, should recover")
    
    print_header("FILES MODIFIED")
    
    files = [
        ("src/sim/robot.py", "Added door_halo_radius parameter"),
        ("src/sim/sim.py", "Added curriculum parameter support"),
        ("src/learning/train.py", "Added curriculum scheduler + updated loop"),
    ]
    
    for filepath, description in files:
        print(f"    ✓ {filepath:30s} - {description}")
    
    print_header("DOCUMENTATION")
    
    print("\n  📖 For detailed information, see:")
    print("      CURRICULUM_LEARNING.md")
    
    print("\n  📊 For implementation summary, see:")
    print("      This file explains:")
    print("        - Why curriculum learning helps")
    print("        - What behaviors the agent should learn")
    print("        - How to evaluate generalization")
    print("        - How to customize the curriculum")
    
    print_header("EXPECTED RESULTS")
    
    print("\n  After training, the agent should:")
    print("    ✓ Adapt clearance distance based on door radius")
    print("    ✓ Avoid doors at different corridor positions")
    print("    ✓ Handle doors on both left and right sides")
    print("    ✓ Generalize to unseen configurations")
    print("    ✓ Use the radius observation feature effectively")
    
    print("\n" + "="*70)
    print("  Ready to train! Run the commands above to get started.")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()



