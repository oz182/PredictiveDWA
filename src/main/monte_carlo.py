#!/usr/bin/env python3
"""
Monte Carlo Simulation Runner for Predictive DWA
Runs multiple simulations without GUI/display and records data for analysis

Supports two modes:
  - default: Standard simulation using robot.py's configured navigator
  - learned: Uses trained TD3 model to control heading offset (like test.py)

Usage:
  # Run with default navigator (configured in robot.py)
  python monte_carlo.py --runs 10 --mode default

  # Run with learned model
  python monte_carlo.py --runs 10 --mode learned --model checkpoints/td3_policy_WorkingOffset2.pt

  # Run learned model with custom action interval
  python monte_carlo.py --runs 10 --mode learned --action-select-interval 5
"""

import sys
import os
import time
import random
import argparse
import statistics
import math
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import torch

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sim.sim import Simulation


class MonteCarloRunner:
    def __init__(self, num_runs: int = 10, corridor_width: float = 4.0, 
                 door_side: str = "right", num_people: int = 3,
                 people_speed_range: tuple = (0.6, 1.2), 
                 max_simulation_time: float = 60.0,
                 mode: str = "default",
                 model_path: Optional[str] = None,
                 action_select_interval: int = 1,
                 render: bool = False,
                 seed: Optional[int] = None):
        """
        Initialize Monte Carlo runner
        
        Args:
            num_runs: Number of simulation runs to perform
            corridor_width: Width of the corridor
            door_side: Which side the door is on ("left" or "right")
            num_people: Number of people in each simulation
            people_speed_range: Tuple of (min_speed, max_speed) for people
            max_simulation_time: Maximum time per simulation (seconds)
            mode: "default" for standard simulation, "learned" for TD3 model
            model_path: Path to trained TD3 model (required if mode="learned")
            action_select_interval: How often to select new action in learned mode
            render: Whether to render the simulation (slower but visual)
            seed: Base random seed for reproducibility. Each run uses seed + run_id.
                  If None, uses current time (non-reproducible).
        """
        self.num_runs = num_runs
        self.corridor_width = corridor_width
        self.door_side = door_side
        self.num_people = num_people
        self.people_speed_range = people_speed_range
        self.max_simulation_time = max_simulation_time
        self.mode = mode
        self.model_path = model_path
        self.action_select_interval = action_select_interval
        self.render = render
        self.base_seed = seed
        
        # Results storage
        self.run_results: List[Dict[str, Any]] = []
        self.start_time = None
        self.end_time = None
        
        # Learned mode: load the PPO agent
        self.agent = None
        self.extract_nav_features = None
        if self.mode == "learned":
            self._load_learned_model()
        
    def _load_learned_model(self):
        """Load the trained TD3 model for learned mode."""
        from agents.td3 import TD3
        from learning.train import extract_nav_features
        
        if self.model_path is None:
            # Default path
            self.model_path = os.path.join(
                os.path.dirname(__file__), '..', 'learning', 'checkpoints', 'td3_policy_WorkingOffset2.pt'
            )
        
        print(f"Loading learned model from: {self.model_path}")
        
        # Infer input dimension using a temporary simulation
        tmp_sim = Simulation(
            corridor_width=self.corridor_width,
            door_side=self.door_side,
            num_people=3,
            people_speeds=[random.uniform(0.6, 1.2) for _ in range(10)]
        )
        _ = tmp_sim.step(1/60.0)
        input_dim = int(len(extract_nav_features(tmp_sim)))
        
        # Instantiate TD3 agent and load checkpoint
        self.agent = TD3(
            state_dim=input_dim,
            action_dim=1,  # Single offset value
            lr_actor=1e-3,
            lr_critic=1e-3,
            gamma=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_delay=2,
            max_action=1.0,
            hidden_size=128,
        )
        self.agent.load(self.model_path)
        self.extract_nav_features = extract_nav_features
        print("Model loaded successfully!")

    def _set_seed(self, run_id: int):
        """Set random seeds for reproducibility.
        
        Args:
            run_id: Current run ID (0-indexed). Combined with base_seed for unique per-run seed.
        """
        if self.base_seed is not None:
            seed = self.base_seed + run_id
        else:
            # Use time-based seed if no base seed provided
            seed = int(time.time()) + run_id
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        return seed

    def generate_people_speeds(self) -> List[float]:
        """Generate random people speeds for a single run"""
        return [random.uniform(*self.people_speed_range) for _ in range(self.num_people)]
    
    def run_single_simulation(self, run_id: int) -> Dict[str, Any]:
        """Run a single simulation and return results"""
        # Set seeds for reproducibility BEFORE generating any random values
        seed_used = self._set_seed(run_id)
        
        print(f"  Starting run {run_id + 1}/{self.num_runs} [{self.mode}] (seed={seed_used})...")
        
        # Create simulation with random people speeds
        people_speeds = self.generate_people_speeds()
        sim = Simulation(
            corridor_width=self.corridor_width,
            door_side=self.door_side,
            num_people=self.num_people,
            people_speeds=people_speeds
        )
        
        # Warm-up step to initialize internal state
        _ = sim.step(1/60.0)
        
        # Initialize rendering if enabled
        screen = None
        clock = None
        if self.render:
            import pygame
            pygame.init()
            width, height = 1000, 400
            screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption(f"Monte Carlo Run {run_id + 1}/{self.num_runs} [{self.mode}]")
            clock = pygame.time.Clock()
        
        # Run simulation
        start_time = time.time()
        step_count = 0
        prev_offset = None  # For learned mode
        cancelled = False
        termination_reason = "timeout"  # Default: timeout
        wall_collision = False
        
        while time.time() - start_time < self.max_simulation_time:
            elapsed = time.time() - start_time
            
            # Handle rendering timing and events
            if self.render:
                import pygame
                dt = clock.tick(60) / 1000.0  # Delta time in seconds
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        cancelled = True
                        break
            else:
                dt = 1.0 / 60.0  # Fixed timestep for consistency
            
            if cancelled:
                break
            
            # Apply learned offset if in learned mode
            if self.mode == "learned" and self.agent is not None:
                feat = self.extract_nav_features(sim)
                
                # Select new action based on interval (no exploration noise during evaluation)
                if (step_count % self.action_select_interval == 0) or (prev_offset is None):
                    offset_normalized = self.agent.select_action(feat, add_noise=False)
                    prev_offset = float(offset_normalized[0])
                
                # Scale offset from [-1, 1] to [-π/6, π/6] (±30 degrees)
                max_offset = math.pi / 6
                offset = prev_offset * max_offset
                
                # Apply offset to planner
                if hasattr(sim.robot, 'nav'):
                    sim.robot.nav.agent_offset = offset
            
            state, reward, done = sim.step(dt)
            step_count += 1
            
            # Check for wall collision
            robot_pos = sim.robot.position
            robot_radius = sim.robot.radius
            bounds = sim.corridor_bounds
            
            if (robot_pos[0] - robot_radius <= bounds['x_min'] or
                robot_pos[0] + robot_radius >= bounds['x_max'] or
                robot_pos[1] - robot_radius <= bounds['y_min'] or
                robot_pos[1] + robot_radius >= bounds['y_max']):
                wall_collision = True
                termination_reason = "wall_collision"
                print(f"    ✗ Wall collision at {elapsed:.2f}s")
                break
            
            # Render if enabled
            if self.render and screen is not None:
                import pygame
                screen.fill((255, 255, 255))
                if self.mode == "learned":
                    # Pass state features for visualization
                    feat = self.extract_nav_features(sim) if self.extract_nav_features else None
                    sim.draw_v0(screen, state_input=feat)
                else:
                    sim.draw_v0(screen)
                pygame.display.flip()
            
            if done:
                termination_reason = "goal_reached"
                print(f"    ✓ Goal reached in {elapsed:.2f}s")
                break
        
        # Check if terminated due to timeout
        simulation_time = time.time() - start_time
        if termination_reason == "timeout":
            print(f"    ✗ Timeout after {simulation_time:.2f}s")
        
        # Cleanup rendering
        if self.render:
            import pygame
            pygame.quit()
        
        # Clear agent buffer if in learned mode
        if self.mode == "learned" and self.agent is not None:
            self.agent.buffer.clear()
        
        # Handle cancelled simulation
        if cancelled:
            print(f"    ✗ Run cancelled by user")
            return None
        
        # Get simulation summary
        summary = sim.get_simulation_summary()
        
        # Override goal_reached based on actual termination reason
        if termination_reason != "goal_reached":
            summary['goal_reached'] = False
            summary['time_to_reach_goal'] = None
        
        # Add termination info to summary
        summary['termination_reason'] = termination_reason
        summary['wall_collision'] = wall_collision
        
        # Export data for this run (include mode in filename)
        filename = f"mc_{self.mode}_run_{run_id + 1:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = sim.export_data_to_csv(filename)
        
        # Compile results
        result = {
            'run_id': run_id + 1,
            'mode': self.mode,
            'seed': seed_used,
            'termination_reason': termination_reason,
            'simulation_time': simulation_time,
            'step_count': step_count,
            'people_speeds': people_speeds,
            'csv_path': csv_path,
            'summary': summary
        }
        
        print(f"    Completed: {simulation_time:.2f}s, {step_count} steps, "
              f"{summary['total_collisions']} collisions, "
              f"termination: {termination_reason}")
        
        return result
    
    def run_monte_carlo(self):
        """Run all Monte Carlo simulations"""
        print(f"Starting Monte Carlo simulation with {self.num_runs} runs...")
        print(f"Parameters:")
        print(f"  Mode: {self.mode}")
        if self.mode == "learned":
            print(f"  Model path: {self.model_path}")
            print(f"  Action select interval: {self.action_select_interval}")
        print(f"  Corridor width: {self.corridor_width}m")
        print(f"  Door side: {self.door_side}")
        print(f"  Number of people: {self.num_people}")
        print(f"  People speed range: {self.people_speed_range[0]}-{self.people_speed_range[1]} m/s")
        print(f"  Max simulation time: {self.max_simulation_time}s")
        print(f"  Render: {'enabled' if self.render else 'disabled'}")
        print(f"  Base seed: {self.base_seed if self.base_seed is not None else 'time-based (non-reproducible)'}")
        print("-" * 60)
        
        self.start_time = time.time()
        
        for i in range(self.num_runs):
            try:
                result = self.run_single_simulation(i)
                if result is not None:  # Skip cancelled runs
                    self.run_results.append(result)
                else:
                    print(f"  Skipping cancelled run {i + 1}")
            except Exception as e:
                print(f"  ❌ Run {i + 1} failed: {e}")
                # Continue with next run
                continue
        
        self.end_time = time.time()
        
        print("-" * 60)
        print(f"Monte Carlo simulation completed!")
        print(f"Total time: {self.end_time - self.start_time:.2f}s")
        print(f"Successful runs: {len(self.run_results)}/{self.num_runs}")
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate statistics across all runs"""
        if not self.run_results:
            return {}
        
        # Extract metrics
        simulation_times = [r['simulation_time'] for r in self.run_results]
        step_counts = [r['step_count'] for r in self.run_results]
        goal_reached = [r['summary']['goal_reached'] for r in self.run_results]
        collision_counts = [r['summary']['total_collisions'] for r in self.run_results]
        avg_velocities = [r['summary']['average_velocity'] for r in self.run_results]
        distances_traveled = [r['summary']['total_distance_traveled'] for r in self.run_results]
        
        # Calculate statistics
        stats = {
            'total_runs': len(self.run_results),
            'successful_runs': sum(goal_reached),
            'success_rate': sum(goal_reached) / len(self.run_results) * 100,
            
            'simulation_time': {
                'mean': statistics.mean(simulation_times),
                'median': statistics.median(simulation_times),
                'std': statistics.stdev(simulation_times) if len(simulation_times) > 1 else 0,
                'min': min(simulation_times),
                'max': max(simulation_times)
            },
            
            'collisions': {
                'mean': statistics.mean(collision_counts),
                'median': statistics.median(collision_counts),
                'std': statistics.stdev(collision_counts) if len(collision_counts) > 1 else 0,
                'min': min(collision_counts),
                'max': max(collision_counts),
                'total': sum(collision_counts)
            },
            
            'average_velocity': {
                'mean': statistics.mean(avg_velocities),
                'median': statistics.median(avg_velocities),
                'std': statistics.stdev(avg_velocities) if len(avg_velocities) > 1 else 0,
                'min': min(avg_velocities),
                'max': max(avg_velocities)
            },
            
            'distance_traveled': {
                'mean': statistics.mean(distances_traveled),
                'median': statistics.median(distances_traveled),
                'std': statistics.stdev(distances_traveled) if len(distances_traveled) > 1 else 0,
                'min': min(distances_traveled),
                'max': max(distances_traveled)
            }
        }
        
        return stats
    
    def print_summary(self):
        """Print comprehensive summary of all runs"""
        stats = self.calculate_statistics()
        
        if not stats:
            print("No results to summarize.")
            return
        
        print("\n" + "=" * 80)
        print(f"MONTE CARLO SIMULATION SUMMARY ({self.mode.upper()} MODE)")
        print("=" * 80)
        
        print(f"Total runs: {stats['total_runs']}")
        print(f"Successful runs: {stats['successful_runs']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        
        # Termination reasons breakdown
        term_reasons = [r['summary'].get('termination_reason', 'unknown') for r in self.run_results]
        goal_reached_count = term_reasons.count('goal_reached')
        wall_collision_count = term_reasons.count('wall_collision')
        timeout_count = term_reasons.count('timeout')
        
        print(f"\nTermination Reasons:")
        print(f"  Goal reached: {goal_reached_count}")
        print(f"  Wall collision: {wall_collision_count}")
        print(f"  Timeout: {timeout_count}")
        
        print(f"\nSimulation Time (seconds):")
        print(f"  Mean: {stats['simulation_time']['mean']:.2f}")
        print(f"  Median: {stats['simulation_time']['median']:.2f}")
        print(f"  Std Dev: {stats['simulation_time']['std']:.2f}")
        print(f"  Range: {stats['simulation_time']['min']:.2f} - {stats['simulation_time']['max']:.2f}")
        
        print(f"\nCollisions:")
        print(f"  Mean: {stats['collisions']['mean']:.2f}")
        print(f"  Median: {stats['collisions']['median']:.2f}")
        print(f"  Std Dev: {stats['collisions']['std']:.2f}")
        print(f"  Range: {stats['collisions']['min']} - {stats['collisions']['max']}")
        print(f"  Total: {stats['collisions']['total']}")
        
        print(f"\nAverage Velocity (m/s):")
        print(f"  Mean: {stats['average_velocity']['mean']:.3f}")
        print(f"  Median: {stats['average_velocity']['median']:.3f}")
        print(f"  Std Dev: {stats['average_velocity']['std']:.3f}")
        print(f"  Range: {stats['average_velocity']['min']:.3f} - {stats['average_velocity']['max']:.3f}")
        
        print(f"\nDistance Traveled (meters):")
        print(f"  Mean: {stats['distance_traveled']['mean']:.2f}")
        print(f"  Median: {stats['distance_traveled']['median']:.2f}")
        print(f"  Std Dev: {stats['distance_traveled']['std']:.2f}")
        print(f"  Range: {stats['distance_traveled']['min']:.2f} - {stats['distance_traveled']['max']:.2f}")
        
        # New comparison metrics
        overlap_persons = [r['summary'].get('overlap_time_persons', 0) for r in self.run_results]
        overlap_door = [r['summary'].get('overlap_time_door', 0) for r in self.run_results]
        overlap_both = [r['summary'].get('overlap_time_both', 0) for r in self.run_results]
        min_door_clearances = [r['summary'].get('min_clearance_to_door', -1) for r in self.run_results 
                               if r['summary'].get('min_clearance_to_door', -1) >= 0]
        
        print(f"\nOverlap Time with Persons (seconds):")
        print(f"  Mean: {statistics.mean(overlap_persons):.3f}")
        print(f"  Total: {sum(overlap_persons):.3f}")
        
        print(f"\nOverlap Time in Door Zone (seconds):")
        print(f"  Mean: {statistics.mean(overlap_door):.3f}")
        print(f"  Total: {sum(overlap_door):.3f}")
        
        print(f"\nOverlap Time Both (seconds):")
        print(f"  Mean: {statistics.mean(overlap_both):.3f}")
        print(f"  Total: {sum(overlap_both):.3f}")
        
        if min_door_clearances:
            print(f"\nMin Clearance to Door (meters):")
            print(f"  Mean: {statistics.mean(min_door_clearances):.3f}")
            print(f"  Min: {min(min_door_clearances):.3f}")
        
        print(f"\nIndividual Run Results:")
        print(f"{'Run':<4} {'Time(s)':<10} {'Dist(m)':<10} {'Avg Vel':<10} {'Ovlp Pers':<10} {'Ovlp Door':<10} {'MinDoorClr':<10} {'Termination':<15}")
        print("-" * 95)
        
        for result in self.run_results:
            summary = result['summary']
            time_to_goal = summary.get('time_to_reach_goal') if summary.get('time_to_reach_goal') else result['simulation_time']
            termination = summary.get('termination_reason', 'unknown')
            print(f"{result['run_id']:<4} "
                  f"{time_to_goal:<10.2f} "
                  f"{summary['total_distance_traveled']:<10.2f} "
                  f"{summary['average_velocity']:<10.3f} "
                  f"{summary.get('overlap_time_persons', 0):<10.3f} "
                  f"{summary.get('overlap_time_door', 0):<10.3f} "
                  f"{summary.get('min_clearance_to_door', -1):<10.3f} "
                  f"{termination:<15}")
        
        print("=" * 95)
    
    def export_summary_csv(self, filename: str = None):
        """Export summary statistics to CSV"""
        if not self.run_results:
            print("No results to export.")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"monte_carlo_{self.mode}_summary_{timestamp}.csv"
        
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # Create data directory if it doesn't exist
        data_dir = "simulation_data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        filepath = os.path.join(data_dir, filename)
        
        try:
            import csv
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'run_id', 
                    'mode',
                    'seed',
                    'goal_reached',
                    'termination_reason',
                    'wall_collision',
                    # Key comparison metrics
                    'time_to_goal',
                    'total_distance_traveled',
                    'average_velocity',
                    'overlap_time_persons',
                    'overlap_time_door',
                    'overlap_time_both',
                    'min_clearance_to_door',
                    # Additional info
                    'total_collisions',
                    'step_count',
                    'simulation_time',
                    'people_speeds',
                    'csv_path'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                
                for result in self.run_results:
                    summary = result['summary']
                    writer.writerow({
                        'run_id': result['run_id'],
                        'mode': result.get('mode', self.mode),
                        'seed': result.get('seed', -1),
                        'goal_reached': summary['goal_reached'],
                        'termination_reason': summary.get('termination_reason', 'unknown'),
                        'wall_collision': summary.get('wall_collision', False),
                        # Key comparison metrics
                        'time_to_goal': summary.get('time_to_reach_goal') if summary.get('time_to_reach_goal') else -1,
                        'total_distance_traveled': summary['total_distance_traveled'],
                        'average_velocity': summary['average_velocity'],
                        'overlap_time_persons': summary.get('overlap_time_persons', 0.0),
                        'overlap_time_door': summary.get('overlap_time_door', 0.0),
                        'overlap_time_both': summary.get('overlap_time_both', 0.0),
                        'min_clearance_to_door': summary.get('min_clearance_to_door', -1.0),
                        # Additional info
                        'total_collisions': summary['total_collisions'],
                        'step_count': result['step_count'],
                        'simulation_time': result['simulation_time'],
                        'people_speeds': str(result['people_speeds']),
                        'csv_path': result['csv_path']
                    })
            
            print(f"\nSummary exported to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error exporting summary: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(
        description='Run Monte Carlo simulations for Predictive DWA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  default  - Uses standard simulation with navigator configured in robot.py
  learned  - Uses trained TD3 model to control heading offset

Examples:
  # Run with default navigator
  python monte_carlo.py --runs 10 --mode default

  # Run with learned model (default checkpoint)
  python monte_carlo.py --runs 10 --mode learned

  # Run with learned model (custom checkpoint)
  python monte_carlo.py --runs 10 --mode learned --model path/to/model.pt

  # Run with learned model and custom action interval
  python monte_carlo.py --runs 10 --mode learned --action-select-interval 5

  # Run with visualization enabled (slower)
  python monte_carlo.py --runs 5 --mode learned --render

  # Run with fixed seed for reproducibility (compare methods with same scenarios)
  python monte_carlo.py --runs 10 --mode default --seed 42 --export-summary
  python monte_carlo.py --runs 10 --mode learned --seed 42 --export-summary
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['default', 'learned'], default='default',
                        help='Simulation mode: "default" or "learned" (default: default)')
    
    # Learned mode arguments
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model (for learned mode, default: checkpoints/td3_policy_WorkingOffset2.pt)')
    parser.add_argument('--action-select-interval', type=int, default=1,
                        help='Select a new action every N steps in learned mode (default: 1)')
    
    # Simulation parameters
    parser.add_argument('--runs', type=int, default=10, help='Number of simulation runs (default: 10)')
    parser.add_argument('--corridor-width', type=float, default=4.0, help='Corridor width in meters (default: 4.0)')
    parser.add_argument('--door-side', choices=['left', 'right'], default='right', help='Door side (default: right)')
    parser.add_argument('--num-people', type=int, default=3, help='Number of people (default: 3)')
    parser.add_argument('--people-speed-min', type=float, default=0.6, help='Minimum people speed (default: 0.6)')
    parser.add_argument('--people-speed-max', type=float, default=1.2, help='Maximum people speed (default: 1.2)')
    parser.add_argument('--max-time', type=float, default=30.0, help='Maximum simulation time per run in seconds (default: 30.0)')
    parser.add_argument('--seed', type=int, default=None, 
                        help='Base random seed for reproducibility. Run i uses seed+i. (default: time-based)')
    parser.add_argument('--render', action='store_true', help='Enable visualization (slower)')
    parser.add_argument('--export-summary', action='store_true', help='Export summary statistics to CSV')
    
    args = parser.parse_args()
    
    # Create and run Monte Carlo simulation
    runner = MonteCarloRunner(
        num_runs=args.runs,
        corridor_width=args.corridor_width,
        door_side=args.door_side,
        num_people=args.num_people,
        people_speed_range=(args.people_speed_min, args.people_speed_max),
        max_simulation_time=args.max_time,
        mode=args.mode,
        model_path=args.model,
        action_select_interval=args.action_select_interval,
        render=args.render,
        seed=args.seed
    )
    
    runner.run_monte_carlo()
    runner.print_summary()
    
    if args.export_summary:
        runner.export_summary_csv()


if __name__ == "__main__":
    main()
