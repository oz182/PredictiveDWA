#!/usr/bin/env python3
"""
Monte Carlo Simulation Runner for Predictive DWA
Runs multiple simulations without GUI/display and records data for analysis
"""

import sys
import os
import time
import random
import argparse
import statistics
from datetime import datetime
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from sim.sim import Simulation


class MonteCarloRunner:
    def __init__(self, num_runs: int = 10, corridor_width: float = 4.0, 
                 door_side: str = "right", num_people: int = 3,
                 people_speed_range: tuple = (0.6, 1.2), 
                 max_simulation_time: float = 60.0):
        """
        Initialize Monte Carlo runner
        
        Args:
            num_runs: Number of simulation runs to perform
            corridor_width: Width of the corridor
            door_side: Which side the door is on ("left" or "right")
            num_people: Number of people in each simulation
            people_speed_range: Tuple of (min_speed, max_speed) for people
            max_simulation_time: Maximum time per simulation (seconds)
        """
        self.num_runs = num_runs
        self.corridor_width = corridor_width
        self.door_side = door_side
        self.num_people = num_people
        self.people_speed_range = people_speed_range
        self.max_simulation_time = max_simulation_time
        
        # Results storage
        self.run_results: List[Dict[str, Any]] = []
        self.start_time = None
        self.end_time = None
        
    def generate_people_speeds(self) -> List[float]:
        """Generate random people speeds for a single run"""
        return [random.uniform(*self.people_speed_range) for _ in range(self.num_people)]
    
    def run_single_simulation(self, run_id: int) -> Dict[str, Any]:
        """Run a single simulation and return results"""
        print(f"  Starting run {run_id + 1}/{self.num_runs}...")
        
        # Create simulation with random people speeds
        people_speeds = self.generate_people_speeds()
        sim = Simulation(
            corridor_width=self.corridor_width,
            door_side=self.door_side,
            num_people=self.num_people,
            people_speeds=people_speeds
        )
        
        # Run simulation
        start_time = time.time()
        step_count = 0
        dt = 1.0 / 60.0  # Fixed timestep for consistency
        
        while time.time() - start_time < self.max_simulation_time:
            state, reward, done = sim.step(dt)
            step_count += 1
            
            if done:
                print(f"    ✓ Goal reached in {time.time() - start_time:.2f}s")
                break
        
        simulation_time = time.time() - start_time
        
        # Get simulation summary
        summary = sim.get_simulation_summary()
        
        # Export data for this run
        filename = f"mc_run_{run_id + 1:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = sim.export_data_to_csv(filename)
        
        # Compile results
        result = {
            'run_id': run_id + 1,
            'simulation_time': simulation_time,
            'step_count': step_count,
            'people_speeds': people_speeds,
            'csv_path': csv_path,
            'summary': summary
        }
        
        print(f"    Completed: {simulation_time:.2f}s, {step_count} steps, "
              f"{summary['total_collisions']} collisions, "
              f"{'Goal reached' if summary['goal_reached'] else 'Goal not reached'}")
        
        return result
    
    def run_monte_carlo(self):
        """Run all Monte Carlo simulations"""
        print(f"Starting Monte Carlo simulation with {self.num_runs} runs...")
        print(f"Parameters:")
        print(f"  Corridor width: {self.corridor_width}m")
        print(f"  Door side: {self.door_side}")
        print(f"  Number of people: {self.num_people}")
        print(f"  People speed range: {self.people_speed_range[0]}-{self.people_speed_range[1]} m/s")
        print(f"  Max simulation time: {self.max_simulation_time}s")
        print("-" * 60)
        
        self.start_time = time.time()
        
        for i in range(self.num_runs):
            try:
                result = self.run_single_simulation(i)
                self.run_results.append(result)
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
        print("MONTE CARLO SIMULATION SUMMARY")
        print("=" * 80)
        
        print(f"Total runs: {stats['total_runs']}")
        print(f"Successful runs: {stats['successful_runs']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        
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
        
        print(f"\nIndividual Run Results:")
        print(f"{'Run':<4} {'Time(s)':<8} {'Steps':<6} {'Collisions':<10} {'Goal':<5} {'Avg Vel':<8}")
        print("-" * 50)
        
        for result in self.run_results:
            summary = result['summary']
            print(f"{result['run_id']:<4} "
                  f"{result['simulation_time']:<8.2f} "
                  f"{result['step_count']:<6} "
                  f"{summary['total_collisions']:<10} "
                  f"{'Yes' if summary['goal_reached'] else 'No':<5} "
                  f"{summary['average_velocity']:<8.3f}")
        
        print("=" * 80)
    
    def export_summary_csv(self, filename: str = None):
        """Export summary statistics to CSV"""
        if not self.run_results:
            print("No results to export.")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"monte_carlo_summary_{timestamp}.csv"
        
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
                    'run_id', 'simulation_time', 'step_count', 'goal_reached',
                    'total_collisions', 'average_velocity', 'total_distance_traveled',
                    'people_speeds', 'csv_path'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                
                for result in self.run_results:
                    summary = result['summary']
                    writer.writerow({
                        'run_id': result['run_id'],
                        'simulation_time': result['simulation_time'],
                        'step_count': result['step_count'],
                        'goal_reached': summary['goal_reached'],
                        'total_collisions': summary['total_collisions'],
                        'average_velocity': summary['average_velocity'],
                        'total_distance_traveled': summary['total_distance_traveled'],
                        'people_speeds': str(result['people_speeds']),
                        'csv_path': result['csv_path']
                    })
            
            print(f"\nSummary exported to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error exporting summary: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='Run Monte Carlo simulations for Predictive DWA')
    parser.add_argument('--runs', type=int, default=10, help='Number of simulation runs (default: 10)')
    parser.add_argument('--corridor-width', type=float, default=4.0, help='Corridor width in meters (default: 4.0)')
    parser.add_argument('--door-side', choices=['left', 'right'], default='right', help='Door side (default: right)')
    parser.add_argument('--num-people', type=int, default=3, help='Number of people (default: 3)')
    parser.add_argument('--people-speed-min', type=float, default=0.6, help='Minimum people speed (default: 0.6)')
    parser.add_argument('--people-speed-max', type=float, default=1.2, help='Maximum people speed (default: 1.2)')
    parser.add_argument('--max-time', type=float, default=60.0, help='Maximum simulation time per run (default: 60.0)')
    parser.add_argument('--export-summary', action='store_true', help='Export summary statistics to CSV')
    
    args = parser.parse_args()
    
    # Create and run Monte Carlo simulation
    runner = MonteCarloRunner(
        num_runs=args.runs,
        corridor_width=args.corridor_width,
        door_side=args.door_side,
        num_people=args.num_people,
        people_speed_range=(args.people_speed_min, args.people_speed_max),
        max_simulation_time=args.max_time
    )
    
    runner.run_monte_carlo()
    runner.print_summary()
    
    if args.export_summary:
        runner.export_summary_csv()


if __name__ == "__main__":
    main()
