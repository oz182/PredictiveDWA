# Monte Carlo Simulation Runner

This script runs multiple simulations without GUI/display for statistical analysis of the Predictive DWA algorithm.

## Usage

### Basic Usage
```bash
python monte_carlo.py
```
Runs 10 simulations with default parameters.

### Custom Parameters
```bash
python monte_carlo.py --runs 50 --max-time 30 --export-summary
```

### All Available Options
```bash
python monte_carlo.py --help
```

## Command Line Arguments

- `--runs N`: Number of simulation runs (default: 10)
- `--corridor-width W`: Corridor width in meters (default: 4.0)
- `--door-side {left,right}`: Door side (default: right)
- `--num-people N`: Number of people (default: 3)
- `--people-speed-min X`: Minimum people speed (default: 0.6)
- `--people-speed-max X`: Maximum people speed (default: 1.2)
- `--max-time T`: Maximum simulation time per run (default: 60.0)
- `--export-summary`: Export summary statistics to CSV

## Output

### Terminal Output
- Progress updates for each run
- Real-time collision detection notifications
- Comprehensive summary statistics
- Individual run results table

### Data Files
- Individual CSV files for each run: `mc_run_XXX_YYYYMMDD_HHMMSS.csv`
- Summary CSV file (if --export-summary): `monte_carlo_summary_YYYYMMDD_HHMMSS.csv`

## Example Output

```
Starting Monte Carlo simulation with 10 runs...
Parameters:
  Corridor width: 4.0m
  Door side: right
  Number of people: 3
  People speed range: 0.6-1.2 m/s
  Max simulation time: 60.0s
------------------------------------------------------------
  Starting run 1/10...
    ✓ Goal reached in 15.23s
    Completed: 15.23s, 914 steps, 2 collisions, Goal reached
  Starting run 2/10...
    Completed: 60.00s, 3600 steps, 8 collisions, Goal not reached
...

================================================================================
MONTE CARLO SIMULATION SUMMARY
================================================================================
Total runs: 10
Successful runs: 7
Success rate: 70.0%

Simulation Time (seconds):
  Mean: 25.45
  Median: 18.32
  Std Dev: 18.23
  Range: 12.15 - 60.00

Collisions:
  Mean: 3.20
  Median: 2.50
  Std Dev: 2.15
  Range: 0 - 8
  Total: 32
...
```

## Data Analysis

Each individual run CSV contains:
- `timestamp`: ISO timestamp
- `elapsed_time`: Time since simulation start
- `robot_x`, `robot_y`: Robot coordinates
- `robot_velocity_x`, `robot_velocity_y`: Velocity components
- `robot_velocity_magnitude`: Speed
- `total_distance_traveled`: Cumulative distance
- `goal_reached`: Boolean goal completion
- `num_people`: Number of people in simulation
- `collision_count`: Cumulative collision count
- `dt`: Time step duration

The summary CSV contains aggregated statistics across all runs for easy analysis.
