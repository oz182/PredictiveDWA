import numpy as np
from typing import List, Tuple

class StraightLineGlobalPlanner:
    """Simple global planner that returns a straight-line path along the
    corridor centre-line.

    The planner ignores dynamic obstacles – it merely discretises a straight
    segment between *start* and *goal* at a fixed resolution.  This suffices
    for initial testing of local planners such as TS-DWA.
    """

    def __init__(self, corridor_length: float, corridor_width: float, *, resolution: float = 0.5):
        """Parameters
        ----------
        corridor_length : float
            Total length of the corridor (m).
        corridor_width : float
            Width of the corridor (m).
        resolution : float, optional
            Distance (m) between successive way-points along the path.
        """
        self.corridor_length = corridor_length
        self.corridor_width = corridor_width
        self.resolution = max(0.05, resolution)  # sanity lower-bound

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[np.ndarray]:
        """Compute a straight-line path between *start* and *goal*.

        The path follows a straight line from start to goal, with intermediate
        waypoints along the corridor centerline for smooth navigation.
        """
        start = np.array(start, dtype=float)
        goal = np.array(goal, dtype=float)

        # Middle of the hallway (y-coordinate) for intermediate waypoints
        y_mid = self.corridor_width / 2.0

        # Calculate the total distance to travel
        total_distance = np.linalg.norm(goal - start)
        
        # If start and goal are very close, just return direct path
        if total_distance < self.resolution:
            return [start, goal]

        # Generate intermediate waypoints along the corridor centerline
        # First, move from start to centerline
        start_to_center = np.array([start[0], y_mid])
        
        # Then, move along centerline towards goal
        # Calculate how many waypoints we need
        center_distance = abs(goal[0] - start[0])
        num_waypoints = max(1, int(center_distance / self.resolution))
        
        # Generate waypoints along centerline
        if start[0] <= goal[0]:
            xs = np.linspace(start[0], goal[0], num_waypoints + 1)
        else:
            xs = np.linspace(start[0], goal[0], num_waypoints + 1)
        
        # Assemble the complete path
        waypoints = []
        
        # Add start point
        waypoints.append(start)
        
        # Add intermediate waypoints along centerline (skip if start is already on centerline)
        if not np.isclose(start[1], y_mid, atol=0.1):
            waypoints.append(start_to_center)
        
        # Add waypoints along centerline
        for x in xs[1:-1]:  # Skip first and last to avoid duplicates
            waypoints.append(np.array([x, y_mid]))
        
        # Add goal point (skip if goal is already on centerline)
        if not np.isclose(goal[1], y_mid, atol=0.1):
            goal_from_center = np.array([goal[0], y_mid])
            if len(waypoints) == 0 or not np.allclose(waypoints[-1], goal_from_center):
                waypoints.append(goal_from_center)
        
        # Always add the exact goal point at the end
        waypoints.append(goal)
        
        # Ensure all waypoints are within corridor bounds
        bounded_waypoints = []
        for wp in waypoints:
            bounded_wp = np.array([
                np.clip(wp[0], 0, self.corridor_length),
                np.clip(wp[1], 0, self.corridor_width)
            ])
            bounded_waypoints.append(bounded_wp)
        
        return bounded_waypoints 