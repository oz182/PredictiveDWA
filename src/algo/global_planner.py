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

        The path is constrained to lie along the corridor centre-line (y = W/2).
        The *x*-coordinates are linearly interpolated from the robot's start
        position to the goal.
        """
        start = np.array(start, dtype=float)
        goal = np.array(goal, dtype=float)

        # Middle of the hallway (y-coordinate)
        y_mid = self.corridor_width / 2.0

        # Generate samples along the x-axis
        if start[0] <= goal[0]:
            xs = np.arange(start[0], goal[0] + 1e-6, self.resolution)
        else:
            xs = np.arange(start[0], goal[0] - 1e-6, -self.resolution)

        # Assemble way-points (ensure start & goal included)
        waypoints = [np.array([x, y_mid]) for x in xs]
        if not np.allclose(waypoints[0], start):
            waypoints.insert(0, np.array([start[0], y_mid]))
        if not np.allclose(waypoints[-1], goal):
            waypoints.append(np.array([goal[0], y_mid]))

        return waypoints 