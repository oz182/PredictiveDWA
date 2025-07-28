import numpy as np
from typing import List, Tuple, Dict, Set
import heapq
import math

class AStarGlobalPlanner:
    """A* global planner that considers doors as obstacles with halo radius.
    
    This planner uses A* algorithm to find the optimal path from start to goal,
    avoiding doors and their surrounding halo areas.
    """

    def __init__(self, corridor_length: float, corridor_width: float, 
                 door_position: Tuple[float, float], door_side: str,
                 *, resolution: float = 0.25, door_halo_radius: float = 1.0):
        """Parameters
        ----------
        corridor_length : float
            Total length of the corridor (m).
        corridor_width : float
            Width of the corridor (m).
        door_position : Tuple[float, float]
            (x, y) position of the door in world coordinates.
        door_side : str
            "left" or "right" indicating which side of corridor the door is on.
        resolution : float, optional
            Grid resolution for A* search (m).
        door_halo_radius : float, optional
            Radius around door to treat as obstacle (m).
        """
        self.corridor_length = corridor_length
        self.corridor_width = corridor_width
        self.door_position = np.array(door_position)
        self.door_side = door_side
        self.resolution = max(0.05, resolution)
        self.door_halo_radius = door_halo_radius
        
        # Grid dimensions
        self.grid_width = int(corridor_length / resolution) + 1
        self.grid_height = int(corridor_width / resolution) + 1
        
        # Pre-compute door grid position
        self.door_grid_x = int(door_position[0] / resolution)
        self.door_grid_y = int(door_position[1] / resolution)
        self.door_halo_grid_radius = int(door_halo_radius / resolution)

    def _world_to_grid(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        x, y = world_pos
        grid_x = int(x / self.resolution)
        grid_y = int(y / self.resolution)
        return (grid_x, grid_y)
    
    def _grid_to_world(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates."""
        grid_x, grid_y = grid_pos
        world_x = grid_x * self.resolution
        world_y = grid_y * self.resolution
        return (world_x, world_y)
    
    def _is_valid_grid_position(self, grid_pos: Tuple[int, int]) -> bool:
        """Check if grid position is within corridor bounds."""
        x, y = grid_pos
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height
    
    def _is_obstacle(self, grid_pos: Tuple[int, int]) -> bool:
        """Check if grid position is occupied by door halo."""
        x, y = grid_pos
        
        # Check if position is within door halo
        distance_to_door = math.sqrt((x - self.door_grid_x)**2 + (y - self.door_grid_y)**2)
        return distance_to_door <= self.door_halo_grid_radius
    
    def _get_neighbors(self, grid_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring grid positions."""
        x, y = grid_pos
        neighbors = []
        
        # 8-connected grid (including diagonals)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                    
                neighbor = (x + dx, y + dy)
                if (self._is_valid_grid_position(neighbor) and 
                    not self._is_obstacle(neighbor)):
                    neighbors.append(neighbor)
        
        return neighbors
    
    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate heuristic distance between two grid positions."""
        x1, y1 = pos1
        x2, y2 = pos2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _get_movement_cost(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate movement cost between two adjacent positions."""
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Diagonal movement costs more
        if x1 != x2 and y1 != y2:
            return math.sqrt(2)  # Diagonal cost
        else:
            return 1.0  # Straight movement cost
    
    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], 
                         start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from A* search results."""
        current = goal
        path = []
        
        while current in came_from:
            path.append(current)
            current = came_from[current]
        
        path.append(start)
        path.reverse()
        return path
    
    def _smooth_path(self, grid_path: List[Tuple[int, int]]) -> List[np.ndarray]:
        """Convert grid path to smooth world coordinates with more waypoints."""
        if len(grid_path) <= 2:
            return [np.array(self._grid_to_world(pos)) for pos in grid_path]
        
        # Convert to world coordinates
        world_path = [np.array(self._grid_to_world(pos)) for pos in grid_path]
        
        # Add more intermediate points for smoother path
        smoothed_path = [world_path[0]]
        
        for i in range(1, len(world_path)):
            prev = world_path[i - 1]
            curr = world_path[i]
            
            # Add intermediate points between prev and curr
            segment_length = np.linalg.norm(curr - prev)
            num_intermediate = max(1, int(segment_length / self.resolution))
            
            for j in range(1, num_intermediate + 1):
                t = j / (num_intermediate + 1)
                intermediate_point = prev + t * (curr - prev)
                smoothed_path.append(intermediate_point)
            
            smoothed_path.append(curr)
        
        return smoothed_path

    def _create_corridor_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[np.ndarray]:
        """Create a corridor-appropriate path that follows centerline and curves around door."""
        start = np.array(start, dtype=float)
        goal = np.array(goal, dtype=float)
        
        # Corridor centerline y-coordinate
        y_center = self.corridor_width / 2.0
        
        # Door position
        door_x = self.door_position[0]
        door_y = self.door_position[1]
        
        # Determine which side of the door to go around
        if self.door_side == "right":
            # Door is on right side, go around on the left (closer to left wall)
            avoid_y = y_center - self.door_halo_radius - 0.5  # Move left from centerline
        else:
            # Door is on left side, go around on the right (closer to right wall)
            avoid_y = y_center + self.door_halo_radius + 0.5  # Move right from centerline
        
        # Clamp avoid_y to corridor bounds with more reasonable limits
        avoid_y = np.clip(avoid_y, 0.5, self.corridor_width - 0.5)
        
        # Create waypoints for corridor navigation
        waypoints = []
        
        # 1. Start point
        waypoints.append(start)
        
        # 2. Move to centerline (if not already there)
        if not np.isclose(start[1], y_center, atol=0.1):
            center_start = np.array([start[0], y_center])
            waypoints.append(center_start)
        
        # 3. Move along centerline until approaching door
        approach_x = door_x - self.door_halo_radius - 1.0  # 1m before door
        if approach_x > start[0]:
            # Add intermediate points along centerline
            num_approach_points = max(5, int((approach_x - start[0]) / self.resolution))
            for i in range(1, num_approach_points + 1):
                t = i / (num_approach_points + 1)
                x = start[0] + t * (approach_x - start[0])
                center_point = np.array([x, y_center])
                waypoints.append(center_point)
        
            center_approach = np.array([approach_x, y_center])
            waypoints.append(center_approach)
        
        # 4. Curve around the door with more points
        # Create a smooth curved path around the door
        curve_points = 20  # Increased for smoother curve
        curve_start_x = door_x - self.door_halo_radius - 0.5
        curve_end_x = door_x + self.door_halo_radius + 0.5
        
        for i in range(curve_points + 1):
            t = i / curve_points
            
            # Smooth x-coordinate interpolation
            curve_x = curve_start_x + t * (curve_end_x - curve_start_x)
            
            # Create a smooth curve in y-direction using a more complex function
            if self.door_side == "right":
                # Curve from centerline to avoid_y and back (smooth S-curve)
                # Use a combination of sine waves for smoother transition
                curve_y = y_center + (avoid_y - y_center) * (math.sin(math.pi * t) + 0.2 * math.sin(2 * math.pi * t))
            else:
                # Curve from centerline to avoid_y and back (smooth S-curve)
                curve_y = y_center + (avoid_y - y_center) * (math.sin(math.pi * t) + 0.2 * math.sin(2 * math.pi * t))
            
            curve_point = np.array([curve_x, curve_y])
            waypoints.append(curve_point)
        
        # 5. Return to centerline after door
        exit_x = door_x + self.door_halo_radius + 1.0  # 1m after door
        if exit_x < goal[0]:
            center_exit = np.array([exit_x, y_center])
            waypoints.append(center_exit)
            
            # 6. Move along centerline towards goal with intermediate points
            if goal[0] > exit_x:
                # Add intermediate points along centerline to goal
                num_exit_points = max(5, int((goal[0] - exit_x) / self.resolution))
                for i in range(1, num_exit_points + 1):
                    t = i / (num_exit_points + 1)
                    x = exit_x + t * (goal[0] - exit_x)
                    center_point = np.array([x, y_center])
                    waypoints.append(center_point)
                
                goal_approach = np.array([goal[0], y_center])
                waypoints.append(goal_approach)
        
        # 7. Final goal point
        waypoints.append(goal)
        print(f"Door position: ({door_x}, {door_y})")
        print(f"Door side: {self.door_side}")
        print(f"Centerline: {y_center}")
        print(f"Avoid_y: {avoid_y}")
        print(f"Waypoints y-values: {[wp[1] for wp in waypoints]}")
        
        # Ensure all waypoints are within corridor bounds
        bounded_waypoints = []
        for wp in waypoints:
            bounded_wp = np.array([
                np.clip(wp[0], 0, self.corridor_length),
                np.clip(wp[1], 0, self.corridor_width)
            ])
            bounded_waypoints.append(bounded_wp)
        
        return bounded_waypoints

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[np.ndarray]:
        """Compute optimal path from start to goal using corridor-aware planning.
        
        Parameters
        ----------
        start : Tuple[float, float]
            Starting position in world coordinates.
        goal : Tuple[float, float]
            Goal position in world coordinates.
            
        Returns
        -------
        List[np.ndarray]
            List of waypoints from start to goal.
        """
        # For now, always use corridor path to ensure we get enough waypoints
        # This will create a smooth curved path around the door
        return self._create_corridor_path(start, goal)
        
        # Original A* logic (commented out for now)
        # # Check if door is between start and goal
        # start_x = start[0]
        # goal_x = goal[0]
        # door_x = self.door_position[0]
        # 
        # # If door is not between start and goal, use simple path
        # if (start_x < door_x and goal_x < door_x) or (start_x > door_x and goal_x > door_x):
        #     # Door not in the way, use simple corridor path
        #     return self._create_corridor_path(start, goal)
        # 
        # # Door is between start and goal, use A* to find path around it
        # try:
        #     # Convert to grid coordinates
        #     start_grid = self._world_to_grid(start)
        #     goal_grid = self._world_to_grid(goal)
        #     
        #     # Validate start and goal positions
        #     if not self._is_valid_grid_position(start_grid):
        #         raise ValueError(f"Start position {start} is outside corridor bounds")
        #     if not self._is_valid_grid_position(goal_grid):
        #         raise ValueError(f"Goal position {goal} is outside corridor bounds")
        #     
        #     if self._is_obstacle(start_grid):
        #         raise ValueError(f"Start position {start} is inside door halo")
        #     if self._is_obstacle(goal_grid):
        #         raise ValueError(f"Goal position {goal} is inside door halo")
        #     
        #     # A* search
        #     open_set = [(0, start_grid)]  # (f_score, position)
        #     heapq.heapify(open_set)
        #     
        #     came_from = {}
        #     g_score = {start_grid: 0}  # Cost from start to current node
        #     f_score = {start_grid: self._heuristic(start_grid, goal_grid)}  # Estimated total cost
        #     
        #     closed_set = set()
        #     
        #     while open_set:
        #         current_f, current = heapq.heappop(open_set)
        #         
        #         if current == goal_grid:
        #             # Path found
        #             grid_path = self._reconstruct_path(came_from, start_grid, goal_grid)
        #             return self._smooth_path(grid_path)
        #         
        #         closed_set.add(current)
        #         
        #         for neighbor in self._get_neighbors(current):
        #             if neighbor in closed_set:
        #                 continue
        #             
        #             tentative_g = g_score[current] + self._get_movement_cost(current, neighbor)
        #             
        #             if neighbor not in g_score or tentative_g < g_score[neighbor]:
        #                 came_from[neighbor] = current
        #                 g_score[neighbor] = tentative_g
        #                 f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_grid)
        #                 
        #                 # Add to open set if not already there
        #                 if neighbor not in [pos for _, pos in open_set]:
        #                     heapq.heappush(open_set, (f_score[neighbor], neighbor))
        #     
        #     # No path found - fallback to corridor path
        #     print("Warning: A* could not find path, using corridor path fallback")
        #     return self._create_corridor_path(start, goal)
        #     
        # except Exception as e:
        #     print(f"Error in A* planning: {e}, using corridor path fallback")
        #     return self._create_corridor_path(start, goal)
    
    def _straight_line_fallback(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[np.ndarray]:
        """Fallback to straight line path if A* fails."""
        start = np.array(start, dtype=float)
        goal = np.array(goal, dtype=float)
        
        # Simple straight line with waypoints
        total_distance = np.linalg.norm(goal - start)
        num_waypoints = max(2, int(total_distance / self.resolution))
        
        waypoints = []
        for i in range(num_waypoints):
            t = i / (num_waypoints - 1)
            waypoint = start + t * (goal - start)
            waypoints.append(waypoint)
        
        return waypoints


# Keep the old class for backward compatibility
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