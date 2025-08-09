import numpy as np
import math
from typing import List, Tuple
from sim.person import Person

class TSDWA_2:
    """
    Targeted Sampling Dynamic Window Approach (TS-DWA) local planner.
    Implements the path-aware DWA strategy from Shen & Soh (2024):contentReference[oaicite:56]{index=56}:contentReference[oaicite:57]{index=57},
    sampling velocities preferentially along the global path direction.
    """
    def __init__(self, position: np.ndarray, velocity: np.ndarray, max_speed: float, goal: np.ndarray, 
                 radius: float, corridor_bounds: dict, global_plan: List[Tuple[float, float]] = None):
        # Initial state
        self.position = position
        self.velocity = velocity
        self.max_speed = max_speed
        self.goal = goal  # target goal position in world coordinates
        self.radius = radius
        self.corridor_bounds = corridor_bounds
        # Orientation and current velocities
        self.orientation = 0.0  # robot's current heading (radians, 0 points to right):contentReference[oaicite:58]{index=58}
        self.v = 0.0  # current linear velocity (speed)
        self.w = 0.0  # current angular velocity
        # DWA parameters (dynamic limits and time settings)
        self.max_rotation = math.pi           # Max angular velocity (rad/s)
        self.max_accel = 1.0 * 4             # Max linear acceleration (m/s²):contentReference[oaicite:59]{index=59}
        self.max_angular_accel = math.pi * 2  # Max angular acceleration (rad/s²)
        self.dt = 0.1            # Time step for trajectory simulation (s)
        self.predict_time = 2.0  # Prediction horizon (s)
        # TS-DWA sampling strategy parameters
        self.heading_range = math.pi / 3     # Angular half-range around path heading for sampling:contentReference[oaicite:60]{index=60}
        self.n_heading_samples = 5           # Number of heading samples in [θ_ph - range, θ_ph + range]
        self.alpha_ph = 4.0                  # Heading bias factor α_ph (for preferred orientation):contentReference[oaicite:61]{index=61}
        self.i_look = 4                      # Look-ahead index for path heading extraction:contentReference[oaicite:62]{index=62}
        self.n_skip = 3                      # Skip count for curvature point selection:contentReference[oaicite:63]{index=63}:contentReference[oaicite:64]{index=64}
        # Global plan waypoints (if provided)
        self.global_plan = [np.array(p, dtype=float) for p in global_plan] if global_plan is not None else None
        # Visualization data
        self.trajectories = []      # List of trajectories (each a np.array of points) for visualization
        self.best_trajectory = None # Best trajectory from last update cycle
        # Scoring weights for path, goal, and obstacle components
        self.weights = {
            'path': 0.2,
            'goal': 0.3,
            'obstacle': 0.5
        }
    
    def set_goal(self, goal: Tuple[float, float]):
        """Update the goal position."""
        self.goal = np.array(goal, dtype=float)
    
    def normalize_angle(self, angle: float) -> float:
        # Normalize angle to [-π, π]
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def update(self, dt: float, people: List[Person]):
        """
        Compute the optimal velocity command for this time step using TS-DWA.
        Returns (velocity_vector, new_position, goal_position).
        """
        if self.goal is None:
            return  # No goal set
        ## 1. Dynamic window: limit linear speed range based on current v and acceleration:contentReference[oaicite:65]{index=65}
        v_min = max(0.0, self.v - self.max_accel * self.dt)
        v_max = min(self.max_speed, self.v + self.max_accel * self.dt)
        ## 2. Extract path heading θ_ph and curvature κ from the global plan:contentReference[oaicite:66]{index=66}:contentReference[oaicite:67]{index=67}
        theta_ph = 0.0
        kappa = 0.0
        if self.global_plan is not None and len(self.global_plan) > 1:
            # Find nearest waypoint on global plan to current position
            distances = [np.linalg.norm(self.position - wp) for wp in self.global_plan]
            current_index = int(np.argmin(distances))
            # Determine look-ahead and further points indices
            idx1 = min(current_index + self.i_look, len(self.global_plan) - 1)
            idx2 = min(current_index + 2 * self.i_look, len(self.global_plan) - 1)
            p0 = self.position.copy()        # current position
            p1 = self.global_plan[idx1]      # look-ahead waypoint
            p2 = self.global_plan[idx2]      # further waypoint for curvature
            # Compute heading vector to look-ahead point in robot frame:contentReference[oaicite:68]{index=68}
            vec_to_p1 = p1 - p0   # vector from robot to look-ahead (world frame)
            # Rotate this vector into robot's frame (using negative of orientation):contentReference[oaicite:69]{index=69}
            rel_x =  vec_to_p1[0] * math.cos(self.orientation) + vec_to_p1[1] * math.sin(self.orientation)
            rel_y = -vec_to_p1[0] * math.sin(self.orientation) + vec_to_p1[1] * math.cos(self.orientation)
            theta_ph = math.atan2(rel_y, rel_x)  # path heading in robot frame:contentReference[oaicite:70]{index=70}
            # Compute path curvature κ via Menger curvature (using p0, p1, p2):contentReference[oaicite:71]{index=71}:contentReference[oaicite:72]{index=72}
            p1_rel = p1 - p0  # position of p1 relative to p0
            p2_rel = p2 - p0  # position of p2 relative to p0
            cross_prod = p1_rel[0]*p2_rel[1] - p1_rel[1]*p2_rel[0]  # 2A (twice area) via cross product
            area = 0.5 * cross_prod
            a = np.linalg.norm(p1_rel)
            b = np.linalg.norm(p2_rel)
            c = np.linalg.norm(p2_rel - p1_rel)
            if a > 1e-6 and b > 1e-6 and c > 1e-6:
                kappa = abs(4 * area / (a * b * c))  # curvature formula:contentReference[oaicite:73]{index=73}:contentReference[oaicite:74]{index=74}
            else:
                kappa = 0.0
            # Add heading bias for preferred orientation (if any):contentReference[oaicite:75]{index=75}:contentReference[oaicite:76]{index=76}
            kappa += self.alpha_ph * theta_ph
        else:
            # No global plan available – use direct goal direction and zero curvature
            vec_to_goal = self.goal - self.position
            rel_x =  vec_to_goal[0] * math.cos(self.orientation) + vec_to_goal[1] * math.sin(self.orientation)
            rel_y = -vec_to_goal[0] * math.sin(self.orientation) + vec_to_goal[1] * math.cos(self.orientation)
            theta_ph = math.atan2(rel_y, rel_x)
            kappa = 0.0
            kappa += self.alpha_ph * theta_ph  # bias towards facing goal direction
        ## 3. Generate translational velocity samples (polar) around θ_ph:contentReference[oaicite:77]{index=77}:contentReference[oaicite:78]{index=78}
        headings = []
        if self.n_heading_samples > 1:
            for i in range(self.n_heading_samples):
                # Evenly spaced angles from (θ_ph - heading_range) to (θ_ph + heading_range)
                offset = -self.heading_range + i * (2 * self.heading_range / (self.n_heading_samples - 1))
                angle = self.normalize_angle(theta_ph + offset)
                headings.append(angle)
        else:
            headings.append(theta_ph)
        # Include explicit left, right, and reverse direction samples:contentReference[oaicite:79]{index=79}
        extra_dirs = [theta_ph + math.pi/2, theta_ph - math.pi/2, theta_ph + math.pi]
        for angle in extra_dirs:
            angle = self.normalize_angle(angle)
            # avoid duplicates (within a small tolerance)
            if all(abs(self.normalize_angle(angle - h)) > 1e-3 for h in headings):
                headings.append(angle)
        # Prepare linear speed samples between v_min and v_max
        linear_samples = np.linspace(v_min, v_max, num=5)
        ## 4. Evaluate each velocity sample via trajectory simulation and scoring
        best_score = -float('inf')
        best_v_cmd = 0.0
        best_w_cmd = 0.0
        self.trajectories.clear()
        self.best_trajectory = None
        for v_cmd in linear_samples:
            for angle in headings:
                # Compute robot-frame velocity components for this sample
                v_x_r = v_cmd * math.cos(angle)
                v_y_r = v_cmd * math.sin(angle)
                # Compute corresponding angular velocity ω from curvature and bias:contentReference[oaicite:80]{index=80}
                w_cmd = v_cmd * kappa
                # Clamp ω to feasible range
                if w_cmd > self.max_rotation:
                    w_cmd = self.max_rotation
                if w_cmd < -self.max_rotation:
                    w_cmd = -self.max_rotation
                # Roll out trajectory with constant v_x_r, v_y_r, w_cmd over predict_time:contentReference[oaicite:81]{index=81}
                traj_points = [self.position.copy()]
                current_pos = self.position.copy()
                current_theta = self.orientation
                t = 0.0
                while t < self.predict_time:
                    # update orientation
                    current_theta += w_cmd * self.dt
                    current_theta = self.normalize_angle(current_theta)
                    # update position (translate in robot frame and convert to world frame)
                    v_x_w = v_x_r * math.cos(current_theta) - v_y_r * math.sin(current_theta)
                    v_y_w = v_x_r * math.sin(current_theta) + v_y_r * math.cos(current_theta)
                    current_pos[0] += v_x_w * self.dt
                    current_pos[1] += v_y_w * self.dt
                    traj_points.append(current_pos.copy())
                    t += self.dt
                trajectory = np.array(traj_points)
                self.trajectories.append(trajectory)
                # Collision check and scoring
                clear_score = self.clearance_score(trajectory, people)
                if clear_score == -float('inf'):
                    continue  # skip colliding trajectory:contentReference[oaicite:82]{index=82}:contentReference[oaicite:83]{index=83}
                # Distance from end of trajectory to global path (path adherence):contentReference[oaicite:84]{index=84}:contentReference[oaicite:85]{index=85}
                path_score = 0.0
                if self.global_plan is not None and len(self.global_plan) > 0:
                    final_pt = trajectory[-1]
                    # Compute min distance from final_pt to any waypoint in global_plan
                    min_dist = min(np.linalg.norm(final_pt - wp) for wp in self.global_plan)
                    # Normalize to [0,1] (assume 1m as reference for good alignment)
                    path_score = max(0.0, 1.0 - min_dist / 1.0)
                # Distance from end of trajectory to goal relative to start (goal progress):contentReference[oaicite:86]{index=86}
                final_pt = trajectory[-1]
                dist_to_goal = np.linalg.norm(final_pt - self.goal) if self.goal is not None else 0.0
                start_dist = np.linalg.norm(self.position - self.goal) if self.goal is not None else 0.0
                goal_score = 0.0
                if start_dist > 1e-6:
                    goal_score = 1.0 - (dist_to_goal / start_dist)
                else:
                    goal_score = 1.0
                # Combine scores with weights:contentReference[oaicite:87]{index=87}
                total_score = (self.weights['path'] * path_score +
                               self.weights['goal'] * goal_score +
                               self.weights['obstacle'] * clear_score)
                if total_score > best_score:
                    best_score = total_score
                    best_v_cmd = v_cmd
                    best_w_cmd = w_cmd
                    self.best_trajectory = trajectory
        ## 5. Set the chosen command and update robot state
        self.v = best_v_cmd
        self.w = best_w_cmd
        # Update orientation and position over the actual dt time step
        self.orientation = self.normalize_angle(self.orientation + self.w * dt)
        # Compute resulting velocity vector in world frame from v (for omni robot):contentReference[oaicite:88]{index=88}
        self.velocity = np.array([
            self.v * math.cos(self.orientation),
            self.v * math.sin(self.orientation)
        ])
        self.position += self.velocity * dt
        return self.velocity, self.position, self.goal
    
    def clearance_score(self, trajectory: np.ndarray, people: List[Person]) -> float:
        """
        Compute a clearance/obstacle score for a trajectory, considering people and corridor boundaries.
        Returns -inf if a collision occurs, or a value in [0,1] indicating safety (1 = safest).:contentReference[oaicite:89]{index=89}
        """
        if (not people or len(people) == 0) and not hasattr(self, 'corridor_bounds'):
            return 1.0  # no obstacles to consider
        min_dist = float('inf')
        # Dynamic obstacles (people)
        for person in people:
            if not person.active:
                continue
            for point in trajectory:
                dist = np.linalg.norm(point - person.position) - self.radius - person.radius
                if dist < min_dist:
                    min_dist = dist
                    if min_dist <= 0:
                        return -float('inf')  # collision with a person:contentReference[oaicite:90]{index=90}
        # Static obstacles (corridor walls)
        if hasattr(self, 'corridor_bounds') and self.corridor_bounds is not None:
            b = self.corridor_bounds
            for point in trajectory:
                # Distances to each wall:contentReference[oaicite:91]{index=91}
                dist_left   = point[0] - b['x_min'] - self.radius
                dist_right  = b['x_max'] - point[0] - self.radius
                dist_bottom = point[1] - b['y_min'] - self.radius
                dist_top    = b['y_max'] - point[1] - self.radius
                current_min = min(dist_left, dist_right, dist_bottom, dist_top)
                if current_min < min_dist:
                    min_dist = current_min
                    if min_dist <= 0:
                        return -float('inf')  # collision with wall:contentReference[oaicite:92]{index=92}
        # Normalize min_dist to [0,1] with 1.0 as safe distance ≥1m:contentReference[oaicite:93]{index=93}
        if min_dist == float('inf'):
            return 1.0  # no obstacles encountered
        safe_distance = 1.0
        return min(min_dist / safe_distance, 1.0)
