import pygame
import numpy as np
import random
from typing import List, Tuple
import math

from sim.person import Person
from copy import deepcopy


class DWA:
    def __init__(self, position, velocity, max_speed, goal, radius, corridor_bounds):
        self.position = position
        self.velocity = velocity
        self.max_speed = max_speed
        self.goal = goal
        self.radius = radius
        self.corridor_bounds = corridor_bounds

        # DWA
        if np.linalg.norm(velocity) > 1e-3:
            self.orientation = math.atan2(velocity[1], velocity[0])
        else:
            self.orientation = 0.0  # Angle in radians (0 points to right)
        
        # DWA parameters
        self.max_rotation = math.pi / 2  # Max angular velocity (rad/s) - capped to prevent dynamic window trapping
        self.max_accel = 1.0 * 4  # Max linear acceleration (m/s²)
        self.max_angular_accel = math.pi * 2  # Max angular acceleration (rad/s²)
        self.dt = 0.167  # Time step for simulation
        self.predict_time = 12.0 * self.dt  # How far ahead to predict (seconds)
        self.goal = None
        self.trajectories = []  # For visualization
        self.best_trajectory = None  # For visualization
        
        # Sample space parameters
        self.v_samples = 8  # Number of linear velocity samples
        self.w_samples = 12  # Number of angular velocity samples
        self.v_min = 0.0  # Minimum linear velocity
        self.v_max = max_speed  # Maximum linear velocity
        #self.w_min = -math.pi / 2  # Minimum angular velocity
        self.w_min = -math.pi  # Minimum angular velocity
        #self.w_max = math.pi / 2  # Maximum angular velocity
        self.w_max = math.pi  # Maximum angular velocity
        
        # Door-aware sampling parameters
        self.door_position = None  # Will be set when door position is known
        self.door_side = None  # Will be set when door side is known
        self.door_influence_radius = 7.5  # Distance in meters where door affects sampling
        self.door_sampling_bias = 0.8  # How strongly to bias sampling (0-1)
        
        # Scoring weights (classic DWA)
        self.weights = {
            'heading': 0.3,   # Alignment toward goal
            'goal': 0.1,      # Progress toward goal (distance reduction)
            'clearance': 0.5, # Obstacle avoidance
            'velocity': 0.4,  # Forward motion
        }
        
        # Robot dynamics
        self.v = 0.0  # Current linear velocity
        self.w = 0.0  # Current angular velocity
        
        # Door angle in robot frame (set by robot.py)
        self.door_angle_robot_frame = None

        # Wall checking parameters
        self.wall_check_points = 6  # Default value, will be updated dynamically
        
        # Agent-controlled w_max (for RL training)
        # When set, this overrides the default w_max sampling range
        # Value should be in [0, 1] where 1 = full range (w_max = pi), 0 = no turning allowed
        self.agent_w_max = None  # None means use default behavior

    def set_goal(self, goal: Tuple[float, float]):
        self.goal = np.array(goal)

    def set_door_info(self, door_position: Tuple[float, float], door_side: str):
        """Set the door position and side for door-aware sampling.
        
        Args:
            door_position: (x,y) position of the door in world coordinates
            door_side: "left" or "right" indicating which side of corridor the door is on
        """
        self.door_position = np.array(door_position)
        self.door_side = door_side

    def set_door_angle_robot_frame(self, angle_rad: float):
        """Set the door angle in robot reference frame (radians)."""
        self.door_angle_robot_frame = angle_rad

    def get_door_aware_sampling_params(self):
        """Calculate sampling parameters based on proximity to door.
        
        Returns:
            tuple: (w_min, w_max) angular velocity limits biased away from door
        """
        if self.door_position is None or self.door_side is None:
            return self.w_min, self.w_max
            
        # Calculate euclidean distance to door
        dist_to_door = np.linalg.norm(self.position - self.door_position)

        # If far from door, use normal sampling
        if dist_to_door > self.door_influence_radius:
            return self.w_min, self.w_max
        
        # Use door angle in robot frame (set by robot.py)
        if self.door_angle_robot_frame is not None:
            door_angle = math.degrees(self.door_angle_robot_frame)
        else:
            # Fallback: Calculate door-relative angle
            door_direction = self.door_position - self.position
            door_angle_rad = math.atan2(door_direction[1], door_direction[0])
            door_angle_rad = self.normalize_angle(door_angle_rad - self.orientation)
            door_angle = math.degrees(door_angle_rad)

        # Calculate influence factor (1 when at door, 0 at influence radius)
        # Use distance to door and orientation reletive to the door
        influence = (1.0 - (dist_to_door / self.door_influence_radius)) + (1 - (abs(door_angle) / 180.0))
        influence *= self.door_sampling_bias  # Scale by bias factor

        # Function to clamp values to a range
        clamp = lambda n, minn, maxn: max(min(maxn, n), minn)

        # Determine which way to bias based on door side
        if self.door_side == "right":
            # Bias towards negative angles (left turns) when door is on right
            w_min = -math.pi
            #w_max = clamp(math.pi -(math.pi + math.pi/2)*influence, -math.pi, math.pi)
            #w_max = clamp(math.pi * (1 - influence) + math.radians(abs(door_angle)), -math.pi, math.pi)
            #w_max = (math.pi + math.radians(abs(door_angle))) * (1 - influence)
            if math.radians(door_angle) > 0 and math.radians(door_angle) < math.pi:
                #w_max = (math.pi + math.radians(door_angle)) * max(0, (1 - influence))
                w_min = -math.pi/2
                w_max = (math.pi + math.radians(door_angle)) * (1 - influence)
            else:
                w_max = math.pi
            #w_max = (math.pi + math.radians(abs(door_angle))) * max(0, (1 - influence))
            #w_max = math.pi * (1 - influence)
            #print(f"influence: {influence}")
           # print(f"w_min: {w_min}, w_max: {w_max}")
            #print(f"door_angle: {door_angle}")
        else:
            # Bias towards positive angles (right turns) when door is on left
            w_min = -math.pi * (1 - influence)
            w_max = math.pi
            
        return w_min, w_max

    def update(self, dt: float, people: List[Person]):
        if self.goal is None:
            return
            
        # Generate dynamic window
        dw = self.dynamic_window()
        
        # Get angular velocity sampling limits
        # If agent_w_max is set (RL control), use it to scale w_max
        # Otherwise use default full range
        if self.agent_w_max is not None:
            # agent_w_max is in [0, 1], scale to actual w_max
            # 1.0 = full range (pi), 0.0 = no turning (but keep small minimum for stability)
            min_w_max = 0.1  # Minimum w_max to prevent getting stuck
            scaled_w_max = min_w_max + self.agent_w_max * (math.pi - min_w_max)
            w_min, w_max = -scaled_w_max, scaled_w_max
        else:
            # Default: use door-aware sampling or full range
            #w_min, w_max = self.get_door_aware_sampling_params()
            w_min, w_max = -math.pi, math.pi
        
        # Sample velocities and evaluate
        best_score = -float('inf')
        best_v, best_w = 0.0, 0.0
        self.trajectories = []  # Reset for visualization
        
        # Sample velocities like in the reference implementation
        # Create arrays of possible velocity changes (not absolute velocities)
        v_samples = np.linspace(max(dw[0], self.v_min), min(dw[1], self.v_max), num=self.v_samples)
        w_samples = np.linspace(max(dw[2], w_min), min(dw[3], w_max), num=self.w_samples)
        
        for v in v_samples:
            for w in w_samples:
                trajectory = self.predict_trajectory(v, w)
                self.trajectories.append(trajectory)  # For visualization
                
                # Calculate scores
                heading_score = self.heading_score(trajectory, v, w)
                goal_score = self.goal_score(trajectory)
                clearance_score = self.clearance_score(trajectory, people)
                
                # Velocity score: prefer higher speeds, penalize near-zero velocity
                velocity_score = v / self.max_speed if v > 0.1 else -1.0
                
                # Total weighted score
                total_score = (self.weights['heading'] * heading_score +
                              self.weights['goal'] * goal_score +
                              self.weights['clearance'] * clearance_score +
                              self.weights['velocity'] * velocity_score)
                
                if total_score > best_score:
                    best_score = total_score
                    best_v, best_w = v, w
                    self.best_trajectory = trajectory  # For visualization
        
        # Update velocities with constraints
        self.v = best_v
        self.w = best_w
        
        # Update position and orientation
        self.orientation += self.w * dt
        self.orientation = self.normalize_angle(self.orientation)
        
        # Calculate velocity vector
        self.velocity = np.array([
            self.v * math.cos(self.orientation),
            self.v * math.sin(self.orientation)
        ])
        self.position += self.velocity * dt

        return self.velocity, self.position, self.goal
    
    def dynamic_window(self):
        # Calculate the dynamic window based on current velocities and constraints  
        # Velocity limits
        vs = [self.v_min, self.v_max, -self.max_rotation, self.max_rotation]
        
        # Add acceleration constraints
        vd = [
            self.v - self.max_accel * self.dt,
            self.v + self.max_accel * self.dt,
            self.w - self.max_angular_accel * self.dt,
            self.w + self.max_angular_accel * self.dt
        ]
        
        # Combine and clamp to absolute limits
        dw = [
            max(vs[0], vd[0]),  # Min linear velocity
            min(vs[1], vd[1]),  # Max linear velocity
            max(vs[2], vd[2]),  # Min angular velocity
            min(vs[3], vd[3])   # Max angular velocity
        ]
        
        return dw
    
    def predict_trajectory(self, v: float, w: float):
        # Simulate trajectory with given velocities
        time = 0
        trajectory = [self.position.copy()]
        current_pos = self.position.copy()
        current_theta = self.orientation
        
        while time < self.predict_time:
            current_theta += w * self.dt
            current_theta = self.normalize_angle(current_theta)
            
            current_pos[0] += v * math.cos(current_theta) * self.dt
            current_pos[1] += v * math.sin(current_theta) * self.dt
            trajectory.append(current_pos.copy())
            time += self.dt
            
        return np.array(trajectory)
    
    def goal_score(self, trajectory):
        """Score based on progress toward goal (distance reduction over the horizon)."""
        final_pos = trajectory[-1]
        distance_to_goal = np.linalg.norm(self.goal - final_pos)

        previous_distance = np.linalg.norm(self.goal - self.position)
        progress = previous_distance - distance_to_goal

        if previous_distance == 0:
            return 1.0

        return progress / previous_distance

    def heading_score(self, trajectory, v, w):
        """Score based on how well the trajectory heads toward the goal.
        
        Classic DWA uses the angle between current heading and goal direction.
        We also penalize excessive rotation to prevent wrap-around exploits.
        """
        if v < 0.01:  # Penalize near-zero velocity heavily
            return -1.0
        
        # Goal direction from current position
        goal_direction = self.goal - self.position
        desired_theta = math.atan2(goal_direction[1], goal_direction[0])
        
        # Heading after ONE time step (not full predict_time to avoid wrap-around)
        next_theta = self.orientation + w * self.dt
        next_theta = self.normalize_angle(next_theta)
        
        # Angular difference between next heading and goal direction
        angle_diff = abs(self.normalize_angle(desired_theta - next_theta))
        
        # Base score: 1.0 when aligned, 0.0 when perpendicular, negative when opposite
        base_score = 1.0 - (angle_diff / math.pi)
        
        # Penalize high angular velocities to prevent spinning (diminishing returns)
        # Trajectories with |w| > pi/2 get progressively penalized
        rotation_penalty = max(0.0, (abs(w) - math.pi/2) / (math.pi/2))  # 0 to 1 for w from pi/2 to pi
        
        return base_score - 0.3 * rotation_penalty
    
    def clearance_score_v0(self, trajectory, people):
        if not people:
            return 1.0
            
        min_distance = float('inf')
        
        for person in people:
            if not person.active:
                continue
                
            for point in trajectory:
                distance = np.linalg.norm(point - person.position) - self.radius - person.radius
                if distance < min_distance:
                    min_distance = distance
                    if min_distance <= 0:  # Collision
                        return -float('inf')
        
        # Normalize to [0, 1] range with 1 being safe distance
        safe_distance = 1.0  # 1 meter is considered safe
        return min(min_distance / safe_distance, 1.0)

    def clearance_score(self, trajectory, people):
        """Calculate clearance score considering both people and corridor boundaries"""
        if not people and not hasattr(self, 'corridor_bounds'):
            return 1.0  # No obstacles or boundaries to consider
        #people = deepcopy(people)
        #for person in people:
        #    if np.linalg.norm(self.position - person.position) < 4.0:
        #        person.position = np.array([20.0, 20.0])

        min_distance = float('inf')
        
        # Check distance to people
        for person in people:
            if not person.active:
                continue
                
            for point in trajectory:
                distance = np.linalg.norm(point - person.position) - self.radius - person.radius
                if distance < min_distance:
                    min_distance = distance
                    if min_distance <= 0:  # Collision
                        return -float('inf')
        
        # Update wall check points based on trajectory length and corridor width
        corridor_width = self.corridor_bounds['y_max'] - self.corridor_bounds['y_min']
        self.wall_check_points = max(1, int(len(trajectory) / (0.5 * corridor_width)))
        
        # Check distance to corridor boundaries if they exist
        
        if hasattr(self, 'corridor_bounds'):
            bounds = self.corridor_bounds  # Need to take in account the we need to extract the walls from the cost map
            for point in trajectory[:self.wall_check_points]:  # Use dynamic parameter
                # Distance to left wall (x_min)
                dist_left = point[0] - bounds['x_min'] - self.radius
                # Distance to right wall (x_max)
                dist_right = bounds['x_max'] - point[0] - self.radius
                # Distance to bottom wall (y_min)
                dist_bottom = point[1] - bounds['y_min'] - self.radius
                # Distance to top wall (y_max)
                dist_top = bounds['y_max'] - point[1] - self.radius
                
                # Find the minimum distance to any boundary
                current_min = min(dist_left, dist_right, dist_bottom, dist_top)
                
                if current_min < min_distance:
                    min_distance = current_min
                    if min_distance <= 0:  # Collision with boundary
                        return -float('inf')
        
        # Normalize to [0, 1] range with 1 being safe distance
        safe_distance = 1.0  # 1 meter is considered safe
        return min(min_distance / safe_distance, 1.0)

    def clearance_score_v2(self, trajectory, people):
        """Return a clearance score that:
        1. Considers *people* distances for grading (like clearance_score_v0).
        2. Treats *corridor walls* as hard constraints – a trajectory that
           intersects a wall is rejected (-inf) but proximity to the wall is
           **not** penalised.

        This lets the robot skim alongside walls when that is the safest route
        around dynamic obstacles, while still preventing actual collisions.
        """
        # --- 1. collision / scoring against dynamic obstacles (people) --------
        if not people:
            min_dist_people = float('inf')
        else:
            min_dist_people = float('inf')
            for person in people:
                if not person.active:
                    continue
                for p in trajectory:
                    dist = np.linalg.norm(p - person.position) - self.radius - person.radius
                    if dist < min_dist_people:
                        min_dist_people = dist
                        if min_dist_people <= 0:  # collision -> reject
                            return -float('inf')
        # --- 2. hard boundary check for corridor walls -----------------------
        if hasattr(self, 'corridor_bounds') and self.corridor_bounds is not None:
            b = self.corridor_bounds
            for p in trajectory:
                if (p[0] - b['x_min'] - self.radius) <= 0:  # left wall collision
                    return -float('inf')
                if (b['x_max'] - p[0] - self.radius) <= 0:  # right wall collision
                    return -float('inf')
                if (p[1] - b['y_min'] - self.radius) <= 0:  # bottom wall
                    return -float('inf')
                if (b['y_max'] - p[1] - self.radius) <= 0:  # top wall
                    return -float('inf')
        # --- 3. convert people clearance to a [0,1] score --------------------
        if min_dist_people == float('inf'):
            return 1.0  # no people encountered
        safe_distance = 1.0  # 1 m considered comfortable around people
        return min(min_dist_people / safe_distance, 1.0)

    def clearance_score_v3(self, trajectory, people):
        """Calculate wall clearance score that penalizes proximity to walls but allows getting close.
        
        This method treats walls as soft constraints - the robot can get close to walls
        but will prefer trajectories that maintain some distance from walls.
        """
        if not hasattr(self, 'corridor_bounds') or self.corridor_bounds is None:
            return 1.0  # No walls to consider
        
        min_wall_distance = float('inf')
        bounds = self.corridor_bounds
        
        # Check distance to corridor boundaries
        for point in trajectory:
            # Distance to left wall (x_min)
            dist_left = point[0] - bounds['x_min'] - self.radius
            # Distance to right wall (x_max)
            dist_right = bounds['x_max'] - point[0] - self.radius
            # Distance to bottom wall (y_min)
            dist_bottom = point[1] - bounds['y_min'] - self.radius
            # Distance to top wall (y_max)
            dist_top = bounds['y_max'] - point[1] - self.radius
            
            # Find the minimum distance to any boundary
            current_min = min(dist_left, dist_right, dist_bottom, dist_top)
            
            if current_min < min_wall_distance:
                min_wall_distance = current_min
                if min_wall_distance <= 0:  # Collision with boundary
                    return -float('inf')
        
        # Convert to a score: prefer some distance from walls but don't penalize too much
        # 0.3m is considered comfortable distance from walls
        comfortable_distance = 0.5
        if min_wall_distance >= comfortable_distance:
            return 1.0  # Full score when maintaining comfortable distance
        else:
            # Gradual penalty as we get closer to walls
            return min_wall_distance / comfortable_distance
    
    def normalize_angle(self, angle):
        # Normalize angle to [-π, π]
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def set_orientation(self, orientation: float):
        """Set the robot's orientation (in radians)."""
        self.orientation = self.normalize_angle(orientation)


