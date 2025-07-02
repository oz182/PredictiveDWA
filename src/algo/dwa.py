import pygame
import numpy as np
import random
from typing import List, Tuple
import math

from sim.person import Person


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
        self.max_rotation = math.pi  # Max angular velocity (rad/s)
        self.max_accel = 1.0 * 4  # Max linear acceleration (m/s²)
        self.max_angular_accel = math.pi * 2  # Max angular acceleration (rad/s²)
        self.dt = 0.1  # Time step for simulation
        self.predict_time = 2.0  # How far ahead to predict (seconds)
        self.goal = None
        self.trajectories = []  # For visualization
        self.best_trajectory = None  # For visualization
        
        # Sample space parameters
        self.v_samples = 10  # Number of linear velocity samples
        self.w_samples = 10  # Number of angular velocity samples
        self.v_min = 0.0  # Minimum linear velocity
        self.v_max = max_speed  # Maximum linear velocity
        self.w_min = -math.pi  # Minimum angular velocity
        self.w_max = math.pi  # Maximum angular velocity
        
        # Door-aware sampling parameters
        self.door_position = None  # Will be set when door position is known
        self.door_side = None  # Will be set when door side is known
        self.door_influence_radius = 4.5  # Distance in meters where door affects sampling
        self.door_sampling_bias = 0.6  # How strongly to bias sampling (0-1)
        
        # Scoring weights
        self.weights = {
            'goal': 0.2,
            'clearance': 0.6,
            'velocity': 0.2
        }
        
        # Robot dynamics
        self.v = 0.0  # Current linear velocity
        self.w = 0.0  # Current angular velocity

    def set_goal(self, goal: Tuple[float, float]):
        self.goal = np.array(goal)
    
    def set_sample_space_params(self, v_samples=None, w_samples=None, 
                              v_min=None, v_max=None, w_min=None, w_max=None):
        """Modify the DWA sample space parameters.
        
        Args:
            v_samples (int, optional): Number of linear velocity samples
            w_samples (int, optional): Number of angular velocity samples
            v_min (float, optional): Minimum linear velocity
            v_max (float, optional): Maximum linear velocity
            w_min (float, optional): Minimum angular velocity in radians
            w_max (float, optional): Maximum angular velocity in radians
        """
        if v_samples is not None:
            self.v_samples = max(2, v_samples)  # At least 2 samples
        if w_samples is not None:
            self.w_samples = max(2, w_samples)  # At least 2 samples
        if v_min is not None:
            self.v_min = max(0.0, v_min)  # Can't go backwards
        if v_max is not None:
            self.v_max = min(self.max_speed, v_max)  # Can't exceed max speed
        if w_min is not None:
            self.w_min = max(-math.pi, w_min)  # Limit to -π
        if w_max is not None:
            self.w_max = min(math.pi, w_max)  # Limit to π

    def set_door_info(self, door_position: Tuple[float, float], door_side: str):
        """Set the door position and side for door-aware sampling.
        
        Args:
            door_position: (x,y) position of the door in world coordinates
            door_side: "left" or "right" indicating which side of corridor the door is on
        """
        self.door_position = np.array(door_position)
        self.door_side = door_side

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
        
        # Calculate door-relative angle
        door_direction = self.door_position - self.position
        door_angle = math.atan2(door_direction[1], door_direction[0])
        door_angle = self.normalize_angle(door_angle - self.orientation)

        # print the door angle, orientation and door direction in degrees
        door_angle = math.degrees(door_angle)
        door_direction = np.array([math.cos(math.radians(door_angle)), 
                                   math.sin(math.radians(door_angle))])
        orientation_deg = math.degrees(self.orientation)
        
        # Calculate influence factor (1 when at door, 0 at influence radius)
        # Use distance to door and orientation reletive to the door
        influence = (1.0 - (dist_to_door / self.door_influence_radius)) + (1 - abs(door_angle) / 180.0)
        influence *= self.door_sampling_bias  # Scale by bias factor

        # Determine which way to bias based on door side
        if self.door_side == "right":
            # Bias towards negative angles (left turns) when door is on right
            w_min = -math.pi
            w_max = math.pi -(math.pi + math.pi/2)*influence
            #print(f"w_min: {w_min}, w_max: {w_max}")
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
        
        # Get door-aware sampling parameters
        w_min, w_max = self.get_door_aware_sampling_params()
        
        # Sample velocities and evaluate
        best_score = -float('inf')
        best_v, best_w = 0.0, 0.0
        self.trajectories = []  # Reset for visualization
        
        # Sample linear and angular velocities using configured parameters
        for v in np.linspace(max(dw[0], self.v_min), min(dw[1], self.v_max), num=self.v_samples):
            # Use door-aware angular velocity limits
            for w in np.linspace(max(dw[2], w_min), min(dw[3], w_max), num=self.w_samples):
                trajectory = self.predict_trajectory(v, w)
                self.trajectories.append(trajectory)  # For visualization
                
                # Calculate scores
                goal_score = self.goal_score(trajectory)
                clearance_score = self.clearance_score(trajectory, people)
                velocity_score = v / self.max_speed
                
                # Total weighted score
                total_score = (self.weights['goal'] * goal_score +
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
        vs = [0, self.max_speed,
              -self.max_rotation, self.max_rotation]
        
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
        # Score based on distance to goal
        final_pos = trajectory[-1]
        distance_to_goal = np.linalg.norm(self.goal - final_pos)
        
        # Also consider alignment with goal direction
        goal_direction = self.goal - self.position
        if np.linalg.norm(goal_direction) > 0:
            goal_direction = goal_direction / np.linalg.norm(goal_direction)
        
        traj_direction = trajectory[-1] - trajectory[0]
        if np.linalg.norm(traj_direction) > 0:
            traj_direction = traj_direction / np.linalg.norm(traj_direction)
        
        alignment = np.dot(goal_direction, traj_direction)
        
        # Combine distance and alignment
        max_distance = np.linalg.norm(self.goal - self.position)
        if max_distance == 0:
            return 1.0
            
        distance_score = 1.0 - (distance_to_goal / max_distance)
        return 0.7 * distance_score + 0.3 * alignment
    
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
        
        # Check distance to corridor boundaries if they exist
        if hasattr(self, 'corridor_bounds'):
            bounds = self.corridor_bounds
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
                
                if current_min < min_distance:
                    min_distance = current_min
                    if min_distance <= 0:  # Collision with boundary
                        return -float('inf')
        
        # Normalize to [0, 1] range with 1 being safe distance
        safe_distance = 1.0  # 1 meter is considered safe
        return min(min_distance / safe_distance, 1.0)
    
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


