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
        self.orientation = 0.0  # Angle in radians (0 points to right)
        
        # DWA parameters
        self.max_rotation = math.pi  # Max angular velocity (rad/s)
        self.max_accel = 1.0  * 4# Max linear acceleration (m/s²)
        self.max_angular_accel = math.pi * 2 # Max angular acceleration (rad/s²)
        self.dt = 0.1  # Time step for simulation
        self.predict_time = 2.0  # How far ahead to predict (seconds)
        self.goal = None
        self.trajectories = []  # For visualization
        self.best_trajectory = None  # For visualization
        
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
    
    def update(self, dt: float, people: List[Person]):
        if self.goal is None:
            return
            
        # Generate dynamic window
        dw = self.dynamic_window()
        
        # Sample velocities and evaluate
        best_score = -float('inf')
        best_v, best_w = 0.0, 0.0
        self.trajectories = []  # Reset for visualization
        
        # Sample linear and angular velocities
        for v in np.linspace(dw[0], dw[1], num=10):
            for w in np.linspace(dw[2], dw[3], num=10):
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
    

