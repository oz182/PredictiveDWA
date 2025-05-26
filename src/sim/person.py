import pygame
import numpy as np
import random
from typing import List, Tuple
import math

class Person:
    def __init__(self, position: Tuple[float, float], radius: float, speed: float, door_side: str, corridor_width: float, corridor_length: float):
        self.position = np.array(position, dtype=float)
        self.radius = radius
        self.speed = speed
        self.door_side = door_side
        self.active = True
        self.state = "entering"  # "entering", "turning", "moving"
        self.direction = None
        self.target = None
        self.turn_angle = 0
        self.turn_duration = 0.5  # seconds to complete turn
        self.turn_progress = 0
        self.corridor_width = corridor_width
        self.corridor_length = corridor_length

        self.active = True
        self.state = "entering"  # "entering", "turning", "moving"
        self.direction = None
        self.travel_distance = 0
        self.max_distance = random.uniform(3.0, 4.0)  # Distance before disappearing
        self.turn_angle = 0
        self.turn_dist = random.uniform(self.corridor_width / 3, self.corridor_width * (2/3))
        
    def update(self, dt: float):
        if not self.active:
            return
            
        if self.state == "entering":
            # Move straight into the corridor
            if self.door_side == "right":
                self.position[1] -= self.speed * dt  # Move down (into corridor)
                # Check if reached midline
                if self.position[1] <= self.corridor_width - self.turn_dist: #self.corridor_width / 2:
                    self.state = "turning"
                    self.turn_angle = random.choice([math.pi, 0])  # 90° left or right
            else:  # left side
                self.position[1] += self.speed * dt  # Move up (into corridor)
                # Check if reached midline
                if self.position[1] >= self.turn_dist: #self.corridor_width / 2:
                    self.state = "turning"
                    self.turn_angle = random.choice([math.pi/2, -math.pi/2])  # 90° left or right
                    
        elif self.state == "turning":
            # Immediately set new direction (no smooth turning)
            self.direction = np.array([math.cos(self.turn_angle), math.sin(self.turn_angle)])
            self.state = "moving"
            
        elif self.state == "moving":
            # Move in chosen direction
            movement = self.direction * self.speed * dt
            self.position += movement
            self.travel_distance += np.linalg.norm(movement)
            
            # Deactivate if gone far enough or left corridor
            if (self.travel_distance >= self.max_distance or
                self.position[0] < -self.radius or 
                self.position[0] > self.corridor_length + self.radius or
                self.position[1] < -self.radius or 
                self.position[1] > self.corridor_width + self.radius):
                self.active = False
    
    def draw(self, screen, scale, offset):
        pos = (self.position * scale + offset).astype(int)
        color = (255, 0, 0) if self.state == "entering" else (200, 50, 50)  # Red when entering, darker when moving
        pygame.draw.circle(screen, color, pos, int(self.radius * scale))
        
        # Draw direction arrow if moving
        if self.state == "moving" and self.direction is not None:
            end_pos = (self.position + self.direction * self.radius * 1.5) * scale + offset
            pygame.draw.line(screen, (255, 255, 0), pos, end_pos.astype(int), 2)