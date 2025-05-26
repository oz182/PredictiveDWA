import pygame
import numpy as np
import random
from typing import List, Tuple
from sim.person import Person


class Simple:
    def __init__(self, position, velocity, max_speed, goal, radius):
        self.position = position
        self.velocity = velocity
        self.max_speed = max_speed
        self.goal = goal
        self.radius = radius

    def set_goal(self, goal):
        self.goal = goal
    
    def update(self, dt: float, people: List[Person]):
            
        # Simple movement towards goal (will be replaced with DWA later)
        direction = self.goal - self.position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction = direction / distance
            
        # Simple obstacle avoidance (very basic)
        for person in people:
            if person.active:
                to_person = person.position - self.position
                dist_to_person = np.linalg.norm(to_person)
                if dist_to_person < (self.radius + person.radius + 1.0):  # Safety margin
                    # Avoid by moving perpendicular
                    avoid_dir = np.array([-to_person[1], to_person[0]])
                    if np.linalg.norm(avoid_dir) > 0:
                        avoid_dir = avoid_dir / np.linalg.norm(avoid_dir)
                    direction += avoid_dir * 0.5
        
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
            
        self.velocity = direction * self.max_speed
        self.position += self.velocity * dt

        return self.velocity, self.position, self.goal

