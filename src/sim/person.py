import pygame
import numpy as np
import random
from typing import List, Tuple
import math

class Person:
    """
    Simple corridor pedestrian with a minimal Social-Force-inspired steering.

    Behavior overview
    - Finite-state motion: "entering" -> "turning" -> "moving".
      • entering: step straight into the corridor from the door side
      • turning: pick a corridor heading (left/right along +x or -x)
      • moving: walk along chosen heading

    - Avoidance steering (lightweight):
      A small vector that repels the person from nearby persons, the robot,
      and walls using exponentially decaying radial terms. During "moving",
      only the lateral (orthogonal) component of avoidance is applied to reduce
      oscillations and preserve forward progress.
    """
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
        self.turn_duration = 0.9  # seconds to complete turn
        self.turn_progress = 0
        self.corridor_width = corridor_width
        self.corridor_length = corridor_length

        self.active = True
        self.state = "entering"  # "entering", "turning", "moving"
        self.direction = None
        self.travel_distance = 0
        self.max_distance = random.uniform(3.0, 7.0)  # Distance before disappearing
        self.turn_angle = 0
        self.turn_dist = random.uniform(self.corridor_width * 0.2, self.corridor_width * 0.5)

        # Proxemic footprint (semi-major/minor axes in meters)
        self.proxemic_axes = np.array([radius * 1.5, radius * 0.5], dtype=float)
        self.proxemic_color = (255, 150, 150, 90)  # RGBA for translucent halo
        self.heading_angle = -math.pi / 2 if self.door_side == "right" else math.pi / 2
        
    def update(self, dt: float, people: List["Person"] = None, robot=None, corridor_bounds: dict = None):
        """
            This is a first-order steering model (no explicit acceleration).
            The avoidance vector is deliberately small and short-ranged to keep
            motion stable and visually natural without second-order dynamics.
        """
        if not self.active:
            return

        # ------- Minimal avoidance steering (optional) -------
        # Sum of small repulsive influences (people, robot, walls). The magnitude
        # decays exponentially with distance beyond the combined radii.
        avoidance = np.zeros(2, dtype=float)
        if people is not None:
            for other in people:
                if other is self or not other.active:
                    continue
                diff = self.position - other.position
                dist = np.linalg.norm(diff)
                if dist <= 1e-6:
                    continue
                # Exponential falloff radial repulsion from other persons
                overlap = (self.radius + other.radius) - dist
                if overlap > -0.01:  # within sensible influence range
                    dir_vec = diff / dist
                    strength = np.exp(-(dist - (self.radius + other.radius)) / 0.3)
                    avoidance += dir_vec * strength * 0.3  # scale small for stability
        if robot is not None:
            diff = self.position - robot.position
            dist = np.linalg.norm(diff)
            if dist > 1e-6:
                dir_vec = diff / dist
                strength = np.exp(-(dist - (self.radius + getattr(robot, 'radius', 0.2))) / 0.4)
                # Slightly stronger avoidance from the robot (tunable scale)
                avoidance += dir_vec * strength * 0.8
        if corridor_bounds is not None:
            # Soft pushes from walls if too close (short-range exponential)
            x, y = self.position
            # Bottom (y_min)
            d = y - corridor_bounds['y_min'] - self.radius
            if d < 0.4:
                avoidance += np.array([0.0, 1.0]) * np.exp(-(d) / 0.2) * 0.1
            # Top (y_max)
            d = corridor_bounds['y_max'] - y - self.radius
            if d < 0.4:
                avoidance += np.array([0.0, -1.0]) * np.exp(-(d) / 0.2) * 0.1
            # Left (x_min)
            d = x - corridor_bounds['x_min'] - self.radius
            if d < 0.4:
                avoidance += np.array([1.0, 0.0]) * np.exp(-(d) / 0.2) * 0.1
            # Right (x_max)
            d = corridor_bounds['x_max'] - x - self.radius
            if d < 0.4:
                avoidance += np.array([-1.0, 0.0]) * np.exp(-(d) / 0.2) * 0.1

        if self.state == "entering":
            # Move straight into the corridor, plus avoidance
            if self.door_side == "right":
                step = np.array([0.0, -self.speed]) * dt
                self.position += step + avoidance * dt
                # Check if reached midline
                if self.position[1] <= self.corridor_width - self.turn_dist: #self.corridor_width / 2:
                    self.state = "turning"
                    self.turn_angle = random.choice([math.pi, 0])  # 90° left or right
            else:  # left side
                step = np.array([0.0, self.speed]) * dt
                self.position += step + avoidance * dt
                # Check if reached midline
                if self.position[1] >= self.turn_dist: #self.corridor_width / 2:
                    self.state = "turning"
                    self.turn_angle = random.choice([math.pi/2, -math.pi/2])  # 90° left or right

        elif self.state == "turning":
            # Immediately set new direction (no smooth turning)
            self.direction = np.array([math.cos(self.turn_angle), math.sin(self.turn_angle)])
            self.state = "moving"
            self.heading_angle = math.atan2(self.direction[1], self.direction[0])
            
        elif self.state == "moving":
            # Move in chosen direction with lateral-only avoidance to reduce oscillations
            base = self.direction * self.speed
            # Project avoidance to be orthogonal-biased to base to avoid oscillations
            if np.linalg.norm(base) > 0:
                dir_unit = base / np.linalg.norm(base)
                lateral = avoidance - dir_unit * np.dot(avoidance, dir_unit)
            else:
                lateral = avoidance
            movement = (base + lateral) * dt
            self.position += movement
            self.travel_distance += np.linalg.norm(movement)
            if self.direction is not None:
                self.heading_angle = math.atan2(self.direction[1], self.direction[0])
            
            # Deactivate if gone far enough or left corridor
            if (self.travel_distance >= self.max_distance or
                self.position[0] < -self.radius or 
                self.position[0] > self.corridor_length + self.radius or
                self.position[1] < -self.radius or 
                self.position[1] > self.corridor_width + self.radius):
                self.active = False
    
    def draw(self, screen, scale, offset):
        """Render the person and, when moving, a short heading arrow."""
        pos = (self.position * scale + offset).astype(int)
        color = (255, 0, 0) if self.state == "entering" else (200, 50, 50)  # Red when entering, darker when moving

        #### Draw proxemic ellipse (visual inflation area - out of the costmap square) ####
        # a_pix = max(int(self.proxemic_axes[0] * scale), 1)
        # b_pix = max(int(self.proxemic_axes[1] * scale), 1)
        # halo_surface = pygame.Surface((2 * a_pix, 2 * b_pix), pygame.SRCALPHA)
        # pygame.draw.ellipse(halo_surface, self.proxemic_color, halo_surface.get_rect())
        # angle_deg = -math.degrees(self.heading_angle)
        # if abs(angle_deg) > 1e-2:
        #     halo_surface = pygame.transform.rotate(halo_surface, angle_deg)
        # halo_rect = halo_surface.get_rect(center=(pos[0], pos[1]))
        # screen.blit(halo_surface, halo_rect)

        pygame.draw.circle(screen, color, pos, int(self.radius * scale))
        
        # Draw direction arrow if moving
        if self.state == "moving" and self.direction is not None:
            end_pos = (self.position + self.direction * self.radius * 1.5) * scale + offset
            pygame.draw.line(screen, (255, 255, 0), pos, end_pos.astype(int), 2)