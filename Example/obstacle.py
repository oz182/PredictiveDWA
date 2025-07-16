import random
import math

class Obstacle:
    """Represents a moving obstacle/barrier in the simulation."""
    
    def __init__(self, x, y, vx, vy, radius=0.1):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
    
    def update(self, dt, playfield_corners):
        """Update obstacle position and handle boundary bouncing."""
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Bounce off boundaries
        if self.x < playfield_corners[0]:
            self.vx = -self.vx
        if self.x > playfield_corners[2]:
            self.vx = -self.vx
        if self.y < playfield_corners[1]:
            self.vy = -self.vy
        if self.y > playfield_corners[3]:
            self.vy = -self.vy
    
    def get_position(self):
        """Get current position as tuple."""
        return (self.x, self.y)
    
    def get_velocity(self):
        """Get current velocity as tuple."""
        return (self.vx, self.vy)
    
    def distance_to(self, x, y):
        """Calculate distance from obstacle center to a point."""
        return math.sqrt((self.x - x)**2 + (self.y - y)**2)
    
    @classmethod
    def create_random(cls, playfield_corners, velocity_range=0.15, radius=0.1):
        """Create a random obstacle within the playfield."""
        x = random.uniform(playfield_corners[0], playfield_corners[2])
        y = random.uniform(playfield_corners[1], playfield_corners[3])
        vx = random.gauss(0.0, velocity_range)
        vy = random.gauss(0.0, velocity_range)
        return cls(x, y, vx, vy, radius) 