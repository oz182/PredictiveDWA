import math

class Robot:
    """Represents the robot in the simulation."""
    
    def __init__(self, x, y, theta=0.0, radius=0.1):
        self.x = x
        self.y = y
        self.theta = theta
        self.radius = radius
        self.vL = 0.0  # Left wheel velocity
        self.vR = 0.0  # Right wheel velocity
        self.location_history = []  # For displaying trail
        
    def update(self, vL, vR, dt):
        """Update robot position based on wheel velocities."""
        # Store current position in history
        self.location_history.append((self.x, self.y))
        
        # Update velocities
        self.vL = vL
        self.vR = vR
        
        # Update position using differential drive kinematics
        W = 2 * self.radius  # Distance between wheels
        
        # Simple special cases
        if round(vL, 3) == round(vR, 3):
            # Straight line motion
            self.x += vL * dt * math.cos(self.theta)
            self.y += vL * dt * math.sin(self.theta)
        elif round(vL, 3) == -round(vR, 3):
            # Pure rotation
            self.theta += ((vR - vL) * dt / W)
        else:
            # General circular motion
            R = W / 2.0 * (vR + vL) / (vR - vL)
            deltatheta = (vR - vL) * dt / W
            self.x += R * (math.sin(deltatheta + self.theta) - math.sin(self.theta))
            self.y -= R * (math.cos(deltatheta + self.theta) - math.cos(self.theta))
            self.theta += deltatheta
        
        # Normalize angle to [-π, π]
        while self.theta > math.pi:
            self.theta -= 2 * math.pi
        while self.theta < -math.pi:
            self.theta += 2 * math.pi
    
    def get_position(self):
        """Get current position as tuple."""
        return (self.x, self.y)
    
    def get_pose(self):
        """Get current pose as tuple (x, y, theta)."""
        return (self.x, self.y, self.theta)
    
    def distance_to(self, x, y):
        """Calculate distance from robot center to a point."""
        return math.sqrt((self.x - x)**2 + (self.y - y)**2)
    
    def reset_trail(self):
        """Clear the location history."""
        self.location_history = [] 