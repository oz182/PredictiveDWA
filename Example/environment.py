import random
from obstacle import Obstacle

class Environment:
    """Manages the simulation environment including obstacles and boundaries."""
    
    def __init__(self, playfield_corners=(-4.0, -3.0, 4.0, 3.0), 
                 num_obstacles=20, velocity_range=0.12):
        self.playfield_corners = playfield_corners
        self.velocity_range = velocity_range
        self.obstacles = []
        self.target_index = 0
        
        # Generate initial random obstacles
        for i in range(num_obstacles):
            obstacle = Obstacle.create_random(playfield_corners, velocity_range)
            self.obstacles.append(obstacle)
        
        # Choose random target
        self.target_index = random.randint(0, len(self.obstacles) - 1)
    
    def update(self, dt):
        """Update all obstacles in the environment."""
        for obstacle in self.obstacles:
            obstacle.update(dt, self.playfield_corners)
    
    def add_obstacles(self, count=3):
        """Add new random obstacles to the environment."""
        for i in range(count):
            obstacle = Obstacle.create_random(self.playfield_corners, self.velocity_range)
            self.obstacles.append(obstacle)
    
    def choose_new_target(self):
        """Choose a new random target obstacle."""
        self.target_index = random.randint(0, len(self.obstacles) - 1)
    
    def get_target_position(self):
        """Get the current target obstacle position."""
        return self.obstacles[self.target_index].get_position()
    
    def get_obstacles(self):
        """Get list of all obstacles."""
        return self.obstacles
    
    def get_target_index(self):
        """Get the current target index."""
        return self.target_index
    
    def check_target_reached(self, robot_position, tolerance=0.2):
        """Check if robot has reached the target."""
        target_pos = self.get_target_position()
        distance = ((robot_position[0] - target_pos[0])**2 + 
                   (robot_position[1] - target_pos[1])**2)**0.5
        return distance < tolerance
    
    def print_obstacles(self):
        """Print all obstacle information (for debugging)."""
        for i, obstacle in enumerate(self.obstacles):
            print(f"{i}: {obstacle.x:.3f}, {obstacle.y:.3f}, {obstacle.vx:.3f}, {obstacle.vy:.3f}") 