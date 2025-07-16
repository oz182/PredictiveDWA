import pygame
import time
import math
from robot import Robot
from environment import Environment
from navigation import NavigationAlgorithm

class Simulation:
    """Main simulation class that orchestrates the entire simulation."""
    
    def __init__(self, width=1500, height=1000, dt=0.1):
        # Initialize Pygame
        pygame.init()
        self.width = width
        self.height = height
        self.dt = dt
        self.size = [width, height]
        
        # Graphics constants
        self.k = 160  # pixels per metre
        self.u0 = width / 2   # Screen centre x
        self.v0 = height / 2  # Screen centre y
        
        # Colors
        self.black = (20, 20, 40)
        self.lightblue = (0, 120, 255)
        self.darkblue = (0, 40, 160)
        self.red = (255, 100, 0)
        self.white = (255, 255, 255)
        self.blue = (0, 0, 255)
        self.grey = (70, 70, 70)
        self.green = (0, 200, 0)
        
        # Initialize display
        self.screen = pygame.display.set_mode(self.size)
        pygame.mouse.set_visible(1)
        self.clock = pygame.time.Clock()
        
        # Initialize simulation components
        self.environment = Environment()
        self.robot = Robot(x=-4.5, y=0.0, theta=0.0, radius=0.1)
        self.navigation = NavigationAlgorithm()
        
        # Simulation state
        self.running = True
        self.paths_to_draw = []
        self.new_positions_to_draw = []
        self.best_trajectory_index = 0
    
    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates."""
        u = self.u0 + self.k * x
        v = self.v0 - self.k * y
        return (int(u), int(v))
    
    def draw_obstacles(self):
        """Draw all obstacles on the screen."""
        for i, obstacle in enumerate(self.environment.get_obstacles()):
            if i == self.environment.get_target_index():
                color = self.red
            else:
                color = self.lightblue
            
            screen_pos = self.world_to_screen(obstacle.x, obstacle.y)
            radius_pixels = int(self.k * obstacle.radius)
            pygame.draw.circle(self.screen, color, screen_pos, radius_pixels, 0)
    
    def draw_robot(self):
        """Draw the robot on the screen."""
        # Draw robot body
        screen_pos = self.world_to_screen(self.robot.x, self.robot.y)
        radius_pixels = int(self.k * self.robot.radius)
        pygame.draw.circle(self.screen, self.white, screen_pos, radius_pixels, 3)
        
        # Draw wheels as little blobs
        W = 2 * self.robot.radius
        # Left wheel
        wlx = self.robot.x - (W/2.0) * math.sin(self.robot.theta)
        wly = self.robot.y + (W/2.0) * math.cos(self.robot.theta)
        wheel_pos = self.world_to_screen(wlx, wly)
        wheel_radius = int(self.k * 0.04)
        pygame.draw.circle(self.screen, self.blue, wheel_pos, wheel_radius)
        
        # Right wheel
        wrx = self.robot.x + (W/2.0) * math.sin(self.robot.theta)
        wry = self.robot.y - (W/2.0) * math.cos(self.robot.theta)
        wheel_pos = self.world_to_screen(wrx, wry)
        pygame.draw.circle(self.screen, self.blue, wheel_pos, wheel_radius)
    
    def draw_trail(self):
        """Draw the robot's movement trail."""
        for loc in self.robot.location_history:
            screen_pos = self.world_to_screen(loc[0], loc[1])
            pygame.draw.circle(self.screen, self.grey, screen_pos, 3, 0)
    
    def draw_paths(self):
        """Draw the planned paths."""
        # Add a faint green color for all trajectories
        faint_green = (0, 180, 0)  # Darker green for all trajectories
        bright_green = (0, 255, 0)  # Bright green for the best trajectory
        
        for i, path in enumerate(self.paths_to_draw):
            # Choose color based on whether this is the best trajectory
            color = bright_green if i == self.best_trajectory_index else faint_green
            
            if path[0] == 0:  # Straight line
                straight_path = path[1]
                start_pos = self.world_to_screen(self.robot.x, self.robot.y)
                end_x = self.robot.x + straight_path * math.cos(self.robot.theta)
                end_y = self.robot.y + straight_path * math.sin(self.robot.theta)
                end_pos = self.world_to_screen(end_x, end_y)
                pygame.draw.line(self.screen, color, start_pos, end_pos, 1)
            
            elif path[0] == 1:  # Pure rotation (draw a small circle at robot position)
                # For pure rotation, draw a small circle to indicate the robot is turning in place
                start_pos = self.world_to_screen(self.robot.x, self.robot.y)
                radius_pixels = int(self.k * 0.05)  # Small radius for rotation indicator
                pygame.draw.circle(self.screen, color, start_pos, radius_pixels, 1)
            
            elif path[0] == 2:  # General case: circular arc
                # Draw the circular arc properly
                arc_info = path[1]
                rect_info = arc_info[0]  # (tlx, tly)
                size_info = arc_info[1]  # (Rx, Ry)
                
                # Convert to screen coordinates
                tlx, tly = rect_info
                Rx, Ry = size_info
                
                # Convert world coordinates to screen coordinates
                screen_tlx = int(self.u0 + self.k * tlx)
                screen_tly = int(self.v0 - self.k * tly)
                screen_Rx = int(self.k * Rx)
                screen_Ry = int(self.k * Ry)
                
                # Create pygame rect
                pygame_rect = pygame.Rect(screen_tlx, screen_tly, screen_Rx, screen_Ry)
                
                # Get start and stop angles
                start_angle = path[2]
                stop_angle = path[3]
                
                # Ensure angles are in the right order
                if stop_angle > start_angle:
                    start_angle_final = start_angle
                    stop_angle_final = stop_angle
                else:
                    start_angle_final = stop_angle
                    stop_angle_final = start_angle
                
                # Make angles positive for pygame
                if start_angle_final < 0:
                    start_angle_final += 2 * math.pi
                    stop_angle_final += 2 * math.pi
                
                # Draw the arc if the rect is valid
                if screen_Rx > 0 and screen_Ry > 0 and screen_tlx > 0 and screen_tly > 0:
                    pygame.draw.arc(self.screen, color, pygame_rect, 
                                  start_angle_final, stop_angle_final, 1)
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
    
    def update(self):
        """Update the simulation for one time step."""
        # Get current robot pose
        x, y, theta = self.robot.get_pose()
        vL, vR = self.robot.vL, self.robot.vR
        
        # Plan using navigation algorithm
        vL_new, vR_new, paths, positions, best_index = self.navigation.plan(
            x, y, theta, vL, vR, 
            self.environment.get_obstacles(), 
            self.environment.get_target_index()
        )
        
        # Store paths for drawing
        self.paths_to_draw = paths
        self.new_positions_to_draw = positions
        self.best_trajectory_index = best_index
        
        # Update robot
        self.robot.update(vL_new, vR_new, self.dt)
        
        # Update environment
        self.environment.update(self.dt)
        
        # Check if target reached
        if self.environment.check_target_reached(self.robot.get_position()):
            # Add new obstacles
            self.environment.add_obstacles(3)
            # Choose new target
            self.environment.choose_new_target()
            # Reset robot trail
            self.robot.reset_trail()
    
    def draw(self):
        """Draw everything on the screen."""
        self.screen.fill(self.black)
        
        # Draw trail
        self.draw_trail()
        
        # Draw obstacles
        self.draw_obstacles()
        
        # Draw robot
        self.draw_robot()
        
        # Draw planned paths
        self.draw_paths()
        
        # Update display
        pygame.display.flip()
    
    def run(self):
        """Main simulation loop."""
        while self.running:
            # Handle events
            self.handle_events()
            
            # Update simulation
            self.update()
            
            # Draw everything
            self.draw()
            
            # Control frame rate
            self.clock.tick(60)
            time.sleep(self.dt / 40)
        
        pygame.quit()
    
    def print_obstacles(self):
        """Print obstacle information (for debugging)."""
        self.environment.print_obstacles() 