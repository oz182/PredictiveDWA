import pygame
import numpy as np
import random
from typing import List, Tuple
import math
from copy import deepcopy, copy

from sim.person import Person
from algo.simple import Simple
from algo.dwa import DWA
from algo.ts_dwa import TSDWA
from algo.ts_dwa_Try import TSDWA_2
from algo.global_planner import StraightLineGlobalPlanner
from algo.global_planner import AStarGlobalPlanner


class Robot:
    def __init__(self, position: Tuple[float, float], radius: float, corridor_bounds: dict, door_position: Tuple[float, float]):
        self.position = np.array(position, dtype=float)
        self.radius = radius
        self.velocity = np.array([0.0, 0.0])
        self.max_speed = 2.0
        self.goal = None
        self.corridor_bounds = corridor_bounds
        self.door_position = door_position
        self.people = None

        # ------ Global planner (A* with door avoidance) ------
        corridor_length = self.corridor_bounds['x_max'] - self.corridor_bounds['x_min']
        corridor_width  = self.corridor_bounds['y_max'] - self.corridor_bounds['y_min']

        # Get door information from simulation
        door_pos = self.door_position
        door_side = "right"  # You'll need to pass this from simulation
        if door_pos[1] < corridor_width / 2:
            door_side = "left"

        self.global_planner = AStarGlobalPlanner(
            corridor_length, 
            corridor_width, 
            door_pos, 
            door_side,
            resolution=0.25,
            door_halo_radius=1.0  # 1 meter radius around door
        )
        self.global_path = None  # will be initialised after goal is set

        #self.nav = Simple(self.position, self.velocity, self.max_speed, self.goal, self.radius)
        #self.nav = DWA(self.position, self.velocity, self.max_speed, self.goal, self.radius, self.corridor_bounds)
        self.nav = TSDWA(self.position, self.velocity, self.max_speed, self.goal, self.radius, self.corridor_bounds)
        #self.nav = TSDWA_2(self.position, self.velocity, self.max_speed, self.goal, self.radius, self.corridor_bounds)
        
        #self.nav_type = "dwa"
        self.nav_type = "ts_dwa"

        # Learning
        self.state = None
        self.reward = None
        self.done = False
    
    def set_goal(self, goal: Tuple[float, float]):
        self.goal = np.array(goal)
        self.nav.set_goal(goal)
        # Compute (or recompute) global path whenever a new goal is set
        self.global_path = self.global_planner.plan(tuple(self.position), tuple(goal))
    
    def update(self, dt: float, people: List[Person]):
        self.people = people
        if self.goal is None:
            return
        # Pass global path to TS-DWA; other planners ignore the extra arg
        if self.nav_type == "ts_dwa":
            self.velocity, self.position, self.goal = self.nav.update(dt, people, self.global_path)
        else:
            self.velocity, self.position, self.goal = self.nav.update(dt, people)

        # Get state, reward and done
        nav_info = self.get_navigation_info(2)
        self.state = {"nav_info": nav_info, "costmap": self.get_egocentric_costmap()}
        self.reward = nav_info["closest_obstacle_distance"]
        self.done = np.linalg.norm(self.position - self.goal) < 1.0

        return deepcopy(self.state), self.reward, self.done     

    def get_egocentric_costmap(self, size=4.0, resolution=0.1, inflation_radius=0.2):
        """
        Generate an egocentric costmap centered on the robot.
        
        Args:
            size: Size of the costmap in meters (will be square)
            resolution: Meters per pixel
            inflation_radius: How far obstacles influence the costmap
        
        Returns:
            A 2D numpy array (grayscale 0-255) where:
            - 0 = free space
            - 255 = obstacle
            - Values in between represent inflated obstacle areas
        """
        # Calculate grid dimensions
        grid_size = int(size / resolution)
        center_pixel = grid_size // 2
        costmap = np.zeros((grid_size, grid_size), dtype=np.uint8)
        
        # Get robot's current orientation
        robot_angle = self.nav.orientation
        
        # Transform function from world to robot-centric coordinates
        def world_to_robot(p):
            # Translate to robot center
            translated = p - self.position
            # Rotate to align with robot orientation
            rotated = np.array([
                -translated[0] * np.sin(robot_angle) + translated[1] * np.cos(robot_angle),
                translated[0] * np.cos(robot_angle) + translated[1] * np.sin(robot_angle)
            ])
            # Convert to grid coordinates
            grid_x = int(rotated[0] / resolution + center_pixel)
            grid_y = int(rotated[1] / resolution + center_pixel)
            return grid_x, grid_y
        
        # Add corridor boundaries to costmap
        if hasattr(self, 'corridor_bounds'):
            bounds = self.corridor_bounds
            # Check all four walls
            walls = [
                (bounds['x_min'], bounds['y_min'], bounds['x_max'], bounds['y_min']),  # Bottom
                (bounds['x_min'], bounds['y_max'], bounds['x_max'], bounds['y_max']),  # Top
                (bounds['x_min'], bounds['y_min'], bounds['x_min'], bounds['y_max']),  # Left
                (bounds['x_max'], bounds['y_min'], bounds['x_max'], bounds['y_max'])   # Right
            ]
            
            for wall in walls:
                # Sample points along the wall
                if wall[0] == wall[2]:  # Vertical wall
                    y_values = np.linspace(wall[1], wall[3], num=int(size/resolution)*2)
                    x_values = np.full_like(y_values, wall[0])
                else:  # Horizontal wall
                    x_values = np.linspace(wall[0], wall[2], num=int(size/resolution)*2)
                    y_values = np.full_like(x_values, wall[1])
                
                for x, y in zip(x_values, y_values):
                    grid_x, grid_y = world_to_robot(np.array([x, y]))
                    if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                        # Calculate distance from wall to robot
                        dist = np.linalg.norm([x - self.position[0], y - self.position[1]])
                        if dist <= size/2:
                            # Inflate the obstacle
                            for dx in range(-int(inflation_radius/resolution), int(inflation_radius/resolution)+1):
                                for dy in range(-int(inflation_radius/resolution), int(inflation_radius/resolution)+1):
                                    nx, ny = grid_x + dx, grid_y + dy
                                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                                        # Linear decay with distance
                                        obstacle_dist = np.sqrt(dx**2 + dy**2) * resolution
                                        if obstacle_dist <= inflation_radius:
                                            cost = int(255 * (1 - min(obstacle_dist / inflation_radius, 1)))
                                            costmap[ny, nx] = max(costmap[ny, nx], cost)
        
        # Add people to costmap
        for person in self.people:
            if not person.active:
                continue
                
            # Get person position in robot-centric coordinates
            grid_x, grid_y = world_to_robot(person.position)
            
            # Only process if within costmap bounds
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                # Calculate combined radius
                total_radius = person.radius + self.radius + inflation_radius
                
                # Inflate the person's position
                for dx in range(-int(total_radius/resolution), int(total_radius/resolution)+1):
                    for dy in range(-int(total_radius/resolution), int(total_radius/resolution)+1):
                        nx, ny = grid_x + dx, grid_y + dy
                        if 0 <= nx < grid_size and 0 <= ny < grid_size:
                            # Distance from person center
                            obstacle_dist = np.sqrt(dx**2 + dy**2) * resolution - person.radius - self.radius
                            if obstacle_dist <= inflation_radius:
                                cost = int(255 * (1 - min(max(obstacle_dist, 0) / inflation_radius, 1)))
                                costmap[ny, nx] = max(costmap[ny, nx], cost)
        
        return costmap

    def get_navigation_info(self, lookahead_distance: float = 2.0):
        """
        Returns key navigation information in the robot's egocentric frame.
        
        Args:
            door_position: (x,y) position of the door in world coordinates
            lookahead_distance: Distance along trajectory to find waypoint (meters)
            
        Returns:
            dict containing:
                - waypoint: (x,y) position 6m ahead in robot frame (forward is +x)
                - door_position: (x,y) of door in robot frame
                - linear_velocity: Current forward velocity (m/s)
                - angular_velocity: Current angular velocity (rad/s)
        """
        # Get current velocities
        linear_vel = self.velocity[0]
        angular_vel = self.velocity[1]
        
        # Convert door position to robot frame
        door_world = np.array(self.door_position)
        door_relative = door_world - self.position
        door_robot_frame = np.array([
            door_relative[0] * math.cos(self.nav.orientation) + door_relative[1] * math.sin(self.nav.orientation),
            -door_relative[0] * math.sin(self.nav.orientation) + door_relative[1] * math.cos(self.nav.orientation)
        ])

        # Angle to door in robot frame (radians)
        door_angle_rad = math.atan2(door_robot_frame[1], door_robot_frame[0])
        
        # Find waypoint N meters ahead along best trajectory
        waypoint_robot_frame = np.array([lookahead_distance, 0.0])  # Default straight ahead
        
        if self.nav.best_trajectory is not None and len(self.nav.best_trajectory) > 1:
            # Find point along trajectory that's approximately 6m ahead
            cumulative_dist = 0.0
            prev_point = self.nav.best_trajectory[0]
            
            for i in range(1, len(self.nav.best_trajectory)):
                current_point = self.nav.best_trajectory[i]
                segment_length = np.linalg.norm(current_point - prev_point)
                cumulative_dist += segment_length
                
                if cumulative_dist >= lookahead_distance:
                    # Interpolate to get exact point
                    alpha = (lookahead_distance - (cumulative_dist - segment_length)) / segment_length
                    waypoint_world = prev_point + alpha * (current_point - prev_point)
                    
                    # Convert to robot frame
                    waypoint_relative = waypoint_world - self.position
                    waypoint_robot_frame = np.array([
                        waypoint_relative[0] * math.cos(self.nav.orientation) + waypoint_relative[1] * math.sin(self.nav.orientation),
                        -waypoint_relative[0] * math.sin(self.nav.orientation) + waypoint_relative[1] * math.cos(self.nav.orientation)
                    ])
                    break
                prev_point = current_point
    
        # ----- Closest obstacle distance -----
        min_distance = float('inf')

        # Check people
        for person in self.people:
            if not person.active:
                continue
            dist = np.linalg.norm(person.position - self.position) - person.radius - self.radius
            min_distance = min(min_distance, max(0.0, dist))

        # Check corridor bounds (walls are line segments)
        bounds = self.corridor_bounds
        walls = [
            ((bounds['x_min'], bounds['y_min']), (bounds['x_max'], bounds['y_min'])),  # Bottom
            ((bounds['x_min'], bounds['y_max']), (bounds['x_max'], bounds['y_max'])),  # Top
            ((bounds['x_min'], bounds['y_min']), (bounds['x_min'], bounds['y_max'])),  # Left
            ((bounds['x_max'], bounds['y_min']), (bounds['x_max'], bounds['y_max']))   # Right
        ]

        def point_line_distance(p, a, b):
            """Shortest distance from point p to line segment ab."""
            p, a, b = np.array(p), np.array(a), np.array(b)
            ap = p - a
            ab = b - a
            t = np.dot(ap, ab) / np.dot(ab, ab)
            t = np.clip(t, 0, 1)
            closest = a + t * ab
            return np.linalg.norm(p - closest)

        for wall_start, wall_end in walls:
            dist_to_wall = point_line_distance(self.position, wall_start, wall_end) - self.radius
            min_distance = min(min_distance, max(0.0, dist_to_wall))
        
        return {
            'waypoint': waypoint_robot_frame,
            'door_position': door_robot_frame,
            'door_angle': door_angle_rad,
            'linear_velocity': linear_vel,
            'angular_velocity': angular_vel,
            'closest_obstacle_distance': min_distance
        }
    
    def draw(self, screen, scale, offset):
        if self.nav_type == "simple":
            pos = (self.position * scale + offset).astype(int)
            pygame.draw.circle(screen, (0, 0, 255), pos, int(self.radius * scale))
            # Draw direction indicator
            end_pos = (self.position + self.velocity * 0.5) * scale + offset
            pygame.draw.line(screen, (0, 255, 0), pos, end_pos.astype(int), 2)
        elif self.nav_type in ("dwa", "ts_dwa"):
            # Draw robot
            pos = (self.position * scale + offset).astype(int)
            pygame.draw.circle(screen, (0, 0, 255), pos, int(self.radius * scale))
            
            # Draw orientation
            end_pos = (self.position + np.array([
                math.cos(self.nav.orientation),
                math.sin(self.nav.orientation)
            ]) * self.radius * 1.5) * scale + offset
            pygame.draw.line(screen, (0, 255, 0), pos, end_pos.astype(int), 2)
            
            # Draw global path as green line
            if self.global_path is not None and len(self.global_path) > 1:
                global_path_points = [(p * scale + offset).astype(int) for p in self.global_path]
                pygame.draw.lines(screen, (0, 255, 0), False, global_path_points, 3)  # Green, thick line
                
                # Draw waypoints along the global path
                for i, waypoint in enumerate(self.global_path):
                    wp_pos = (waypoint * scale + offset).astype(int)
                    if i == 0:  # Start point
                        pygame.draw.circle(screen, (0, 255, 0), wp_pos, 5)  # Green circle
                    elif i == len(self.global_path) - 1:  # End point
                        pygame.draw.circle(screen, (255, 0, 0), wp_pos, 5)  # Red circle
                    else:  # Intermediate waypoints
                        pygame.draw.circle(screen, (0, 200, 0), wp_pos, 3)  # Darker green
            
            # Draw all sampled trajectories (faint)
            for traj in self.nav.trajectories:
                points = [(p * scale + offset).astype(int) for p in traj]
                if len(points) > 1:
                    pygame.draw.lines(screen, (200, 200, 255, 50), False, points, 1)
            
            # Draw best trajectory
            if self.nav.best_trajectory is not None:
                points = [(p * scale + offset).astype(int) for p in self.nav.best_trajectory]
                if len(points) > 1:
                    pygame.draw.lines(screen, (0, 255, 255), False, points, 2)

            # Draw costmap (semi-transparent overlay)
            size = 4.0
            costmap = self.get_egocentric_costmap(size=size, resolution=0.1)
            
            # Create a colored version of the costmap (red for obstacles)
            colored_costmap = np.zeros((*costmap.shape, int(size)), dtype=np.uint8)  # Now with 4 channels (RGBA)
            colored_costmap[..., 0] = costmap  # Red channel
            colored_costmap[..., 1] = 0        # Green channel
            colored_costmap[..., 2] = 0        # Blue channel
            colored_costmap[..., 3] = costmap  # Alpha channel (for transparency)
            
            # Create surface from the costmap
            costmap_surface = pygame.surfarray.make_surface(colored_costmap[..., :3])  # Only use RGB channels
            costmap_surface.set_alpha(None)  # Clear any existing alpha
            alpha_surface = pygame.Surface(costmap_surface.get_size(), pygame.SRCALPHA)
            alpha_surface.blit(costmap_surface, (0, 0))
            alpha_surface.fill((255, 0, 0, 128), special_flags=pygame.BLEND_RGBA_MULT)
            
            # Scale to world coordinates (4m x 4m -> pixels)
            costmap_size_pixels = int(size * scale)
            alpha_surface = pygame.transform.scale(alpha_surface, 
                                                (costmap_size_pixels, costmap_size_pixels))
            
            # Rotate to match robot orientation
            angle_deg = -math.degrees(self.nav.orientation)
            alpha_surface = pygame.transform.rotate(alpha_surface, angle_deg)
            
            # Calculate position (center the costmap on the robot)
            costmap_rect = alpha_surface.get_rect(center=pos)
            
            # Draw the costmap
            screen.blit(alpha_surface, costmap_rect)

            # Visualize navigation info with arrow indicators
            if hasattr(self, 'corridor_bounds'):
                nav_info = self.get_navigation_info(2)
                
                # Convert robot-frame positions to world coordinates for drawing
                def robot_to_world(robot_pos):
                    world_x = (self.position[0] + 
                            robot_pos[0] * math.cos(self.nav.orientation) - 
                            robot_pos[1] * math.sin(self.nav.orientation))
                    world_y = (self.position[1] + 
                            robot_pos[0] * math.sin(self.nav.orientation) + 
                            robot_pos[1] * math.cos(self.nav.orientation))
                    return np.array([world_x, world_y])
                
                # Draw waypoint (6m ahead)
                if 'waypoint' in nav_info:
                    waypoint_world = robot_to_world(nav_info['waypoint'])
                    waypoint_screen = (waypoint_world * scale + offset).astype(int)
                    pygame.draw.circle(screen, (255, 255, 0), waypoint_screen, 8)  # Yellow
                
                # Draw door position
                if 'door_position' in nav_info:
                    door_world = robot_to_world(nav_info['door_position'])
                    door_screen = (door_world * scale + offset).astype(int)
                    pygame.draw.circle(screen, (0, 255, 0), door_screen, 6)  # Green
                
                # Draw velocity arrows
                arrow_length = 1  # Base length in pixels
                head_angle = math.pi/6  # 30 degrees
                head_length = 10  # pixels
                
                # Linear velocity arrow (blue)
                lin_vel = nav_info['linear_velocity']
                # Calculate end point in direction of orientation
                end_point = self.position + np.array([
                    math.cos(self.nav.orientation),
                    math.sin(self.nav.orientation)
                ]) * lin_vel * arrow_length / self.max_speed
                
                start_screen = pos
                end_screen = (end_point * scale + offset).astype(int)
                
                # Draw arrow line
                pygame.draw.line(screen, (0, 0, 255), start_screen, end_screen, 2)
                
                # Draw arrow head
                angle = math.atan2(end_screen[1]-start_screen[1], end_screen[0]-start_screen[0])
                points = [
                    end_screen,
                    (end_screen[0] - head_length * math.cos(angle - head_angle), end_screen[1] - head_length * math.sin(angle - head_angle)),
                    (end_screen[0] - head_length * math.cos(angle + head_angle), end_screen[1] - head_length * math.sin(angle + head_angle))
                ]
                pygame.draw.polygon(screen, (0, 0, 255), points)
                
                # # Angular velocity arrow (red)
                # ang_vel = nav_info['angular_velocity']
                # # Position arrow perpendicular to robot orientation
                # radius = 40  # pixels from center
                # angle_offset = math.pi/2 if ang_vel > 0 else -math.pi/2
                # start_point = self.position + np.array([
                #     math.cos(self.nav.orientation + angle_offset),
                #     math.sin(self.nav.orientation + angle_offset)
                # ]) * radius / scale
                
                # # Calculate arrow direction (tangential)
                # arrow_dir = np.array([
                #     -math.sin(self.nav.orientation + angle_offset),
                #     math.cos(self.nav.orientation + angle_offset)
                # ]) * math.copysign(1, ang_vel)
                
                # end_point = start_point + arrow_dir * abs(ang_vel) * arrow_length / self.nav.max_rotation
                
                # start_screen = (start_point * scale + offset).astype(int)
                # end_screen = (end_point * scale + offset).astype(int)
                
                # # Draw arrow line
                # pygame.draw.line(screen, (255, 0, 0), start_screen, end_screen, 2)
                
                # # Draw arrow head
                # angle = math.atan2(end_screen[1]-start_screen[1], end_screen[0]-start_screen[0])
                # points = [end_screen,
                #     (end_screen[0] - head_length * math.cos(angle - head_angle), end_screen[1] - head_length * math.sin(angle - head_angle)),
                #     (end_screen[0] - head_length * math.cos(angle + head_angle), end_screen[1] - head_length * math.sin(angle + head_angle))
                # ]
                # pygame.draw.polygon(screen, (255, 0, 0), points)

                # Constants
                if lin_vel > 0:
                    ang_vel = nav_info['angular_velocity']
                    radius = 30  # Distance from robot center (pixels)
                    arrow_length = 25 * (1 + ang_vel)  # Base length (scaled by ang_vel)
                    head_length = 8
                    head_angle = math.pi/6  # 30 degrees


                    lin_vel_dir = np.array([math.cos(self.nav.orientation), 
                                        math.sin(self.nav.orientation)])

                    # Compute perpendicular direction (90° rotation)
                    perp_dir = np.array([-lin_vel_dir[1], lin_vel_dir[0]])  # Rotate 90° CCW

                    # Position arrow at a fixed radius from robot center
                    start_point = self.position# + perp_dir * radius / scale

                    # Arrow direction depends on sign of angular velocity
                    end_point = start_point + perp_dir * np.sign(ang_vel) * arrow_length * abs(ang_vel) / self.nav.max_rotation

                    # Convert to screen coordinates
                    start_screen = (start_point * scale + offset).astype(int)
                    end_screen = (end_point * scale + offset).astype(int)

                    # Draw arrow shaft
                    pygame.draw.line(screen, (255, 0, 0), start_screen, end_screen, 2)

                    # Draw arrowhead
                    angle = math.atan2(end_screen[1] - start_screen[1], end_screen[0] - start_screen[0])
                    points = [
                        end_screen,
                        (end_screen[0] - head_length * math.cos(angle - head_angle),
                        end_screen[1] - head_length * math.sin(angle - head_angle)),
                        (end_screen[0] - head_length * math.cos(angle + head_angle),
                        end_screen[1] - head_length * math.sin(angle + head_angle))
                    ]
                    pygame.draw.polygon(screen, (255, 0, 0), points)

                    #closest distance
                    #print(nav_info['closest_obstacle_distance'])
                                