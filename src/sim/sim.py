import pygame
import numpy as np
import random
import csv
import os
import math
from datetime import datetime
from typing import List, Tuple

from sim.person import Person
from sim.robot import Robot


class Simulation:
    def __init__(self, corridor_width: float = 4.0, door_side: str = None, 
                 num_people: int = 5, people_speeds: List[float] = None,
                 door_halo_radius: float = None, door_position_x: float = None):
        """
        Initialize simulation with optional curriculum learning parameters.
        
        Args:
            corridor_width: Width of the corridor (m)
            door_side: "left" or "right", or None to randomize
            num_people: Number of people to spawn
            people_speeds: List of speeds for people, or None to randomize
            door_halo_radius: Inflation radius around door (m), or None to randomize
            door_position_x: X position of door along corridor (m), or None to randomize
        """
        self.corridor_width = corridor_width
        self.num_people = num_people
        self.people_speeds = people_speeds if people_speeds else [random.uniform(0.6, 1.2) for _ in range(num_people)]
        
        # Corridor dimensions
        self.corridor_length = 14.0
        
        # Randomize door position if not specified (between 20% and 70% of corridor length)
        if door_position_x is None:
            self.door_position = random.uniform(0.30 * self.corridor_length, 0.65 * self.corridor_length)
        else:
            self.door_position = door_position_x
            
        # Randomize door side if not specified
        if door_side is None:
            self.door_side = random.choice(["left", "right"])
        else:
            self.door_side = door_side
            
        # Randomize door halo radius if not specified (default range: 0.8m to 2.5m)
        if door_halo_radius is None:
            self.door_halo_radius = random.uniform(0.8, 2.5)
        else:
            self.door_halo_radius = door_halo_radius
        
        self.corridor_bounds = {
            'x_min': 0,
            'x_max': self.corridor_length,
            'y_min': 0,
            'y_max': self.corridor_width
        }
        
        # Initialize agents
        self.robot = Robot((0.5, corridor_width/1.25), 0.2, self.corridor_bounds, 
                          self.get_door_position(), door_halo_radius=self.door_halo_radius)
        self.robot.set_goal((self.corridor_length - 1.0, corridor_width/1.2))
        
        # Set door information for DWA
        if hasattr(self.robot.nav, 'set_door_info'):
            self.robot.nav.set_door_info(self.get_door_position(), self.door_side)
        
        self.people: List[Person] = []
        self.spawn_timer = 1.0
        self.spawn_interval = 1.0  # Spawn a person every second
        
        # For visualization
        self.scale = 40  # pixels per meter
        self.offset = np.array([50, 50])

        # For learning
        self.done = False
        
        # Data recording
        self.data_recording_enabled = True
        self.simulation_data = []
        self.start_time = None
        self.goal_reached_time = None
        self.total_distance_traveled = 0.0
        self.previous_position = None
        self.collision_count = 0
        self.collision_history = []  # Track collision timestamps and details
        
        # New metrics for comparison
        self.overlap_time_persons = 0.0  # Time spent overlapping with persons (within threshold)
        self.overlap_time_door = 0.0     # Time spent in door inflation zone
        self.overlap_time_both = 0.0     # Time spent overlapping with both
        self.min_clearance_to_door = float('inf')  # Minimum clearance to door position
        self.person_overlap_threshold = 1.0  # Distance threshold for person overlap (m)

    def get_door_position(self) -> Tuple[float, float]:
        """Returns the precise (x,y) world coordinates of the door"""
        door_x = self.door_position
        if self.door_side == "right":
            door_y = self.corridor_width - 0.5  # 0.5m from right wall
        else:
            door_y = 0.5  # 0.5m from left wall
        return (door_x, door_y)
    
    def spawn_person_with_target(self):
        if len(self.people) >= self.num_people:
            return
            
        door_x = self.door_position
        if self.door_side == "right":
            door_y = self.corridor_width - 0.5
            target = (door_x, -1.0)  # Move down out of corridor
        else:
            door_y = 0.5
            target = (door_x, self.corridor_width + 1.0)  # Move up out of corridor
            
        speed = self.people_speeds[len(self.people)]
        self.people.append(Person((door_x, door_y), 0.3, speed, target))

    def spawn_person(self):
        if len(self.people) >= self.num_people:
            return
            
        door_x = self.door_position
        if self.door_side == "right":
            door_y = self.corridor_width - 0.5
        else:
            door_y = 0.5
            
        speed = self.people_speeds[len(self.people)]
        person = Person((door_x, door_y), 0.3, speed, self.door_side, self.corridor_width, self.corridor_length)
        self.people.append(person)
    
    def step(self, dt: float):
        # Initialize start time on first step
        if self.start_time is None:
            self.start_time = datetime.now()
            self.previous_position = self.robot.position.copy()
        
        # Spawn people
        self.spawn_timer += dt
        if self.spawn_timer >= self.spawn_interval and len(self.people) < self.num_people:
            self.spawn_person()
            self.spawn_timer = 0
            self.spawn_interval = random.uniform(0.5, 2.0)
            
        # Update agents
        state, reward, done = self.robot.update(dt, self.people)
        for person in self.people:
            person.update(dt, self.people, self.robot, self.corridor_bounds)
        
        # Remove inactive people
        self.people = [p for p in self.people if p.active]
        
        # Check for collisions and record data if enabled
        if self.data_recording_enabled:
            self._check_collisions()
            self._record_simulation_data(dt, done)

        return state, reward, done
    
    def draw(self, screen):
        # Draw corridor
        corridor_rect = pygame.Rect(
            self.offset[0],
            self.offset[1],
            int(self.corridor_length * self.scale),
            int(self.corridor_width * self.scale)
        )
        pygame.draw.rect(screen, (200, 200, 200), corridor_rect, 1)
        
        # Draw door
        door_pos = int(self.door_position * self.scale) + self.offset[0]
        if self.door_side == "right":
            door_y = int(self.corridor_width * self.scale) + self.offset[1] - 10
            pygame.draw.line(screen, (0, 255, 0), (door_pos, door_y), (door_pos, door_y + 10), 3)
        else:
            door_y = self.offset[1]
            pygame.draw.line(screen, (0, 255, 0), (door_pos, door_y), (door_pos, door_y + 10), 3)
        
        # Draw agents
        for person in self.people:
            person.draw(screen, self.scale, self.offset)
        self.robot.draw(screen, self.scale, self.offset)

    def draw_v0(self, screen, state_input=None):
        # Different from 'draw' function: Print the number of people, robot's speed, and robot's position
        # on the screen

        # Draw corridor
        corridor_rect = pygame.Rect(
            self.offset[0],
            self.offset[1],
            int(self.corridor_length * self.scale),
            int(self.corridor_width * self.scale)
        )
        pygame.draw.rect(screen, (200, 200, 200), corridor_rect, 1)
        
        # Draw door
        door_pos = int(self.door_position * self.scale) + self.offset[0]
        if self.door_side == "right":
            door_y = int(self.corridor_width * self.scale) + self.offset[1] - 10
            pygame.draw.line(screen, (0, 255, 0), (door_pos, door_y), (door_pos, door_y + 10), 3)
        else:
            door_y = self.offset[1]
            pygame.draw.line(screen, (0, 255, 0), (door_pos, door_y), (door_pos, door_y + 10), 3)
        
        # Draw goal if set
        if self.robot.goal is not None:
            goal_pos = (self.robot.goal * self.scale + self.offset).astype(int)
            pygame.draw.circle(screen, (255, 215, 0), goal_pos, 8)  # Gold color
        
        # Draw people
        for person in self.people:
            person.draw(screen, self.scale, self.offset)
        
        # Draw robot (with trajectories)
        self.robot.draw(screen, self.scale, self.offset)
        
        # Display info
        font = pygame.font.SysFont(None, 24)
        info_text = [
            f"Position: ({self.robot.position[0]:.1f}, {self.robot.position[1]:.1f})",
        ]

        # Optional: display state_input feature vector for debugging.
        # Assumes layout from learning.extract_nav_features:
        # [goal_dx, goal_dy, door_dx, door_dy,
        #  p1_dx, p1_dy, p2_dx, p2_dy, p3_dx, p3_dy,
        #  dist_left, dist_right]
        if state_input is not None:
            feat = np.asarray(state_input, dtype=float).tolist()
            if len(feat) >= 12:
                gdx, gdy = feat[0], feat[1]
                door_dx, door_dy = feat[2], feat[3]
                p1_dx, p1_dy = feat[4], feat[5]
                p2_dx, p2_dy = feat[6], feat[7]
                p3_dx, p3_dy = feat[8], feat[9]
                dist_left = feat[10]
                dist_right = feat[11]

                info_text.extend([
                    f"goal_rel:   ({gdx:5.2f}, {gdy:5.2f})",
                    f"door_rel:   ({door_dx:5.2f}, {door_dy:5.2f})",
                    f"p1_rel:     ({p1_dx:5.2f}, {p1_dy:5.2f})",
                    f"p2_rel:     ({p2_dx:5.2f}, {p2_dy:5.2f})",
                    f"p3_rel:     ({p3_dx:5.2f}, {p3_dy:5.2f})",
                    f"dist_left:  {dist_left:5.2f} m",
                    f"dist_right: {dist_right:5.2f} m",
                ])
            else:
                info_text.append(f"state_input (len={len(feat)}): {feat}")
        
        for i, text in enumerate(info_text):
            text_surface = font.render(text, True, (0, 0, 0))
            screen.blit(text_surface, (10, 10 + i * 25))

    def draw_v1(self, screen):
        # Different from 'draw' function: Print the number of people, robot's speed, and robot's position
        # on the screen

        # Draw corridor
        corridor_rect = pygame.Rect(
            self.offset[0],
            self.offset[1],
            int(self.corridor_length * self.scale),
            int(self.corridor_width * self.scale)
        )
        pygame.draw.rect(screen, (200, 200, 200), corridor_rect, 1)
        
        # Draw door
        door_pos = int(self.door_position * self.scale) + self.offset[0]
        if self.door_side == "right":
            door_y = int(self.corridor_width * self.scale) + self.offset[1] - 10
            pygame.draw.line(screen, (0, 255, 0), (door_pos, door_y), (door_pos, door_y + 10), 3)
        else:
            door_y = self.offset[1]
            pygame.draw.line(screen, (0, 255, 0), (door_pos, door_y), (door_pos, door_y + 10), 3)
        
        # Draw goal if set
        if self.robot.goal is not None:
            goal_pos = (self.robot.goal * self.scale + self.offset).astype(int)
            pygame.draw.circle(screen, (255, 215, 0), goal_pos, 8)  # Gold color
        
        # Draw people
        for person in self.people:
            person.draw(screen, self.scale, self.offset)
        
        # Draw robot (with trajectories)
        self.robot.draw(screen, self.scale, self.offset)
        
        # Display info
        font = pygame.font.SysFont(None, 24)
        info_text = [
            #f"People: {len(self.people)}/{self.num_people}",
            #f"Robot Vel: {np.linalg.norm(self.robot.velocity):.2f} m/s",
            f"Position: ({self.robot.position[0]:.1f}, {self.robot.position[1]:.1f})",
            #f"Distance to door: {self.robot.door_position:.1f}"
        ]
        
        for i, text in enumerate(info_text):
            text_surface = font.render(text, True, (0, 0, 0))
            screen.blit(text_surface, (10, 10 + i * 25))
    
    def _check_collisions(self):
        """Check for collisions between robot and people"""
        robot_pos = self.robot.position
        robot_radius = self.robot.radius
        
        for person in self.people:
            if not person.active:
                continue
                
            person_pos = person.position
            person_radius = person.radius
            
            # Calculate distance between centers
            distance = np.linalg.norm(robot_pos - person_pos)
            
            # Check if collision occurs (distance < sum of radii)
            if distance < (robot_radius + person_radius):
                # Check if this is a new collision (not already recorded)
                collision_id = f"robot_person_{id(person)}"
                
                # Only count if this collision hasn't been recorded recently
                # (to avoid counting the same collision multiple times)
                recent_collision = any(
                    abs(coll['timestamp'] - datetime.now().timestamp()) < 0.1  # Within 0.1 seconds
                    for coll in self.collision_history
                    if coll.get('collision_id') == collision_id
                )
                
                if not recent_collision:
                    self.collision_count += 1
                    collision_info = {
                        'collision_id': collision_id,
                        'timestamp': datetime.now().timestamp(),
                        'robot_position': robot_pos.copy(),
                        'person_position': person_pos.copy(),
                        'distance': distance,
                        'collision_count': self.collision_count
                    }
                    self.collision_history.append(collision_info)
                    print(f"Collision detected! Total collisions: {self.collision_count}")
    
    def _record_simulation_data(self, dt: float, done: bool):
        """Record simulation data for each time step"""
        current_time = datetime.now()
        elapsed_time = (current_time - self.start_time).total_seconds()
        
        # Calculate distance traveled
        if self.previous_position is not None:
            distance_step = np.linalg.norm(self.robot.position - self.previous_position)
            self.total_distance_traveled += distance_step
        
        # Calculate velocity magnitude
        velocity_magnitude = np.linalg.norm(self.robot.velocity)
        
        # Record goal reached time
        if done and self.goal_reached_time is None:
            self.goal_reached_time = elapsed_time
        
        # =====================================================================
        # Track overlap metrics for comparison
        # =====================================================================
        robot_pos = self.robot.position
        robot_radius = self.robot.radius
        
        # Check overlap with persons (within threshold distance)
        in_person_overlap = False
        for person in self.people:
            if not person.active:
                continue
            dist_to_person = np.linalg.norm(robot_pos - person.position) - robot_radius - person.radius
            if dist_to_person < self.person_overlap_threshold:
                in_person_overlap = True
                break
        
        # Check overlap with door inflation zone
        door_pos = np.array(self.get_door_position())
        dist_to_door = np.linalg.norm(robot_pos - door_pos)
        in_door_overlap = dist_to_door < self.door_halo_radius
        
        # Track minimum clearance to door
        clearance_to_door = dist_to_door - robot_radius
        if clearance_to_door < self.min_clearance_to_door:
            self.min_clearance_to_door = clearance_to_door
        
        # Accumulate overlap times
        if in_person_overlap:
            self.overlap_time_persons += dt
        if in_door_overlap:
            self.overlap_time_door += dt
        if in_person_overlap and in_door_overlap:
            self.overlap_time_both += dt
        
        # Store data point
        data_point = {
            'timestamp': current_time.isoformat(),
            'elapsed_time': elapsed_time,
            'robot_x': float(self.robot.position[0]),
            'robot_y': float(self.robot.position[1]),
            'robot_velocity_x': float(self.robot.velocity[0]),
            'robot_velocity_y': float(self.robot.velocity[1]),
            'robot_velocity_magnitude': velocity_magnitude,
            'total_distance_traveled': self.total_distance_traveled,
            'goal_reached': done,
            'num_people': len(self.people),
            'collision_count': self.collision_count,
            'in_person_overlap': in_person_overlap,
            'in_door_overlap': in_door_overlap,
            'clearance_to_door': clearance_to_door,
            'dt': dt
        }
        
        self.simulation_data.append(data_point)
        
        # Update previous position for next iteration
        self.previous_position = self.robot.position.copy()
    
    def enable_data_recording(self, enabled: bool = True):
        """Enable or disable data recording"""
        self.data_recording_enabled = enabled
    
    def get_simulation_summary(self):
        """Get a summary of the simulation results"""
        if not self.simulation_data:
            return None
        
        # Calculate average velocity
        velocities = [point['robot_velocity_magnitude'] for point in self.simulation_data]
        avg_velocity = np.mean(velocities) if velocities else 0.0
        
        # Get final distance traveled
        final_distance = self.total_distance_traveled
        
        # Get time to reach goal
        time_to_goal = self.goal_reached_time if self.goal_reached_time else None
        
        # Get total simulation time
        total_time = self.simulation_data[-1]['elapsed_time'] if self.simulation_data else 0.0
        
        # Min clearance to door (handle inf case)
        min_door_clearance = self.min_clearance_to_door
        if min_door_clearance == float('inf'):
            min_door_clearance = -1.0  # Indicates never measured
        
        return {
            'total_simulation_time': total_time,
            'time_to_reach_goal': time_to_goal,
            'average_velocity': avg_velocity,
            'total_distance_traveled': final_distance,
            'goal_reached': self.goal_reached_time is not None,
            'total_collisions': self.collision_count,
            'total_data_points': len(self.simulation_data),
            # New comparison metrics
            'overlap_time_persons': self.overlap_time_persons,
            'overlap_time_door': self.overlap_time_door,
            'overlap_time_both': self.overlap_time_both,
            'min_clearance_to_door': min_door_clearance
        }
    
    def export_data_to_csv(self, filename: str = None):
        """Export simulation data to CSV file"""
        if not self.simulation_data:
            print("No simulation data to export")
            return None
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_data_{timestamp}.csv"
        
        # Ensure filename has .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # Create data directory if it doesn't exist
        data_dir = "simulation_data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        filepath = os.path.join(data_dir, filename)
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = self.simulation_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header
                writer.writeheader()
                
                # Write data
                writer.writerows(self.simulation_data)
            
            print(f"Simulation data exported to: {filepath}")
            
            # Print summary
            summary = self.get_simulation_summary()
            if summary:
                print("\nSimulation Summary:")
                print(f"  Total simulation time: {summary['total_simulation_time']:.2f} seconds")
                print(f"  Time to reach goal: {summary['time_to_reach_goal']:.2f} seconds" if summary['time_to_reach_goal'] else "  Goal not reached")
                print(f"  Average velocity: {summary['average_velocity']:.2f} m/s")
                print(f"  Total distance traveled: {summary['total_distance_traveled']:.2f} meters")
                print(f"  Total collisions: {summary['total_collisions']}")
                print(f"  Goal reached: {'Yes' if summary['goal_reached'] else 'No'}")
                print(f"  Total data points: {summary['total_data_points']}")
            
            return filepath
            
        except Exception as e:
            print(f"Error exporting data to CSV: {e}")
            return None
    
    def reset_data_recording(self):
        """Reset all recorded data"""
        self.simulation_data = []
        self.start_time = None
        self.goal_reached_time = None
        self.total_distance_traveled = 0.0
        self.previous_position = None
        self.collision_count = 0
        self.collision_history = []
        # Reset comparison metrics
        self.overlap_time_persons = 0.0
        self.overlap_time_door = 0.0
        self.overlap_time_both = 0.0
        self.min_clearance_to_door = float('inf')