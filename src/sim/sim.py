import pygame
import numpy as np
import random
from typing import List, Tuple

from sim.person import Person
from sim.robot import Robot


class Simulation:
    def __init__(self, corridor_width: float = 5.0, door_side: str = "right", 
                 num_people: int = 5, people_speeds: List[float] = None):
        self.corridor_width = corridor_width
        self.door_side = door_side
        self.num_people = num_people
        self.people_speeds = people_speeds if people_speeds else [random.uniform(2.0, 2.5) for _ in range(num_people)]
        
        # Corridor dimensions
        self.corridor_length = 20.0
        self.door_position = 0.4 * self.corridor_length  # Door is 40% along corridor
        self.corridor_bounds = {
            'x_min': 0,
            'x_max': self.corridor_length,
            'y_min': 0,
            'y_max': self.corridor_width
        }
        
        # Initialize agents
        self.robot = Robot((1.0, corridor_width/2), 0.5, self.corridor_bounds, self.get_door_position())
        self.robot.set_goal((self.corridor_length - 1.0, corridor_width/1.2 + 0.5))
        
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
        self.people.append(Person((door_x, door_y), 0.3, speed, self.door_side, self.corridor_width, self.corridor_length))
    
    def step(self, dt: float):
        # Spawn people
        self.spawn_timer += dt
        if self.spawn_timer >= self.spawn_interval and len(self.people) < self.num_people:
            self.spawn_person()
            self.spawn_timer = 0
            self.spawn_interval = random.uniform(0.5, 3.0)
            
        # Update agents
        state, reward, done = self.robot.update(dt, self.people)
        for person in self.people:
            person.update(dt)
        
        # Remove inactive people
        self.people = [p for p in self.people if p.active]

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

    def draw_v0(self, screen):
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