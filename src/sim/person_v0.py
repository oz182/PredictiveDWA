import pygame
import numpy as np
import random
from typing import List, Tuple

class Person:
    def __init__(self, position: Tuple[float, float], radius: float, speed: float, target: Tuple[float, float]):
        self.position = np.array(position, dtype=float)
        self.radius = radius
        self.speed = speed
        self.target = np.array(target)
        self.active = True
    
    def update(self, dt: float):
        if not self.active:
            return
            
        direction = self.target - self.position
        distance = np.linalg.norm(direction)
        
        if distance < 0.1:  # Reached target
            self.active = False
            return
            
        if distance > 0:
            direction = direction / distance
            
        self.position += direction * self.speed * dt
    
    def draw(self, screen, scale, offset):
        pos = (self.position * scale + offset).astype(int)
        pygame.draw.circle(screen, (255, 0, 0), pos, int(self.radius * scale))

class Robot:
    def __init__(self, position: Tuple[float, float], radius: float):
        self.position = np.array(position, dtype=float)
        self.radius = radius
        self.velocity = np.array([0.0, 0.0])
        self.max_speed = 2.0
        self.goal = None
    
    def set_goal(self, goal: Tuple[float, float]):
        self.goal = np.array(goal)
    
    def update(self, dt: float, people: List[Person]):
        if self.goal is None:
            return
            
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

    def dwa_update(self, dt: float, people: List[Person]):
        if self.goal is None:
            return
            
        # DWA parameters
        max_speed = 2.0
        max_rotation = np.pi/2
        dt_lookahead = 1.0  # Time to look ahead for collisions
        v_resolution = 0.1
        w_resolution = np.pi/16
        
        # Generate admissible velocities
        admissible_velocities = []
        for v in np.arange(0, max_speed + v_resolution, v_resolution):
            for w in np.arange(-max_rotation, max_rotation + w_resolution, w_resolution):
                admissible_velocities.append((v, w))
        
        # Evaluate each velocity
        best_score = -float('inf')
        best_velocity = (0, 0)
        
        for v, w in admissible_velocities:
            # Simulate trajectory
            trajectory = self.simulate_trajectory(v, w, dt_lookahead)
            
            # Calculate scores
            goal_score = self.calculate_goal_score(trajectory)
            clearance_score = self.calculate_clearance_score(trajectory, people)
            speed_score = v / max_speed  # Normalized
            
            # Weighted total score
            total_score = 0.5*goal_score + 0.3*clearance_score + 0.2*speed_score
            
            if total_score > best_score:
                best_score = total_score
                best_velocity = (v, w)
        
        # Apply best velocity
        v, w = best_velocity
        self.velocity = np.array([v * np.cos(w), v * np.sin(w)])
        self.position += self.velocity * dt
    
    def draw(self, screen, scale, offset):
        pos = (self.position * scale + offset).astype(int)
        pygame.draw.circle(screen, (0, 0, 255), pos, int(self.radius * scale))
        # Draw direction indicator
        end_pos = (self.position + self.velocity * 0.5) * scale + offset
        pygame.draw.line(screen, (0, 255, 0), pos, end_pos.astype(int), 2)

class Simulation:
    def __init__(self, corridor_width: float = 5.0, door_side: str = "right", 
                 num_people: int = 5, people_speeds: List[float] = None):
        self.corridor_width = corridor_width
        self.door_side = door_side
        self.num_people = num_people
        self.people_speeds = people_speeds if people_speeds else [random.uniform(0.5, 1.5) for _ in range(num_people)]
        
        # Corridor dimensions
        self.corridor_length = 20.0
        self.door_position = 0.4 * self.corridor_length  # Door is 40% along corridor
        
        # Initialize agents
        self.robot = Robot((1.0, corridor_width/2), 0.5)
        self.robot.set_goal((self.corridor_length - 1.0, corridor_width/2))
        
        self.people: List[Person] = []
        self.spawn_timer = 0
        self.spawn_interval = 1.0  # Spawn a person every second
        
        # For visualization
        self.scale = 40  # pixels per meter
        self.offset = np.array([50, 50])
    
    def spawn_person(self):
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
    
    def update(self, dt: float):
        # Spawn people
        self.spawn_timer += dt
        if self.spawn_timer >= self.spawn_interval and len(self.people) < self.num_people:
            self.spawn_person()
            self.spawn_timer = 0
            
        # Update agents
        self.robot.update(dt, self.people)
        for person in self.people:
            person.update(dt)
        
        # Remove inactive people
        self.people = [p for p in self.people if p.active]
    
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

def main():
    pygame.init()
    width, height = 1000, 400
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Corridor Simulation")
    clock = pygame.time.Clock()
    
    # Create simulation with parameters
    sim = Simulation(
        corridor_width=5.0,
        door_side="right",  # Try "left" or "right"
        num_people=10,
        people_speeds=[random.uniform(0.5, 1.5) for _ in range(10)]
    )
    
    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # Delta time in seconds
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        sim.update(dt)
        
        screen.fill((255, 255, 255))
        sim.draw(screen)
        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    main()