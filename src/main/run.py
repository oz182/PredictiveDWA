import pygame
import numpy as np
import random
from typing import List, Tuple

import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim.sim import Simulation

def main(render=False):
    if render:
        pygame.init()
        width, height = 1000, 400
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Corridor Simulation")
        clock = pygame.time.Clock()
    else:
        screen = None
        clock = pygame.time.Clock()  # Still use clock for consistent dt

    # Create simulation with parameters
    sim = Simulation(
        corridor_width=4.0,
        door_side="right",  # Try "left" or "right"
        num_people=5,
        people_speeds=[random.uniform(2.0, 2.5) for _ in range(10)]
    )

    running = True
    while running:
        #dt = clock.tick(60) / 1000.0  # Delta time in seconds
        if render:
            dt = clock.tick(60) / 1000.0
        else:
            dt = 1 / 60.0  # Fixed timestep; simulation runs as fast as CPU allows


        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        # FOr learning
        state, reward, done = sim.step(dt)


        if render:
            screen.fill((255, 255, 255))
            sim.draw(screen)
            pygame.display.flip()

        # Optionally stop headless simulation after some condition
        if done:  # You can define `done()` in your Simulation
            running = False

    pygame.quit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true", help="Run simulation without rendering")
    args = parser.parse_args()

    main(render=not args.no_render)