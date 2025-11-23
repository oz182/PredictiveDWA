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
        people_speeds=[random.uniform(0.6, 1.2) for _ in range(10)]
    )

    running = True
    completed = False
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
            sim.draw_v0(screen)
            pygame.display.flip()

        # Optionally stop headless simulation after some condition
        if done:
            completed = True
            running = False

    # If rendering, wait for a key press before closing
    if render:
        # Draw completion message overlay once more
        try:
            msg = "Simulation complete. Press any key to exit."
            font = pygame.font.SysFont(None, 28)
            text_surface = font.render(msg, True, (0, 0, 0))
            # Draw semi-transparent banner at bottom
            banner_h = 40
            banner = pygame.Surface((width, banner_h))
            banner.set_alpha(180)
            banner.fill((240, 240, 240))
            screen.blit(banner, (0, height - banner_h))
            screen.blit(text_surface, ((width - text_surface.get_width()) // 2, height - banner_h + (banner_h - text_surface.get_height()) // 2))
            pygame.display.flip()
        except Exception:
            pass

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    waiting = False
            pygame.time.wait(50)

    pygame.quit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true", help="Run simulation without rendering")
    args = parser.parse_args()

    main(render=not args.no_render)