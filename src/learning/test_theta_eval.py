import os
import sys
import math
import argparse
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sim.sim import Simulation
from learning.rl_theta_net import ThetaQNet


def get_forward_proxemic_cost(sim) -> float:
    """
    Extract average costmap value in forward-looking region.
    Returns normalized value 0.0-1.0.
    """
    costmap = sim.robot.get_egocentric_costmap(size=4.0, resolution=0.1)
    grid_size = costmap.shape[0]
    
    # Forward region: front quarter of costmap
    # Costmap is robot-centric with robot at center
    center = grid_size // 2
    quarter = grid_size // 4
    
    # Extract front quarter (ahead of robot)
    forward_region = costmap[center-quarter:center+quarter, center:]
    
    # Compute mean and normalize to 0-1
    if forward_region.size > 0:
        mean_cost = np.mean(forward_region) / 255.0
        return float(mean_cost)
    else:
        return 0.0


def extract_nav_features(sim) -> np.ndarray:
    """
    Minimal feature vector from current simulation state.
    Uses the same quantities already computed in Robot.get_navigation_info.
    Must match the training script feature extraction exactly.
    """
    nav = sim.robot.get_navigation_info(2)
    # waypoint(2), door_position(2), door_angle(1), linear_velocity(1), angular_velocity(1), closest_obstacle_distance(1)
    feat = []
    feat.extend(list(nav['waypoint']))
    feat.extend(list(nav['door_position']))
    feat.append(float(nav['door_angle']))
    feat.append(float(nav['linear_velocity']))
    feat.append(float(nav['angular_velocity']))
    feat.append(float(nav['closest_obstacle_distance']))
    
    # Count people within velocity-based dynamic radius
    velocity_magnitude = np.linalg.norm(sim.robot.velocity)
    sensing_radius = np.clip(velocity_magnitude * 2.5, 1.5, 4.0)
    num_people_nearby = 0
    for person in sim.robot.people:
        if person.active:
            dist = np.linalg.norm(person.position - sim.robot.position)
            if dist <= sensing_radius:
                num_people_nearby += 1
    feat.append(float(num_people_nearby))
    
    # Forward proxemic cost from costmap
    forward_cost = get_forward_proxemic_cost(sim)
    feat.append(float(forward_cost))
    
    return np.asarray(feat, dtype=np.float32)


def load_model(model_path, device='cpu'):
    input_dim = 10  # 8 original + num_people_nearby + forward_proxemic_cost
    num_actions = 4  # must match training script
    net = ThetaQNet(input_dim, num_actions)
    state = torch.load(model_path, map_location=device)
    net.load_state_dict(state)
    net.eval()
    return net


def plot_and_print_theta_actions(theta_vals):
    """Plot and then print the selected theta_range values over time (degrees)."""
    steps = list(range(1, len(theta_vals) + 1))
    degrees = []
    for v in theta_vals:
        try:
            degrees.append(math.degrees(float(v)))
        except Exception:
            degrees.append(float('nan'))

    # Plot
    plt.figure(figsize=(8, 3))
    plt.plot(steps, degrees, marker='o', linewidth=1)
    plt.title('theta_range over time (deg)')
    plt.xlabel('step')
    plt.ylabel('degrees')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print
    print("theta_actions over time (deg):")
    for i, deg in zip(steps, degrees):
        print(f"  step {i}: {deg:.1f}")


def main(render=True, model_path='checkpoints/theta_qnet.pt', episodes=3):
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    # Discrete theta_range values (must match training)
    theta_actions = np.array([
        math.radians(10),
        math.radians(20),
        math.radians(30),
        math.radians(45),
    ], dtype=np.float32)

    net = load_model(model_path)

    if render:
        import pygame
        pygame.init()
        width, height = 1000, 400
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Theta-Range Evaluation (TS-DWA)")
        clock = pygame.time.Clock()
    else:
        screen = None
        clock = None

    for ep in range(episodes):
        sim = Simulation(corridor_width=4.0, door_side='right', num_people=3,
                         people_speeds=[random.uniform(0.6, 1.2) for _ in range(10)])
        # Warm-up step to init internal state
        _, _, _ = sim.step(1/60.0)

        max_steps = 800
        theta_history = []
        for t in range(max_steps):
            dt = (clock.tick(60) / 1000.0) if render else (1 / 60.0)

            # Events
            if render:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

            # State and action
            feat = extract_nav_features(sim)
            with torch.no_grad():
                q = net(torch.from_numpy(feat).unsqueeze(0))
                a = int(q.argmax(dim=1).item())
            theta_val = float(theta_actions[a])

            # Apply theta_range
            if hasattr(sim.robot, 'nav') and hasattr(sim.robot.nav, 'theta_range'):
                sim.robot.nav.theta_range = theta_val

            # Log chosen theta for this step
            theta_history.append(theta_val)

            # Step simulation
            _, _, done = sim.step(dt)

            # Render
            if render:
                screen.fill((255, 255, 255))
                sim.draw_v0(screen)
                import pygame
                pygame.display.flip()

            if done:
                break

        print(f"Episode {ep+1}/{episodes} finished in {t+1} steps")
        plot_and_print_theta_actions(theta_history)

    if render:
        import pygame
        pygame.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('--model', type=str, default='checkpoints/theta_qnet.pt', help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=3, help='Number of evaluation episodes')
    args = parser.parse_args()
    main(render=args.render, model_path=args.model, episodes=args.episodes)


