import os
import sys
import math
import argparse
import random
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sim.sim import Simulation
from agents.ppo import PPO
from learning.train import extract_nav_features


def load_model(model_path, device='cpu'):
    # Infer input dimension using a temporary simulation and shared extractor
    tmp_sim = Simulation(corridor_width=4.0, door_side='right', num_people=3,
                         people_speeds=[random.uniform(0.6, 1.2) for _ in range(10)])
    _ = tmp_sim.step(1/60.0)
    input_dim = int(len(extract_nav_features(tmp_sim)))

    num_actions = 1  # Single offset value (must match training script)

    # Instantiate PPO agent (continuous actions) and load checkpoint
    agent = PPO(
        state_dim=input_dim,
        action_dim=num_actions,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.99,
        K_epochs=10,
        eps_clip=0.2,
        has_continuous_action_space=True,
        action_std_init=0.4,
    )
    agent.load(model_path)
    return agent


def plot_and_print_offsets(offset_vals):
    """Plot and print heading offset over time."""
    steps = list(range(1, len(offset_vals) + 1))

    plt.figure(figsize=(10, 4))
    plt.plot(steps, offset_vals, label='heading offset (rad)', color='tab:blue', linewidth=2)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='zero offset (follow path)')
    plt.title('Agent Heading Offset Over Time')
    plt.xlabel('Step')
    plt.ylabel('Offset (radians)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\nHeading offsets over time:")
    for i, offset in enumerate(offset_vals):
        offset_deg = offset * 180 / math.pi
        print(f"  step {i+1}: offset={offset:6.3f} rad ({offset_deg:6.1f}°)")


def main(render=True, model_path='checkpoints/theta_qnet.pt', episodes=3, action_select_interval=1):
    random.seed(int(time.time()))
    np.random.seed(int(time.time()))
    torch.manual_seed(int(time.time()))

    agent = load_model(model_path)

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

        max_steps = 3000
        offset_history = []
        prev_offset = None
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
            # Select new action every N steps, otherwise reuse previous one
            if (t % action_select_interval == 0) or (prev_offset is None):
                offset_normalized = agent.select_action(feat)  # Returns array
                prev_offset = float(offset_normalized[0])
            
            # Scale offset from [-1, 1] to [-π/6, π/6] (±30 degrees)
            max_offset = math.pi / 6
            offset = prev_offset * max_offset

            # Apply offset to planner
            if hasattr(sim.robot, 'nav'):
                sim.robot.nav.agent_offset = offset

            # Log chosen offset for this step
            offset_history.append(offset)

            # Step simulation
            _, _, done = sim.step(dt)

            # Render
            if render:
                screen.fill((255, 255, 255))
                # Pass state features into draw_v0 for debugging/visualization
                sim.draw_v0(screen, state_input=feat)
                import pygame
                pygame.display.flip()

            if done:
                break

        print(f"Episode {ep+1}/{episodes} finished in {t+1} steps")
        # Clear rollout buffer accumulated during select_action calls (no updates during eval)
        agent.buffer.clear()
        plot_and_print_offsets(offset_history)

    if render:
        import pygame
        pygame.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('--model', type=str, default='checkpoints/theta_qnet.pt', help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=3, help='Number of evaluation episodes')
    parser.add_argument('--action-select-interval', type=int, default=1, help='Select a new action every N steps (default 1 = every step)')
    args = parser.parse_args()
    main(render=args.render, model_path=args.model, episodes=args.episodes, action_select_interval=args.action_select_interval)


