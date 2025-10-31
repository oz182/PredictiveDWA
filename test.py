import os
import sys
import math
import argparse
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from sim.sim import Simulation
from agents.ppo import PPO
from train import extract_nav_features


def load_model(model_path, device='cpu'):
    # Infer input dimension using a temporary simulation and shared extractor
    tmp_sim = Simulation(corridor_width=4.0, door_side='right', num_people=3,
                         people_speeds=[random.uniform(0.6, 1.2) for _ in range(10)])
    _ = tmp_sim.step(1/60.0)
    input_dim = int(len(extract_nav_features(tmp_sim)))

    num_actions = 2  # must match training script's num_actions

    # Instantiate PPO agent (discrete actions) and load checkpoint
    agent = PPO(
        state_dim=input_dim,
        action_dim=num_actions,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.99,
        K_epochs=10,
        eps_clip=0.2,
        has_continuous_action_space=True,
        action_std_init=0.6,
    )
    agent.load(model_path)
    return agent


def plot_and_print_weights(left_vals, right_vals):
    """Plot and print left_weight and right_weight over time."""
    steps = list(range(1, max(len(left_vals), len(right_vals)) + 1))

    plt.figure(figsize=(8, 3))
    plt.plot(steps[:len(left_vals)], left_vals, label='left_weight', color='tab:blue', linewidth=2)
    plt.plot(steps[:len(right_vals)], right_vals, label='right_weight', color='tab:orange', linewidth=2)
    plt.title('left/right weights over time')
    plt.xlabel('step')
    plt.ylabel('weight')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("weights over time:")
    for i in range(len(steps)):
        lw = left_vals[i] if i < len(left_vals) else float('nan')
        rw = right_vals[i] if i < len(right_vals) else float('nan')
        print(f"  step {i+1}: left={lw:.3f} right={rw:.3f}")


def main(render=True, model_path='checkpoints/theta_qnet.pt', episodes=3, action_select_interval=1):
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

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
        left_history = []
        right_history = []
        prev_action_tuple = None  # (left_weight, right_weight)
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
            if (t % action_select_interval == 0) or (prev_action_tuple is None):
                left_weight, right_weight = agent.select_action(feat)
                prev_action_tuple = (float(left_weight), float(right_weight))
            else:
                left_weight, right_weight = prev_action_tuple

            # Apply weights
            if hasattr(sim.robot, 'nav'):
                sim.robot.nav.left_weight = float(left_weight)
                sim.robot.nav.right_weight = float(right_weight)

            # Log chosen weights for this step
            left_history.append(float(left_weight))
            right_history.append(float(right_weight))

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
        # Clear rollout buffer accumulated during select_action calls (no updates during eval)
        agent.buffer.clear()
        plot_and_print_weights(left_history, right_history)

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


