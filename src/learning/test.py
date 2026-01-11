import os
import sys
import math
import argparse
import random
import time
import csv
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sim.sim import Simulation
from agents.td3 import TD3
from learning.train import extract_nav_features


def save_episode_csv(sim, episode_id: int, offset_history: list):
    """
    Save episode data to CSV with fields matching monte_carlo.py format.
    Saves to current directory.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"test_learned_ep{episode_id:03d}_{timestamp}.csv"
    
    if not sim.simulation_data:
        print("No simulation data to export")
        return None
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            # Fields matching sim.simulation_data + agent_offset
            fieldnames = [
                'timestamp', 'elapsed_time', 
                'robot_x', 'robot_y',
                'robot_velocity_x', 'robot_velocity_y', 'robot_velocity_magnitude',
                'total_distance_traveled', 'goal_reached', 'num_people',
                'collision_count', 'in_person_overlap', 'in_door_overlap',
                'clearance_to_door', 'dt', 'agent_offset'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, point in enumerate(sim.simulation_data):
                row = {
                    'timestamp': point['timestamp'],
                    'elapsed_time': point['elapsed_time'],
                    'robot_x': point['robot_x'],
                    'robot_y': point['robot_y'],
                    'robot_velocity_x': point['robot_velocity_x'],
                    'robot_velocity_y': point['robot_velocity_y'],
                    'robot_velocity_magnitude': point['robot_velocity_magnitude'],
                    'total_distance_traveled': point['total_distance_traveled'],
                    'goal_reached': point['goal_reached'],
                    'num_people': point['num_people'],
                    'collision_count': point['collision_count'],
                    'in_person_overlap': point['in_person_overlap'],
                    'in_door_overlap': point['in_door_overlap'],
                    'clearance_to_door': point['clearance_to_door'],
                    'dt': point['dt'],
                    'agent_offset': offset_history[i] if i < len(offset_history) else 0.0
                }
                writer.writerow(row)
        
        # Print summary
        summary = sim.get_simulation_summary()
        print(f"  CSV saved: {filename}")
        if summary:
            print(f"  Summary: time={summary['total_simulation_time']:.2f}s, "
                  f"dist={summary['total_distance_traveled']:.2f}m, "
                  f"collisions={summary['total_collisions']}, "
                  f"overlap_door={summary.get('overlap_time_door', 0):.3f}s")
        return filename
    except Exception as e:
        print(f"Error saving CSV: {e}")
        return None


def load_model(model_path, device='cpu'):
    """
    Load a TD3 model from checkpoint, inferring architecture from saved weights.
    
    Handles multiple checkpoint formats:
    - New format: {'actor': state_dict, 'actor_target': ..., 'critic': ..., 'critic_target': ...}
    - Legacy format: direct state dict with 'actor.*' or 'net.*' keys
    """
    # Load checkpoint to infer dimensions
    checkpoint = torch.load(model_path, map_location=device)
    
    # Determine state_dim and hidden_size from checkpoint weights
    if 'actor' in checkpoint and isinstance(checkpoint['actor'], dict):
        # New format: nested dict
        first_layer_key = 'net.0.weight'
        if first_layer_key in checkpoint['actor']:
            weight_shape = checkpoint['actor'][first_layer_key].shape
            hidden_size = weight_shape[0]
            state_dim = weight_shape[1]
        else:
            raise ValueError(f"Cannot infer dimensions: missing {first_layer_key} in checkpoint['actor']")
    else:
        # Legacy format: direct state dict with 'actor.*' or 'net.*' keys
        first_key = next(iter(checkpoint.keys()), '')
        if first_key.startswith('actor.'):
            # Legacy 'actor.*' format - need to remap to 'net.*'
            weight_key = 'actor.0.weight'
        elif first_key.startswith('net.'):
            weight_key = 'net.0.weight'
        else:
            raise ValueError(f"Unknown checkpoint format: first key is '{first_key}'")
        
        if weight_key in checkpoint:
            weight_shape = checkpoint[weight_key].shape
            hidden_size = weight_shape[0]
            state_dim = weight_shape[1]
        else:
            raise ValueError(f"Cannot infer dimensions: missing {weight_key} in checkpoint")
    
    print(f"Inferred from checkpoint: state_dim={state_dim}, hidden_size={hidden_size}")

    num_actions = 1  # Single offset value (must match training script)

    # Instantiate TD3 agent with matching architecture
    agent = TD3(
        state_dim=state_dim,
        action_dim=num_actions,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        max_action=1.0,
        hidden_size=hidden_size,
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


def set_seed(episode_id: int, base_seed: int = None) -> int:
    """Set random seeds for reproducibility.
    
    Args:
        episode_id: Current episode ID (0-indexed). Combined with base_seed for unique per-episode seed.
        base_seed: Base seed value. If None, uses time-based seed (non-reproducible).
    
    Returns:
        The seed that was used.
    """
    if base_seed is not None:
        seed = base_seed + episode_id
    else:
        seed = int(time.time()) + episode_id
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    return seed


def main(render=True, model_path='checkpoints/td3_policy.pt', episodes=3, action_select_interval=3, save_csv=False, seed=None):
    # Set initial seed for model loading
    initial_seed = set_seed(0, seed)
    
    print(f"Base seed: {seed if seed is not None else 'time-based (non-reproducible)'}")

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
        # Set seed for this episode (for reproducibility)
        episode_seed = set_seed(ep, seed)
        print(f"\nStarting episode {ep+1}/{episodes} (seed={episode_seed})")
        
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
                offset_normalized = agent.select_action(feat, add_noise=False)  # No exploration noise during eval
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
        
        # Save CSV if enabled
        if save_csv:
            save_episode_csv(sim, ep + 1, offset_history)
        
        # plot_and_print_offsets(offset_history)

    if render:
        import pygame
        pygame.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('--model', type=str, default='checkpoints/td3_policy.pt', help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=3, help='Number of evaluation episodes')
    parser.add_argument('--action-select-interval', type=int, default=1, help='Select a new action every N steps (default 1 = every step)')
    parser.add_argument('--save-csv', action='store_true', help='Save episode data to CSV')
    parser.add_argument('--seed', type=int, default=None, help='Base random seed for reproducibility. Episode i uses seed+i. (default: time-based)')
    args = parser.parse_args()
    main(render=args.render, model_path=args.model, episodes=args.episodes, action_select_interval=args.action_select_interval, save_csv=args.save_csv, seed=args.seed)


