import os
import sys
import math
import argparse
import random
import csv
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch

import matplotlib.pyplot as plt  # not strictly needed, but handy if you later add plots

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from real.sim import Simulation
from agents.ppo import PPO
from learning.train_v0 import extract_nav_features, compute_reward


def load_ppo_agent(model_path: str,
                   device: Optional[torch.device] = None) -> Tuple[PPO, int]:
    """Load a trained PPO agent and return (agent, input_dim)."""
    tmp_sim = Simulation(corridor_width=4.0,
                         door_side='right',
                         num_people=3,
                         people_speeds=[random.uniform(0.6, 1.2) for _ in range(10)])
    # Warm-up: real Simulation.step() does not take dt
    _ = tmp_sim.step(person_tracking=False)
    input_dim = int(len(extract_nav_features(tmp_sim)))

    num_actions = 1  # single continuous offset in [-1, 1]

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    return agent, input_dim


def configure_nav(sim: Simulation, algo: str) -> None:
    """Select local planner (TS-DWA or DWA) for the robot."""
    from algo.dwa import DWA
    from algo.ts_dwa import TSDWA

    algo = algo.lower()
    if algo == "dwa":
        # Plain DWA
        sim.robot.nav = DWA(
            position=sim.robot.position,
            velocity=sim.robot.velocity,
            max_speed=sim.robot.max_speed,
            goal=tuple(sim.robot.goal),
            radius=sim.robot.radius,
            corridor_bounds=sim.robot.corridor_bounds,
        )
        # Provide door information just like in Simulation.__init__
        if hasattr(sim.robot.nav, "set_door_info"):
            sim.robot.nav.set_door_info(sim.get_door_position(), sim.door_side)
        sim.robot.nav_type = "dwa"
        sim.robot.nav.door_aware_sampling = False
    elif algo == "dwa_door_aware":
        # DWA with door-aware sampling enabled
        sim.robot.nav = DWA(
            position=sim.robot.position,
            velocity=sim.robot.velocity,
            max_speed=sim.robot.max_speed,
            goal=tuple(sim.robot.goal),
            radius=sim.robot.radius,
            corridor_bounds=sim.robot.corridor_bounds,
        )
        if hasattr(sim.robot.nav, "set_door_info"):
            sim.robot.nav.set_door_info(sim.get_door_position(), sim.door_side)
        sim.robot.nav.door_aware_sampling = True
        sim.robot.nav_type = "dwa_door_aware"
    else:
        # Default TS-DWA (also used for PPO + TS-DWA)
        sim.robot.nav = TSDWA(
            position=sim.robot.position,
            velocity=sim.robot.velocity,
            max_speed=sim.robot.max_speed,
            goal=tuple(sim.robot.goal),
            radius=sim.robot.radius,
            corridor_bounds=sim.robot.corridor_bounds,
        )
        sim.robot.nav_type = "ts_dwa"

    # Ensure the local planner knows the goal
    if hasattr(sim.robot.nav, "set_goal") and sim.robot.goal is not None:
        sim.robot.nav.set_goal(tuple(sim.robot.goal))


def run_evaluation(
    algo: str,
    episodes: int,
    max_steps: int,
    render: bool,
    model_path: str,
    action_select_interval: int,
    base_seed: int,
    output_csv: Optional[str] = None,
) -> str:
    """Run multiple evaluation episodes and save per-episode stats to a CSV."""

    algo = algo.lower()
    use_ppo = algo == "ppo"

    agent: Optional[PPO] = None
    if use_ppo:
        agent, _ = load_ppo_agent(model_path)

    if render:
        import pygame
        pygame.init()
        width, height = 1000, 400
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f"Algorithm Evaluation: {algo.upper()}")
        clock = pygame.time.Clock()
    else:
        screen = None
        clock = None

    # Prepare output CSV only if not rendering
    writer: Optional[csv.DictWriter] = None
    if not render:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "simulation_data")
        os.makedirs(data_dir, exist_ok=True)
        if output_csv is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv = os.path.join(data_dir, f"algo_eval_{algo}_{ts}")

        fieldnames = [
            "episode",
            "seed",
            "algo",
            "return",
            "steps",
            "collisions",
            "overlap_free_pct",
            "overlap_person_pct",
            "overlap_door_pct",
            "overlap_both_pct",
            "avg_abs_offset",
            "max_abs_offset",
            "avg_door_dy",
        ]

        f = open(output_csv, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    try:
        for ep in range(episodes):
            # Per-episode seed
            seed = base_seed + ep
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            sim = Simulation(
                corridor_width=4.0,
                door_side="right",
                num_people=3,
                people_speeds=[random.uniform(0.6, 1.2) for _ in range(10)],
                spawn_interval=0.5,
                spawn_timer=0.5,
            )
            configure_nav(sim, "ts_dwa" if use_ppo else algo)

            # Warm-up step to initialize internal state
            _, _, _ = sim.step(person_tracking=False)

            episode_return = 0.0
            overlap_counts = {"none": 0, "person": 0, "door": 0, "both": 0}
            offset_history: List[float] = []
            door_dy_history: List[float] = []

            prev_offset = None

            for t in range(max_steps):
                # In real mode we still use the pygame clock to limit loop rate,
                # but `sim.step()` does not consume dt.
                if render and clock is not None:
                    clock.tick(60)

                # Handle events
                if render:
                    import pygame
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return output_csv

                # Read joystick buttons/axes (X toggles autonomous, Triangle toggles manual)
                # Simulation._button_callback maintains `sim.stop_run` and `sim.x_axis/sim.y_axis`.
                if hasattr(sim, "_button_callback"):
                    sim._button_callback()

                # State and (optional) action
                # Feature extraction is optional in real mode (depends on your train_v0 layout).
                try:
                    feat = extract_nav_features(sim)
                except Exception:
                    feat = None

                if use_ppo:
                    # Only run PPO when in autonomous mode (X mode)
                    if not getattr(sim, "stop_run", False) and feat is not None:
                        if (t % action_select_interval == 0) or (prev_offset is None):
                            offset_norm = agent.select_action(feat)
                            prev_offset = float(offset_norm[0])
                        # Scale offset from [-1, 1] to [-π/6, π/6] (±30°)
                        max_offset = math.pi / 6
                        offset = prev_offset * max_offset
                        if hasattr(sim.robot, "nav"):
                            sim.robot.nav.agent_offset = offset
                    else:
                        offset = 0.0
                else:
                    offset = 0.0  # pure planner, no RL offset

                offset_history.append(abs(offset))

                # Step (updates robot pose from odom + runs planner update_real)
                state, v, reward, done = sim.step(person_tracking=False)

                # Decide what to publish:
                # - X mode (stop_run == False): publish planner command (v,w)
                # - Triangle mode (stop_run == True): publish joystick command
                if getattr(sim, "stop_run", False):
                    cmd = [-float(getattr(sim, "y_axis", 0.0)), float(getattr(sim, "x_axis", 0.0))]
                else:
                    cmd = v

                # Publish using the sim's velocity publisher
                if hasattr(sim, "send_vel"):
                    sim.send_vel(cmd)

                # Reward & info (includes overlap, collisions, door_dy, etc.)
                try:
                    reward, _, info = compute_reward(sim, progress_prev_dist=0.0, offset=offset)
                    episode_return += reward
                    overlap_counts[info["overlap_type"]] += 1
                    if np.isfinite(info.get("door_dy", float("nan"))):
                        door_dy_history.append(float(info["door_dy"]))
                except Exception:
                    # In real mode, train_v0 reward shaping may not apply; keep loop running.
                    pass

                # Render with optional state debug text
                if render:
                    screen.fill((255, 255, 255))
                    if feat is not None:
                        sim.draw_v0(screen, state_input=feat)
                    else:
                        sim.draw_v0(screen, state_input=None)
                    import pygame
                    pygame.display.flip()

                if done:
                    break

            total_steps = t + 1
            overlap_pct = {k: 100.0 * v / total_steps for k, v in overlap_counts.items()}
            collisions = getattr(sim, "collision_count", 0)
            avg_abs_offset = float(np.mean(offset_history)) if offset_history else 0.0
            max_abs_offset = float(np.max(offset_history)) if offset_history else 0.0
            avg_door_dy = float(np.mean(door_dy_history)) if door_dy_history else float("nan")

            print(f"[{algo.upper()}] Episode {ep+1}/{episodes} | "
                  f"Return: {episode_return:.2f} | Steps: {total_steps} | "
                  f"Collisions: {collisions}")
            print(f"  Overlaps - Free: {overlap_pct['none']:.1f}% | "
                  f"Person: {overlap_pct['person']:.1f}% | "
                  f"Door: {overlap_pct['door']:.1f}% | "
                  f"Both: {overlap_pct['both']:.1f}%")

            if writer is not None:
                writer.writerow({
                    "episode": ep + 1,
                    "seed": seed,
                    "algo": algo,
                    "return": episode_return,
                    "steps": total_steps,
                    "collisions": collisions,
                    "overlap_free_pct": overlap_pct["none"],
                    "overlap_person_pct": overlap_pct["person"],
                    "overlap_door_pct": overlap_pct["door"],
                    "overlap_both_pct": overlap_pct["both"],
                    "avg_abs_offset": avg_abs_offset,
                    "max_abs_offset": max_abs_offset,
                    "avg_door_dy": avg_door_dy,
                })
    finally:
        if writer is not None:
            f.close()

    return output_csv if not render else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate navigation algorithms (PPO, TS-DWA, DWA) and log episode stats to CSV"
    )
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "ts_dwa", "dwa", "dwa_door_aware"],
                        help="Algorithm to evaluate: ppo (PPO+TS-DWA), ts_dwa, dwa, or dwa_door_aware.")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes.")
    parser.add_argument("--max-steps", type=int, default=3000,
                        help="Maximum steps per episode.")
    parser.add_argument("--model", type=str, default="checkpoints/theta_qnet.pt",
                        help="Path to trained PPO model (used when --algo ppo).")
    parser.add_argument("--action-select-interval", type=int, default=5,
                        help="Select a new PPO action every N steps (ppo only).")
    parser.add_argument("--seed", type=int, default=123,
                        help="Base random seed; each episode uses seed+episode_index.")
    parser.add_argument("--render", action="store_true",
                        help="Enable pygame rendering.")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Optional path for the output CSV (defaults to simulation_data/algo_eval_*.csv).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = run_evaluation(
        algo=args.algo,
        episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        model_path=args.model,
        action_select_interval=args.action_select_interval,
        base_seed=args.seed,
        output_csv=args.output_csv,
    )
    print(f"\nSaved evaluation results to: {csv_path}")


if __name__ == "__main__":
    main()


