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
from sim.sim import Simulation
from agents.ppo import PPO
from agents.ppo_lstm import PPO_LSTM
from learning.train import extract_nav_features, compute_reward


def load_rl_agent(model_path: str,
                  algo: str,
                  agent_type: str = "ppo",
                  device: Optional[torch.device] = None) -> Tuple[Any, int]:
    """Load a trained RL agent (PPO or PPO_LSTM) and return (agent, input_dim)."""
    tmp_sim = Simulation(corridor_width=4.0,
                         corridor_length=20.0,
                         door_side='right',
                         num_people=5,
                         spawn_interval = random.uniform(0.5, 2.0),
                         people_speeds=[random.uniform(0.6, 1.2) for _ in range(10)])
    # Ensure the feature dimension matches the evaluation algorithm mode.
    configure_nav(tmp_sim, algo)
    _ = tmp_sim.step(1 / 60.0)
    input_dim = int(len(extract_nav_features(tmp_sim)))

    num_actions = 1  # single continuous offset in [-1, 1]

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    agent_type = str(agent_type).lower().strip()
    if agent_type == "ppo_lstm":
        agent = PPO_LSTM(
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
    else:
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
    macro_step: bool = False,
) -> str:
    """Run multiple evaluation episodes and save per-episode stats to a CSV."""

    algo = algo.lower()
    ppo_mode: Optional[str] = None
    if algo in ("ppo_ts_dwa", "ppo_lstm_ts_dwa"):
        ppo_mode = "ts_dwa"
    elif algo in ("ppo_dwa_door_aware", "ppo_lstm_dwa_door_aware"):
        ppo_mode = "dwa_door_aware"
    use_ppo = ppo_mode is not None
    use_lstm = algo.startswith("ppo_lstm_")

    agent: Optional[Any] = None
    if use_ppo:
        # Load PPO with the correct input dimension for the chosen PPO+planner mode
        agent, _ = load_rl_agent(model_path, algo=ppo_mode, agent_type=("ppo_lstm" if use_lstm else "ppo"))

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
                corridor_length=20.0,
                door_side="right",
                num_people = random.randint(2, 4),
                people_speeds=[random.uniform(0.6, 1.2) for _ in range(10)],
                spawn_interval = random.uniform(0.5, 2.0),
            )
            configure_nav(sim, ppo_mode if use_ppo else algo)

            # Warm-up step to initialize internal state
            _, _, _ = sim.step(1 / 60.0)

            episode_return = 0.0
            overlap_counts = {"none": 0, "person": 0, "door": 0, "both": 0}
            # TS-DWA PPO: abs(offset) in radians ; DWA-door-aware PPO: door_sampling_bias
            offset_history: List[float] = []
            door_dy_history: List[float] = []

            prev_offset = None
            lstm_hidden = None
            if use_ppo and use_lstm:
                # Reset LSTM hidden state per episode (evaluation-time recurrence)
                try:
                    lstm_hidden = agent.policy_old.init_hidden(batch_size=1)
                except Exception:
                    lstm_hidden = None

            t = 0
            while t < max_steps:
                dt = (clock.tick(60) / 1000.0) if render else (1 / 60.0)

                # Handle events
                if render:
                    import pygame
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return output_csv

                # Decision boundary: compute features and (optionally) choose a new action.
                feat = extract_nav_features(sim)
                if use_ppo:
                    is_decision = (t % action_select_interval == 0) or (prev_offset is None) or bool(macro_step)
                    if is_decision:
                        if use_lstm:
                            # Recurrent inference: sample from policy_old and advance hidden ONCE per decision.
                            st = torch.as_tensor(feat, dtype=torch.float32, device=next(agent.policy_old.parameters()).device)
                            with torch.no_grad():
                                if lstm_hidden is None:
                                    lstm_hidden = agent.policy_old.init_hidden(batch_size=1)
                                action_t, _, _, lstm_hidden = agent.policy_old.act(st, lstm_hidden)
                            prev_offset = max(-1.0, min(1.0, float(action_t.view(-1)[0].item())))
                        else:
                            offset_norm = agent.select_action(feat)
                            prev_offset = max(-1.0, min(1.0, float(offset_norm[0])))
                    else:
                        # Old (non-macro) action-hold behavior: still advance LSTM hidden each step.
                        if use_lstm:
                            st = torch.as_tensor(feat, dtype=torch.float32, device=next(agent.policy_old.parameters()).device)
                            with torch.no_grad():
                                if lstm_hidden is None:
                                    lstm_hidden = agent.policy_old.init_hidden(batch_size=1)
                                _, lstm_hidden = agent.policy_old._forward_seq(st.view(1, -1), lstm_hidden)

                    if ppo_mode == "ts_dwa":
                        max_offset = math.pi / 2
                        offset = prev_offset * max_offset
                        sim.robot.nav.agent_offset = offset
                        offset_history.append(abs(float(offset)))
                    else:
                        a = max(-1.0, min(1.0, float(prev_offset)))
                        mapped_bias = 0.6 + ((a + 1.0) * 0.5) * (0.85 - 0.6)
                        door_sampling_bias = float(mapped_bias)
                        if hasattr(sim.robot, "nav") and hasattr(sim.robot.nav, "door_sampling_bias"):
                            sim.robot.nav.door_sampling_bias = float(door_sampling_bias)
                        offset = 0.0
                        offset_history.append(float(door_sampling_bias))
                else:
                    offset = 0.0

                # Execute either one sim step or N micro-steps (macro-step).
                inner_steps = action_select_interval if (macro_step and action_select_interval > 1) else 1
                done = False
                reward_sum = 0.0
                last_info: Dict[str, Any] = {}
                for _ in range(inner_steps):
                    if t >= max_steps:
                        break
                    _, _, done = sim.step(dt)

                    reward, _, info = compute_reward(sim, progress_prev_dist=0.0, offset=offset)
                    reward_sum += float(reward)
                    last_info = info

                    overlap_counts[info["overlap_type"]] += 1
                    if np.isfinite(info.get("door_dy", float("nan"))):
                        door_dy_history.append(float(info["door_dy"]))

                    if render:
                        screen.fill((255, 255, 255))
                        sim.draw_v0(screen, state_input=feat)
                        import pygame
                        pygame.display.flip()

                    t += 1
                    if done:
                        break

                episode_return += float(reward_sum)
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
                        choices=["ppo_ts_dwa", "ppo_dwa_door_aware",
                                 "ppo_lstm_ts_dwa", "ppo_lstm_dwa_door_aware",
                                 "ts_dwa", "dwa", "dwa_door_aware"],
                        help="Algorithm to evaluate: ppo_ts_dwa (PPO controls TS-DWA agent_offset), "
                             "ppo_dwa_door_aware (PPO controls DWA door_sampling_bias), "
                             "ppo_lstm_ts_dwa / ppo_lstm_dwa_door_aware (same but with recurrent PPO-LSTM), "
                             "or baselines: ts_dwa, dwa, dwa_door_aware.")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes.")
    parser.add_argument("--max-steps", type=int, default=3000,
                        help="Maximum steps per episode.")
    parser.add_argument("--model", type=str, default="checkpoints/theta_qnet.pt",
                        help="Path to trained PPO model (used when --algo ppo).")
    parser.add_argument("--action-select-interval", type=int, default=5,
                        help="Select a new PPO action every N steps (ppo only).")
    parser.add_argument("--macro-step", action="store_true",
                        help="If set, treat action_select_interval as a true frame-skip: apply one action for N sim steps "
                             "and accumulate reward across those N steps.")
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
        macro_step=bool(args.macro_step),
    )
    print(f"\nSaved evaluation results to: {csv_path}")


if __name__ == "__main__":
    main()


