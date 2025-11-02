import os
import sys
import math
import random
from collections import deque
import argparse
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from sim.sim import Simulation
from models.rl_theta_net import ThetaQNet

# Optional third-party integrations

import wandb  
import optuna  



def extract_nav_features(sim) -> np.ndarray:
    """
    Feature vector from current simulation state.
    Includes:
      - waypoint(2), door_position(2), door_angle(1)
      - linear_velocity(1), angular_velocity(1)
      - three closest people relative positions wrt robot: [(dx, dy) x 3] (pad with large value if <3)
      - distances to corridor left and right boundaries (y - y_min, y_max - y)
    """
    nav = sim.robot.get_navigation_info(2)

    robot_pos = np.asarray(sim.robot.position, dtype=float)
    large_val = 10

    # Compute three closest people relative positions (dx, dy)
    rel_people: list[tuple[float, float, float]] = []  # (dist, dx, dy)
    if hasattr(sim.robot, 'people'):
        for person in sim.robot.people:
            if getattr(person, 'active', True):
                p = np.asarray(person.position, dtype=float)
                dx = float(p[0] - robot_pos[0])
                dy = float(p[1] - robot_pos[1])
                d = math.hypot(dx, dy)
                rel_people.append((d, dx, dy))
    rel_people.sort(key=lambda t: t[0])
    # Take up to 3, pad if fewer
    rel_feats: list[float] = []
    for i in range(3):
        if i < len(rel_people):
            _, dx, dy = rel_people[i]
            rel_feats.extend([dx, dy])
        else:
            rel_feats.extend([large_val, large_val])

    # Distances to corridor left/right sides using y-bounds
    dist_left = float(large_val)
    dist_right = float(large_val)
    if hasattr(sim.robot, 'corridor_bounds'):
        bounds = sim.robot.corridor_bounds
        y_min = float(bounds['y_min'])
        y_max = float(bounds['y_max'])
        y = float(robot_pos[1])
        dist_left = max(0.0, y - y_min)
        dist_right = max(0.0, y_max - y)

    feat = []
    feat.extend(list(map(float, nav['waypoint'])))
    feat.extend(list(map(float, nav['door_position'])))
    feat.append(float(nav['door_angle']))
    feat.append(float(nav['linear_velocity']))
    feat.append(float(nav['angular_velocity']))
    feat.extend(rel_feats)              # (dx, dy) x 3
    feat.append(dist_left)
    feat.append(dist_right)
    return np.asarray(feat, dtype=np.float32)


def check_robot_overlap(sim) -> dict:
    """
    Check if the robot overlaps with person inflation zones or door inflation zone.
    
    Returns:
        dict with keys:
            - 'person_overlap': bool, True if robot is in any person's inflation zone
            - 'door_overlap': bool, True if robot is in door's inflation zone
            - 'overlap_type': str, one of 'none', 'person', 'door', 'both'
    """
    robot_pos = sim.robot.position
    robot_radius = sim.robot.radius
    
    # Check person overlaps
    person_overlap = False
    person_inflation_radius = 0.2  # Same as used in costmap
    
    for person in sim.robot.people:
        if not person.active:
            continue
        
        # Get person's proxemic ellipse parameters
        axes = getattr(person, 'proxemic_axes', np.array([person.radius, person.radius], dtype=float)).astype(float)
        a = max(float(axes[0]), 1e-4)  # semi-minor axis (width)
        b = max(float(axes[1]), 1e-4)  # semi-major axis (length)
        
        # Ellipse offset (person is at rear of ellipse)
        ellipse_offset_ratio = 0.3
        ellipse_offset = b * ellipse_offset_ratio
        
        # Get person heading
        heading_angle = getattr(person, 'heading_angle', 0.0)
        
        # Vector from person to robot
        rel_x = robot_pos[0] - person.position[0]
        rel_y = robot_pos[1] - person.position[1]
        
        # Rotate into ellipse-aligned frame
        cos_t = math.cos(heading_angle)
        sin_t = math.sin(heading_angle)
        local_x = rel_x * cos_t - rel_y * sin_t
        local_y = rel_x * sin_t + rel_y * cos_t
        
        # Apply offset (shift ellipse forward)
        local_x_shifted = local_x + ellipse_offset
        
        # Check if robot center is inside the ellipse
        norm = math.sqrt((local_x_shifted / b) ** 2 + (local_y / a) ** 2)
        
        # Calculate distance to ellipse boundary
        if norm > 1e-6:
            scale = 1.0 / norm
            boundary_x_shifted = local_x_shifted * scale
            boundary_y = local_y * scale
            boundary_x = boundary_x_shifted - ellipse_offset
            diff_x = local_x - boundary_x
            diff_y = local_y - boundary_y
            dist_to_boundary = math.hypot(diff_x, diff_y)
            if norm < 1.0:
                dist_to_boundary = -dist_to_boundary
        else:
            dist_to_boundary = -min(a, b)
        
        # Check if robot overlaps with inflated ellipse
        clearance = dist_to_boundary - robot_radius
        if clearance <= person_inflation_radius:
            person_overlap = True
            break
    
    # Check door overlap
    door_overlap = False
    if hasattr(sim.robot, 'door_position') and hasattr(sim.robot, 'corridor_bounds'):
        door_pos = np.array(sim.robot.door_position, dtype=float)
        bounds = sim.robot.corridor_bounds
        
        # Door halo radius
        door_inflation_radius = float(getattr(sim.robot.global_planner, 'door_halo_radius', 1.0))
        
        # Distance from robot to door
        dist_to_door = np.linalg.norm(robot_pos - door_pos)
        
        # Determine door side and inward normal
        corridor_mid_y = (bounds['y_min'] + bounds['y_max']) * 0.5
        door_side = "left" if door_pos[1] < corridor_mid_y else "right"
        n_world = np.array([0.0, 1.0]) if door_side == "left" else np.array([0.0, -1.0])
        
        # Vector from door to robot
        v_x = robot_pos[0] - door_pos[0]
        v_y = robot_pos[1] - door_pos[1]
        
        # Check if robot is on the inward-facing side (semicircle)
        dot_product = n_world[0] * v_x + n_world[1] * v_y
        
        # Robot overlaps if within door halo radius AND on inward-facing side
        if dist_to_door <= (door_inflation_radius + robot_radius) and dot_product > 0.0:
            door_overlap = True
    
    # Determine overlap type
    if person_overlap and door_overlap:
        overlap_type = 'both'
    elif person_overlap:
        overlap_type = 'person'
    elif door_overlap:
        overlap_type = 'door'
    else:
        overlap_type = 'none'
    
    return {
        'person_overlap': person_overlap,
        'door_overlap': door_overlap,
        'overlap_type': overlap_type
    }


def compute_reward(sim, progress_prev_dist: float) -> tuple[float, float, dict]:
    """
    Reward based on progress, overlap avoidance, and goal achievement.
    Returns (reward, new_progress_distance, info)
    """
    robot_pos = sim.robot.position
    goal_pos = sim.robot.goal
    dist = float(np.linalg.norm(goal_pos - robot_pos))

    progress = (progress_prev_dist - dist)  # positive if moving towards goal
    reward = 1.0 * progress

    # Check overlap with inflation zones
    overlap_info = check_robot_overlap(sim)
    
    # Reward/penalty based on overlap type
    overlap_type = overlap_info['overlap_type']
    if overlap_type == 'none':
        # Positive reward for staying in free space
        reward += 0.2
    elif overlap_type == 'person':
        # Penalty for being in person's proxemic zone
        reward += -0.5
    elif overlap_type == 'door':
        # Penalty for being in door's inflation zone
        reward += -0.3
    elif overlap_type == 'both':
        # Higher penalty for being in both zones
        reward += -0.8

    # time penalty (encourage faster navigation)
    reward += -0.005

    # collision penalty (count increment since last step)
    collisions = sim.collision_count
    info = {
        'distance': dist,
        'collisions': collisions,
        'overlap_type': overlap_type,
        'person_overlap': overlap_info['person_overlap'],
        'door_overlap': overlap_info['door_overlap'],
    }
    # If a collision occurred in this step (heuristic: last history item within ~0.2s)
    if sim.collision_history:
        if abs(sim.collision_history[-1]['timestamp'] - __import__('datetime').datetime.now().timestamp()) < 0.2:
            reward += -5.5

    # goal bonus
    if dist < 1.0:
        reward += 25.0

    return reward, dist, info


def train(config: Dict[str, Any], use_wandb: bool = True, run_name: Optional[str] = None) -> float:
    seed = int(config.get('seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Discrete theta_range actions (keep tiny and intuitive)
    theta_actions = np.array([
        math.radians(10),
        math.radians(20),
        math.radians(30),
        math.radians(45),
    ], dtype=np.float32)
    num_actions = len(theta_actions)

    # Environment / training hyperparameters
    gamma = float(config.get('gamma', 0.99))
    tau = float(config.get('tau', 0.01))
    epsilon_start = float(config.get('epsilon_start', 1.0))
    epsilon_end = float(config.get('epsilon_end', 0.05))
    epsilon_decay_steps = int(config.get('epsilon_decay_steps', 10_000))
    batch_size = int(config.get('batch_size', 64))
    buffer_capacity = int(config.get('buffer_capacity', 50_000))
    buffer = deque(maxlen=buffer_capacity)

    # Env defaults (also used to infer feature dimension)
    dt = float(config.get('dt', 1/60.0))
    corridor_width = float(config.get('corridor_width', 4.0))
    door_side = str(config.get('door_side', 'right'))
    num_people = int(config.get('num_people', 3))
    people_speed_min = float(config.get('people_speed_min', 0.6))
    people_speed_max = float(config.get('people_speed_max', 1.2))

    # Infer input feature dimension using a temporary simulation
    tmp_sim = Simulation(
        corridor_width=corridor_width,
        door_side=door_side,
        num_people=num_people,
        people_speeds=[random.uniform(people_speed_min, people_speed_max) for _ in range(10)],
    )
    _ = tmp_sim.step(dt)
    input_dim = int(len(extract_nav_features(tmp_sim)))

    # Q-network
    hidden_size = int(config.get('hidden_size', 128))
    qnet = ThetaQNet(input_dim, num_actions, hidden=hidden_size).to(device)
    target_qnet = ThetaQNet(input_dim, num_actions, hidden=hidden_size).to(device)
    target_qnet.load_state_dict(qnet.state_dict())
    learning_rate = float(config.get('learning_rate', 1e-3))
    optimizer = optim.Adam(qnet.parameters(), lr=learning_rate)

    if use_wandb and wandb is not None:
        wandb.init(project=str(config.get('wandb_project', 'PredictiveDWA')),
                   name=run_name, config=config, allow_val_change=True)
        wandb.watch(qnet, log='all', log_freq=200)

    def epsilon_by_step(step):
        if step >= epsilon_decay_steps:
            return epsilon_end
        return epsilon_start + (epsilon_end - epsilon_start) * (step / epsilon_decay_steps)

    def select_action(state_feat: np.ndarray, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(num_actions)
        with torch.no_grad():
            s = torch.from_numpy(state_feat).unsqueeze(0).to(device)
            q = qnet(s)
            return int(q.argmax(dim=1).item())

    def soft_update():
        with torch.no_grad():
            for target_param, param in zip(target_qnet.parameters(), qnet.parameters()):
                target_param.data.mul_(1 - tau).add_(tau * param.data)

    def optimize_step():
        if len(buffer) < batch_size:
            return {}
        batch = random.sample(buffer, batch_size)
        states = torch.tensor(np.stack([b[0] for b in batch], axis=0), dtype=torch.float32, device=device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(np.stack([b[3] for b in batch], axis=0), dtype=torch.float32, device=device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device).unsqueeze(1)

        # Q(s,a)
        q_values = qnet(states).gather(1, actions)
        # target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_max_q = target_qnet(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + (1.0 - dones) * gamma * next_max_q

        loss = nn.functional.mse_loss(q_values, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        soft_update()
        return {'loss': float(loss.item())}

    # Training loop (episodes of the headless simulation)
    episodes = int(config.get('episodes', 50))
    max_steps = int(config.get('max_steps', 800))

    global_step = 0
    returns = []
    start_time = time.time()

    for ep in range(episodes):
        sim = Simulation(
            corridor_width=corridor_width,
            door_side=door_side,
            num_people=num_people,
            people_speeds=[random.uniform(people_speed_min, people_speed_max) for _ in range(10)],
        )

        # Reset progress tracker
        _, _, _ = sim.step(dt)  # advance once to initialize internal state
        robot_pos = sim.robot.position
        goal_pos = sim.robot.goal
        prev_dist = float(np.linalg.norm(goal_pos - robot_pos))

        episode_return = 0.0
        overlap_counts = {'none': 0, 'person': 0, 'door': 0, 'both': 0}

        for t in range(max_steps):
            # Build state features
            state_feat = extract_nav_features(sim)
            #print(f"state_feat: {state_feat}")

            # Epsilon-greedy pick of theta_range
            eps = epsilon_by_step(global_step)
            action_idx = select_action(state_feat, eps)
            theta_val = float(theta_actions[action_idx])

            # Set theta_range in the planner
            if hasattr(sim.robot, 'nav') and hasattr(sim.robot.nav, 'theta_range'):
                sim.robot.nav.theta_range = theta_val

            # Step simulation
            _, _, done_flag = sim.step(dt)

            # Reward
            reward, prev_dist, info = compute_reward(sim, prev_dist)
            episode_return += reward

            # Track overlap statistics
            overlap_counts[info['overlap_type']] += 1

            # Next features
            next_feat = extract_nav_features(sim)

            # Store transition
            buffer.append((state_feat, action_idx, reward, next_feat, float(done_flag)))

            # Learn
            metrics = optimize_step()

            global_step += 1
            if done_flag:
                break

        # Episode metrics
        total_steps = t + 1
        overlap_pct = {k: 100 * v / total_steps for k, v in overlap_counts.items()}
        returns.append(episode_return)

        print(f"Episode {ep+1}/{episodes} | Return: {episode_return:.2f} | Steps: {total_steps} | Eps: {eps:.2f}")
        print(f"  Overlaps - Free: {overlap_pct['none']:.1f}% | Person: {overlap_pct['person']:.1f}% | Door: {overlap_pct['door']:.1f}% | Both: {overlap_pct['both']:.1f}%")

        if use_wandb and wandb is not None:
            wandb.log({
                'episode': ep + 1,
                'return': episode_return,
                'steps': total_steps,
                'epsilon': eps,
                'overlap_free_pct': overlap_pct['none'],
                'overlap_person_pct': overlap_pct['person'],
                'overlap_door_pct': overlap_pct['door'],
                'overlap_both_pct': overlap_pct['both'],
                'elapsed_min': (time.time() - start_time) / 60.0,
            })

    avg_return = float(np.mean(returns)) if returns else 0.0

    # Save network
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(qnet.state_dict(), os.path.join('checkpoints', 'theta_qnet.pt'))
    print('Saved model to checkpoints/theta_qnet.pt')

    if use_wandb and wandb is not None:
        wandb.summary['avg_return'] = avg_return
        wandb.finish()

    return avg_return


def run_optuna(study_name: str, num_trials: int, base_config: Dict[str, Any]) -> None:
    if optuna is None:
        raise RuntimeError("Optuna is not installed. Please install optuna to use hyperparameter search.")

    def objective(trial: 'optuna.trial.Trial') -> float:
        # Suggest hyperparameters
        config = dict(base_config)
        config.update({
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),
            'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256, 384]),
            'gamma': trial.suggest_float('gamma', 0.90, 0.999),
            'tau': trial.suggest_float('tau', 0.001, 0.05, log=True),
            'epsilon_decay_steps': trial.suggest_int('epsilon_decay_steps', 2_000, 30_000, step=1000),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        })
        # Shorter training for objective
        config['episodes'] = int(base_config.get('optuna_episodes', 12))
        avg_return = train(config, use_wandb=bool(base_config.get('wandb_during_optuna', False)),
                           run_name=f"optuna_trial_{trial.number}")
        # We want to maximize avg_return
        return avg_return

    storage = None  # In-memory; customize with SQLite if desired
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage)
    study.optimize(objective, n_trials=num_trials, gc_after_trial=True)

    print("Best trial:")
    print(f"  value: {study.best_trial.value}")
    print("  params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train ThetaQNet with optional W&B logging and Optuna tuning')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--max-steps', type=int, default=800)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--eps-decay-steps', type=int, default=10_000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--buffer', type=int, default=50_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='PredictiveDWA')
    # Optuna options
    parser.add_argument('--optuna-trials', type=int, default=0)
    parser.add_argument('--optuna-study', type=str, default='theta_qnet_study')
    parser.add_argument('--optuna-episodes', type=int, default=12)
    parser.add_argument('--wandb-during-optuna', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    base_config: Dict[str, Any] = {
        'episodes': args.episodes,
        'max_steps': args.max_steps,
        'learning_rate': args.lr,
        'hidden_size': args.hidden,
        'gamma': args.gamma,
        'tau': args.tau,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay_steps': args.eps_decay_steps,
        'batch_size': args.batch_size,
        'buffer_capacity': args.buffer,
        'seed': args.seed,
        'wandb_project': args.wandb_project,
        # Env defaults
        'dt': 1/60.0,
        'corridor_width': 4.0,
        'door_side': 'right',
        'num_people': 3,
        'people_speed_min': 0.6,
        'people_speed_max': 1.2,
        # Optuna-specific
        'optuna_episodes': args.optuna_episodes,
        'wandb_during_optuna': args.wandb_during_optuna,
    }

    if args.optuna_trials and args.optuna_trials > 0:
        run_optuna(args.optuna_study, args.optuna_trials, base_config)
    else:
        train(base_config, use_wandb=args.use_wandb)


if __name__ == '__main__':
    main()


