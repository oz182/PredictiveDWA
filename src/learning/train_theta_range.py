import os
import sys
import math
import random
import json
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sim.sim import Simulation
from learning.rl_theta_net import ThetaQNet


def extract_nav_features(sim) -> np.ndarray:
    """
    Minimal feature vector from current simulation state with normalization.
    Uses the same quantities already computed in Robot.get_navigation_info.
    """
    nav = sim.robot.get_navigation_info(2)
    # waypoint(2), door_position(2), door_angle(1), linear_velocity(1), angular_velocity(1), closest_obstacle_distance(1)
    feat = []
    
    # Normalize waypoint position (assuming corridor is ~10m long)
    waypoint = np.array(nav['waypoint'])
    feat.extend(list(waypoint / 10.0))
    
    # Normalize door position (assuming corridor is ~10m long)
    door_pos = np.array(nav['door_position'])
    feat.extend(list(door_pos / 10.0))
    
    # Normalize door angle to [-1, 1]
    feat.append(float(nav['door_angle']) / np.pi)
    
    # Normalize velocities (assuming max ~2 m/s)
    feat.append(float(nav['linear_velocity']) / 2.0)
    feat.append(float(nav['angular_velocity']) / 2.0)
    
    # Normalize closest obstacle distance (assuming max sensing ~5m)
    feat.append(float(nav['closest_obstacle_distance']) / 5.0)
    
    # Count people within velocity-based dynamic radius
    velocity_magnitude = np.linalg.norm(sim.robot.velocity)
    sensing_radius = np.clip(velocity_magnitude * 2.5, 1.5, 4.0)
    num_people_nearby = 0
    for person in sim.robot.people:
        if person.active:
            dist = np.linalg.norm(person.position - sim.robot.position)
            if dist <= sensing_radius:
                num_people_nearby += 1
    # Normalize by max expected people (e.g., 5)
    feat.append(float(num_people_nearby) / 5.0)
    
    # Forward proxemic cost from costmap (already 0-1)
    forward_cost = get_forward_proxemic_cost(sim)
    feat.append(float(forward_cost))
    
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


def compute_proxemic_penetration(sim, person) -> float:
    """
    Calculate continuous penetration depth into a person's proxemic ellipse.
    Returns 0.0 if outside, value 0.0-1.0 if inside (normalized by ellipse size).
    """
    robot_pos = sim.robot.position
    robot_radius = sim.robot.radius
    
    # Get person's proxemic ellipse parameters
    axes = getattr(person, 'proxemic_axes', np.array([person.radius, person.radius], dtype=float)).astype(float)
    a = max(float(axes[0]), 1e-4)  # semi-minor axis
    b = max(float(axes[1]), 1e-4)  # semi-major axis
    
    # Ellipse offset
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
    
    # Apply offset
    local_x_shifted = local_x + ellipse_offset
    
    # Check position relative to ellipse
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
    
    # Account for robot radius
    clearance = dist_to_boundary - robot_radius
    
    # Normalize penetration: 0.0 if outside, increases as robot goes deeper
    if clearance >= 0:
        return 0.0
    else:
        # Normalize by ellipse size
        penetration = -clearance / max(a, b)
        return min(penetration, 1.0)


def compute_door_penetration(sim) -> float:
    """
    Calculate continuous penetration into door halo.
    Returns 0.0 if outside, value 0.0-1.0 if inside (normalized).
    """
    robot_pos = sim.robot.position
    robot_radius = sim.robot.radius
    
    if not hasattr(sim.robot, 'door_position') or not hasattr(sim.robot, 'corridor_bounds'):
        return 0.0
    
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
    
    # Check if on inward-facing side
    dot_product = n_world[0] * v_x + n_world[1] * v_y
    
    if dot_product <= 0.0:
        return 0.0  # Outside semicircle
    
    # Calculate penetration
    clearance = dist_to_door - robot_radius - door_inflation_radius
    
    if clearance >= 0:
        return 0.0
    else:
        # Normalize by halo radius
        penetration = -clearance / door_inflation_radius
        return min(penetration, 1.0)


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


def compute_reward(sim, progress_prev_dist: float) -> tuple[float, float, dict]:
    """
    Reward based on progress, continuous proxemic penetration, and goal achievement.
    Returns (reward, new_progress_distance, info)
    """
    robot_pos = sim.robot.position
    goal_pos = sim.robot.goal
    dist = float(np.linalg.norm(goal_pos - robot_pos))

    # Progress reward: scaled up to be more significant
    progress = (progress_prev_dist - dist)
    reward = 5.0 * progress  # Increased from 1.0 to make progress more rewarding

    # Compute continuous person proxemic penetration
    max_penetration = 0.0
    for person in sim.robot.people:
        if not person.active:
            continue
        penetration = compute_proxemic_penetration(sim, person)
        max_penetration = max(max_penetration, penetration)
    
    # Graduated penalty for person proxemics (exponential) - reduced magnitude
    if max_penetration > 0.0:
        reward += -1.0 * (max_penetration ** 1.5)  # Reduced from -2.0
    else:
        # Small bonus for staying in free space
        reward += 0.05  # Reduced from 0.2 to avoid inflating rewards
    
    # Compute continuous door penetration
    door_penetration = compute_door_penetration(sim)
    
    # Graduated penalty for door (exponential) - reduced
    if door_penetration > 0.0:
        reward += -0.5 * (door_penetration ** 1.5)  # Reduced from -1.0

    # Reduced time penalty to not dominate the reward signal
    reward += -0.001  # Reduced from -0.005

    # Collision penalty - reduced to be less catastrophic
    collisions = sim.collision_count
    info = {
        'distance': dist,
        'collisions': collisions,
        'max_person_penetration': max_penetration,
        'door_penetration': door_penetration,
    }
    # If a collision occurred in this step (heuristic: last history item within ~0.2s)
    if sim.collision_history:
        if abs(sim.collision_history[-1]['timestamp'] - __import__('datetime').datetime.now().timestamp()) < 0.2:
            reward += -2.0  # Reduced from -5.0

    # Graduated goal proximity reward to guide the agent
    if dist < 1.0:
        reward += 50.0  # Increased from 25.0 - major success!
    elif dist < 2.0:
        reward += 10.0  # New: intermediate reward for getting close
    elif dist < 3.0:
        reward += 5.0   # New: smaller reward for approaching
    
    # Additional penalty for moving away from goal (negative progress)
    if progress < -0.1:
        reward += -1.0

    return reward, dist, info


def plot_training_results(episode_returns: list, episodes: int):
    """
    Plot cumulative rewards vs episode number with smoothed average.
    
    Args:
        episode_returns: List of cumulative rewards per episode
        episodes: Total number of episodes
    """
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Episode numbers
    episode_nums = np.arange(1, len(episode_returns) + 1)
    
    # Plot raw episode returns
    plt.plot(episode_nums, episode_returns, alpha=0.3, color='blue', linewidth=1, label='Episode Return')
    
    # Compute moving average for smoothing
    window_size = min(20, len(episode_returns) // 5)  # Use 20 episodes or 1/5 of total, whichever is smaller
    if window_size < 1:
        window_size = 1
    
    moving_avg = []
    for i in range(len(episode_returns)):
        start_idx = max(0, i - window_size + 1)
        window = episode_returns[start_idx:i+1]
        moving_avg.append(np.mean(window))
    
    # Plot smoothed average
    plt.plot(episode_nums, moving_avg, color='red', linewidth=2, label=f'Moving Average (window={window_size})')
    
    # Add labels and formatting
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Cumulative Reward', fontsize=12)
    plt.title('Training Progress: Cumulative Reward vs Episode', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    os.makedirs('checkpoints', exist_ok=True)
    plot_path = os.path.join('checkpoints', 'training_progress.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Training plot saved to {plot_path}")
    
    # Display plot
    plt.tight_layout()
    plt.show()


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Discrete theta_range actions (keep tiny and intuitive)
    theta_actions = np.array([
        math.radians(10),
        math.radians(20),
        math.radians(30),
        math.radians(45),
    ], dtype=np.float32)
    num_actions = len(theta_actions)

    # Q-network
    input_dim = 10  # features extracted above (8 original + num_people_nearby + forward_proxemic_cost)
    qnet = ThetaQNet(input_dim, num_actions)
    target_qnet = ThetaQNet(input_dim, num_actions)
    target_qnet.load_state_dict(qnet.state_dict())
    optimizer = optim.Adam(qnet.parameters(), lr=5e-5)  # Reduced from 1e-3 for stability

    gamma = 0.99
    tau = 0.01
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = 100_000  # Increased from 10k to allow more exploration
    batch_size = 64
    buffer = deque(maxlen=50_000)

    def epsilon_by_step(step):
        if step >= epsilon_decay_steps:
            return epsilon_end
        return epsilon_start + (epsilon_end - epsilon_start) * (step / epsilon_decay_steps)

    def select_action(state_feat: np.ndarray, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(num_actions)
        with torch.no_grad():
            s = torch.from_numpy(state_feat).unsqueeze(0)
            q = qnet(s)
            return int(q.argmax(dim=1).item())

    def soft_update():
        with torch.no_grad():
            for tp, p in zip(target_qnet.parameters(), qnet.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

    def optimize():
        if len(buffer) < batch_size:
            return {}
        batch = random.sample(buffer, batch_size)
        s = torch.tensor(np.stack([b[0] for b in batch], axis=0), dtype=torch.float32)
        a = torch.tensor([b[1] for b in batch], dtype=torch.long).unsqueeze(1)
        r = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1)
        s2 = torch.tensor(np.stack([b[3] for b in batch], axis=0), dtype=torch.float32)
        d = torch.tensor([b[4] for b in batch], dtype=torch.float32).unsqueeze(1)

        # Q(s,a)
        q = qnet(s).gather(1, a)
        # target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            q2 = target_qnet(s2).max(dim=1, keepdim=True)[0]
            target = r + (1.0 - d) * gamma * q2

        loss = nn.functional.mse_loss(q, target)
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(qnet.parameters(), max_norm=10.0)
        optimizer.step()
        soft_update()
        return {'loss': float(loss.item())}

    # Training loop (episodes of the headless simulation)
    episodes = 80
    max_steps = 800
    dt = 1/60.0
    global_step = 0
    episode_returns = []  # Track cumulative rewards for plotting
    warmup_steps = 1000  # Fill buffer before training starts

    for ep in range(episodes):
        sim = Simulation(corridor_width=4.0, door_side='right', num_people=3,
                         people_speeds=[random.uniform(0.6, 1.2) for _ in range(10)])

        # Reset progress tracker
        _, _, _ = sim.step(dt)  # advance once to initialize internal state
        robot_pos = sim.robot.position
        goal_pos = sim.robot.goal
        prev_dist = float(np.linalg.norm(goal_pos - robot_pos))

        episode_return = 0.0
        done = False
        max_person_pen = 0.0
        max_door_pen = 0.0
        
        for t in range(max_steps):
            # Build state features
            state_feat = extract_nav_features(sim)

            # Epsilon-greedy pick of theta_range
            eps = epsilon_by_step(global_step)
            action_idx = select_action(state_feat, eps)
            theta_val = float(theta_actions[action_idx])

            # Set theta_range in the planner
            if hasattr(sim.robot, 'nav') and hasattr(sim.robot.nav, 'theta_range'):
                sim.robot.nav.theta_range = theta_val

            # Step simulation
            next_state, _, done_flag = sim.step(dt)

            # Reward
            reward, prev_dist, info = compute_reward(sim, prev_dist)
            
            # Clip reward to prevent extreme outliers from destabilizing learning
            reward_clipped = np.clip(reward, -10.0, 50.0)
            episode_return += reward  # Track unclipped for monitoring
            
            # Track max penetration values
            max_person_pen = max(max_person_pen, info['max_person_penetration'])
            max_door_pen = max(max_door_pen, info['door_penetration'])

            # Next features
            next_feat = extract_nav_features(sim)

            # Store transition with clipped reward
            buffer.append((state_feat, action_idx, reward_clipped, next_feat, float(done_flag)))

            # Learn (only after warmup period)
            if global_step >= warmup_steps:
                metrics = optimize()
            else:
                metrics = {}

            global_step += 1
            if done_flag:
                break

        # Print episode summary
        total_steps = t + 1
        print(f"Episode {ep+1}/{episodes} | Return: {episode_return:.2f} | Steps: {total_steps} | Eps: {eps:.2f}")
        print(f"  Max Penetrations - Person: {max_person_pen:.3f} | Door: {max_door_pen:.3f}")
        
        # Store episode return for plotting
        episode_returns.append(episode_return)

    # Plot training progress
    print("\nGenerating training plots...")
    plot_training_results(episode_returns, episodes)

    # Save network
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(qnet.state_dict(), os.path.join('checkpoints', 'theta_qnet.pt'))
    print('Saved model to checkpoints/theta_qnet.pt')
    
    # Save hyperparameters
    hyperparams = {
        'learning_rate': 5e-5,
        'gamma': gamma,
        'tau': tau,
        'epsilon_start': epsilon_start,
        'epsilon_end': epsilon_end,
        'epsilon_decay_steps': epsilon_decay_steps,
        'batch_size': batch_size,
        'buffer_size': 50_000,
        'hidden_dim': 256,
        'episodes': episodes,
        'max_steps_per_episode': max_steps,
        'warmup_steps': warmup_steps,
        'reward_clip_range': [-10.0, 50.0],
        'gradient_clip_norm': 10.0,
        'theta_actions_degrees': [10, 20, 30, 45],
        'feature_normalization': True,
    }
    with open(os.path.join('checkpoints', 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparams, f, indent=2)
    print('Saved hyperparameters to checkpoints/hyperparameters.json')


if __name__ == '__main__':
    main()


