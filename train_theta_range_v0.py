import os
import sys
import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from sim.sim import Simulation
from models.rl_theta_net import ThetaQNet


def extract_nav_features(sim) -> np.ndarray:
    """
    Minimal feature vector from current simulation state.
    Uses the same quantities already computed in Robot.get_navigation_info.
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
    input_dim = 8  # features extracted above
    qnet = ThetaQNet(input_dim, num_actions)
    target_qnet = ThetaQNet(input_dim, num_actions)
    target_qnet.load_state_dict(qnet.state_dict())
    optimizer = optim.Adam(qnet.parameters(), lr=1e-3)

    gamma = 0.99
    tau = 0.01
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = 10_000
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
        optimizer.step()
        soft_update()
        return {'loss': float(loss.item())}

    # Training loop (episodes of the headless simulation)
    episodes = 50
    max_steps = 800
    dt = 1/60.0
    global_step = 0

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
        overlap_counts = {'none': 0, 'person': 0, 'door': 0, 'both': 0}
        
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
            episode_return += reward
            
            # Track overlap statistics
            overlap_counts[info['overlap_type']] += 1

            # Next features
            next_feat = extract_nav_features(sim)

            # Store transition
            buffer.append((state_feat, action_idx, reward, next_feat, float(done_flag)))

            # Learn
            metrics = optimize()

            global_step += 1
            if done_flag:
                break

        # Calculate overlap percentages
        total_steps = t + 1
        overlap_pct = {k: 100 * v / total_steps for k, v in overlap_counts.items()}
        
        print(f"Episode {ep+1}/{episodes} | Return: {episode_return:.2f} | Steps: {total_steps} | Eps: {eps:.2f}")
        print(f"  Overlaps - Free: {overlap_pct['none']:.1f}% | Person: {overlap_pct['person']:.1f}% | Door: {overlap_pct['door']:.1f}% | Both: {overlap_pct['both']:.1f}%")

    # Save network
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(qnet.state_dict(), os.path.join('checkpoints', 'theta_qnet.pt'))
    print('Saved model to checkpoints/theta_qnet.pt')


if __name__ == '__main__':
    main()


