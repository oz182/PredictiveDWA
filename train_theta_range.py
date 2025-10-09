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
from rl_theta_net import ThetaQNet


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


def compute_reward(sim, progress_prev_dist: float) -> tuple[float, float, dict]:
    """
    Simple reward: progress to goal, small step penalty, collision penalty.
    Returns (reward, new_progress_distance, info)
    """
    robot_pos = sim.robot.position
    goal_pos = sim.robot.goal
    dist = float(np.linalg.norm(goal_pos - robot_pos))

    progress = (progress_prev_dist - dist)  # positive if moving towards goal
    reward = 1.0 * progress

    # time penalty
    reward += -0.005

    # collision penalty (count increment since last step)
    collisions = sim.collision_count
    info = {
        'distance': dist,
        'collisions': collisions,
    }
    # If a collision occurred in this step (heuristic: last history item within ~0.2s)
    if sim.collision_history:
        if abs(sim.collision_history[-1]['timestamp'] - __import__('datetime').datetime.now().timestamp()) < 0.2:
            reward += -5.0

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

            # Next features
            next_feat = extract_nav_features(sim)

            # Store transition
            buffer.append((state_feat, action_idx, reward, next_feat, float(done_flag)))

            # Learn
            metrics = optimize()

            global_step += 1
            if done_flag:
                break

        print(f"Episode {ep+1}/{episodes} | Return: {episode_return:.2f} | Steps: {t+1} | Eps: {eps:.2f}")

    # Save network
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(qnet.state_dict(), os.path.join('checkpoints', 'theta_qnet.pt'))
    print('Saved model to checkpoints/theta_qnet.pt')


if __name__ == '__main__':
    main()


