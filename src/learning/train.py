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

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sim.sim import Simulation
from agents.ppo import PPO

# Optional third-party integrations

import wandb  
import optuna  

def get_curriculum_params(episode: int, total_episodes: int) -> dict:
    """
    Get curriculum learning parameters based on episode number.
    
    Gradually increases difficulty:
    - Early training: narrow ranges around known working values
    - Mid training: expand ranges
    - Late training: full diversity
    
    Args:
        episode: Current episode number (0-indexed)
        total_episodes: Total number of training episodes
        
    Returns:
        dict with keys:
            - door_halo_radius_range: (min, max) for radius sampling
            - door_position_x_range: (min, max) for door x position
            - randomize_door_side: whether to randomize door side
    """
    # Calculate progress (0.0 to 1.0)
    progress = episode / max(total_episodes, 1)
    
    # Stage 1: Easy (first 20% of training)
    if progress < 0.2:
        return {
            'door_halo_radius_range': (1.6, 2.0),  # Narrow range around known value
            'door_position_x_range': (7.0, 9.0),   # Fixed around 40% of 20m corridor
            'randomize_door_side': False,          # Keep door on right side
            'stage': 'easy'
        }
    
    # Stage 2: Medium (20% - 50% of training)
    elif progress < 0.5:
        return {
            'door_halo_radius_range': (1.2, 2.3),  # Expanded range
            'door_position_x_range': (6.0, 11.0),  # More variation in position
            'randomize_door_side': False,          # Still on right side
            'stage': 'medium'
        }
    
    # Stage 3: Hard (50% - 80% of training)
    elif progress < 0.8:
        return {
            'door_halo_radius_range': (0.8, 2.5),  # Full range
            'door_position_x_range': (5.0, 13.0),  # Wide variation
            'randomize_door_side': True,           # Randomize left/right
            'stage': 'hard'
        }
    
    # Stage 4: Expert (final 20% of training)
    else:
        return {
            'door_halo_radius_range': (0.8, 2.5),  # Full range
            'door_position_x_range': (5.0, 13.0),  # Full range (25%-65% of corridor)
            'randomize_door_side': True,           # Fully randomized
            'stage': 'expert'
        }

def extract_nav_features(sim) -> np.ndarray:
    """
    Feature vector from current simulation state (robot-centric in world frame).
    All features are normalized to roughly [-1, 1] or [0, 1] range for stable training.
    
    Includes:
      - relative goal position in world frame: (goal_dx, goal_dy) / 10.0
      - relative door position in world frame: (door_dx, door_dy) / 10.0
      - three closest people relative positions wrt robot: [(dx, dy) x 3] / 5.0
      - distances to corridor left and right boundaries / 2.0
      - door_inflation_radius / 3.0
    """
    robot_pos = np.asarray(sim.robot.position, dtype=float)
    goal_pos = np.asarray(sim.robot.goal, dtype=float)
    door_pos = np.asarray(sim.robot.door_position, dtype=float)
    
    # Normalization constants
    POSITION_SCALE = 10.0  # Max expected relative position ~20m corridor
    PEOPLE_SCALE = 5.0     # People typically within 5m
    WALL_SCALE = 2.0       # Corridor width ~4m, so max wall dist ~2m
    DOOR_RADIUS_SCALE = 3.0
    
    # Default value for missing people (normalized)
    large_val_normalized = 2.0  # Represents "far away"

    # Relative goal and door positions (world frame), normalized
    goal_dx = float(goal_pos[0] - robot_pos[0]) / POSITION_SCALE
    goal_dy = float(goal_pos[1] - robot_pos[1]) / POSITION_SCALE
    door_dx = float(door_pos[0] - robot_pos[0]) / POSITION_SCALE
    door_dy = float(door_pos[1] - robot_pos[1]) / POSITION_SCALE

    # Compute three closest people relative positions (dx, dy) in world frame
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
    # Take up to 3, pad if fewer (normalized)
    rel_feats: list[float] = []
    for i in range(3):
        if i < len(rel_people):
            _, dx, dy = rel_people[i]
            rel_feats.extend([dx / PEOPLE_SCALE, dy / PEOPLE_SCALE])
        else:
            rel_feats.extend([large_val_normalized, large_val_normalized])

    # Distances to corridor left/right sides using world y-bounds (normalized)
    dist_left = large_val_normalized
    dist_right = large_val_normalized
    if hasattr(sim.robot, 'corridor_bounds'):
        bounds = sim.robot.corridor_bounds
        y_min = float(bounds['y_min'])
        y_max = float(bounds['y_max'])
        y = float(robot_pos[1])
        dist_left = max(0.0, y - y_min) / WALL_SCALE
        dist_right = max(0.0, y_max - y) / WALL_SCALE

    # Door inflation radius (normalized)
    door_inflation_radius = float(getattr(sim.robot.global_planner, 'door_halo_radius', 1.0))
    door_inflation_radius_normalized = door_inflation_radius / DOOR_RADIUS_SCALE

    feat: list[float] = []
    # Goal relative position (most important)
    feat.append(goal_dx)
    feat.append(goal_dy)
    # Door relative position
    feat.append(door_dx)
    feat.append(door_dy)
    # People relative positions
    feat.extend(rel_feats)
    # Wall distances
    feat.append(dist_left)
    feat.append(dist_right)
    # Door inflation radius
    feat.append(door_inflation_radius_normalized)

    return np.asarray(feat, dtype=np.float32)


def extract_nav_features_v0(sim) -> np.ndarray:
    """
    Feature vector from current simulation state.
    Includes:
      - goal_angle(1), goal_distance(1) - CRITICAL for navigation
      - waypoint(2), door_position(2), door_angle(1)
      - linear_velocity(1), angular_velocity(1)
      - three closest people relative positions wrt robot: [(dx, dy) x 3] (pad with large value if <3)
      - distances to corridor left and right boundaries (y - y_min, y_max - y)
    """
    nav = sim.robot.get_navigation_info(2)

    robot_pos = np.asarray(sim.robot.position, dtype=float)
    robot_orientation = float(getattr(sim.robot, 'orientation', 0.0))
    goal_pos = np.asarray(sim.robot.goal, dtype=float)
    large_val = 10.0

    # Goal direction in robot frame (CRITICAL for agent to know where to go!)
    goal_vec = goal_pos - robot_pos
    goal_angle_world = math.atan2(goal_vec[1], goal_vec[0])
    goal_angle_robot = goal_angle_world - robot_orientation
    # Normalize to [-π, π]
    while goal_angle_robot > math.pi:
        goal_angle_robot -= 2 * math.pi
    while goal_angle_robot < -math.pi:
        goal_angle_robot += 2 * math.pi
    goal_dist = float(np.linalg.norm(goal_vec))

    # Compute three closest people relative positions (dx, dy) in robot frame
    rel_people: list[tuple[float, float, float]] = []  # (dist, dx, dy)
    if hasattr(sim.robot, 'people'):
        for person in sim.robot.people:
            if getattr(person, 'active', True):
                p = np.asarray(person.position, dtype=float)
                # Transform to robot frame
                rel_x = p[0] - robot_pos[0]
                rel_y = p[1] - robot_pos[1]
                # Rotate into robot's reference frame
                cos_theta = math.cos(-robot_orientation)
                sin_theta = math.sin(-robot_orientation)
                rel_x_robot = rel_x * cos_theta - rel_y * sin_theta
                rel_y_robot = rel_x * sin_theta + rel_y * cos_theta
                d = math.hypot(rel_x_robot, rel_y_robot)
                rel_people.append((d, rel_x_robot, rel_y_robot))
    
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
    # Add goal information first (most important!)
    feat.append(float(goal_angle_robot))
    feat.append(float(goal_dist))
    # Rest of features
    feat.extend(list(map(float, nav['waypoint'])))
    feat.extend(list(map(float, nav['door_position'])))
    feat.append(float(nav['door_angle']))
    feat.append(float(nav['linear_velocity']))
    feat.append(float(nav['angular_velocity']))
    feat.extend(rel_feats)              # (dx, dy) x 3 in robot frame
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


def compute_min_obstacle_distance(sim) -> tuple[float, float]:
    """
    Compute minimum distances to obstacles for reward shaping.
    
    Returns:
        (min_person_distance, min_door_distance) - distances to closest obstacles
    """
    robot_pos = sim.robot.position
    robot_radius = sim.robot.radius
    
    # Minimum distance to any person's proxemic zone
    min_person_dist = float('inf')
    for person in sim.robot.people:
        if not person.active:
            continue
        
        # Get person's proxemic ellipse parameters
        axes = getattr(person, 'proxemic_axes', np.array([person.radius, person.radius], dtype=float)).astype(float)
        a = max(float(axes[0]), 1e-4)  # semi-minor axis (width)
        b = max(float(axes[1]), 1e-4)  # semi-major axis (length)
        
        # Simple distance to person center (approximation for shaping)
        dist_to_person = float(np.linalg.norm(robot_pos - person.position)) - robot_radius - max(a, b)
        min_person_dist = min(min_person_dist, dist_to_person)
    
    # Distance to door
    min_door_dist = float('inf')
    if hasattr(sim.robot, 'door_position') and hasattr(sim.robot, 'corridor_bounds'):
        door_pos = np.array(sim.robot.door_position, dtype=float)
        bounds = sim.robot.corridor_bounds
        door_inflation_radius = float(getattr(sim.robot.global_planner, 'door_halo_radius', 1.0))
        
        # Vector from door to robot
        v = robot_pos - door_pos
        dist_to_door = float(np.linalg.norm(v))
        
        # Check if robot is on the inward-facing side
        corridor_mid_y = (bounds['y_min'] + bounds['y_max']) * 0.5
        door_side = "left" if door_pos[1] < corridor_mid_y else "right"
        n_world = np.array([0.0, 1.0]) if door_side == "left" else np.array([0.0, -1.0])
        
        dot_product = np.dot(n_world, v)
        if dot_product > 0:  # Robot is on the inward-facing side
            min_door_dist = dist_to_door - door_inflation_radius - robot_radius
        else:
            min_door_dist = float('inf')  # Not on the relevant side
    
    return min_person_dist, min_door_dist


def compute_reward(sim, progress_prev_dist: float, offset: float = 0.0, 
                   prev_collision_count: int = 0) -> tuple[float, float, int, dict]:
    """
    Reward function with proper credit assignment for obstacle avoidance.
    
    Components:
    1. Progress reward: Encourage moving toward goal
    2. Obstacle avoidance: Positive reward for staying clear, penalty for overlap
    3. Distance-based shaping: Continuous signal as robot approaches obstacles
    4. Collision penalty: Large penalty for actual collisions
    
    Returns (reward, new_distance, new_collision_count, info)
    """
    robot_pos = sim.robot.position
    goal_pos = sim.robot.goal
    dist = float(np.linalg.norm(goal_pos - robot_pos))

    reward = 0.0
    
    # ==========================================================================
    # 1. PROGRESS REWARD - Encourage moving toward goal
    # ==========================================================================
    progress = progress_prev_dist - dist  # Positive if moving toward goal
    progress_reward = progress * 0.5  # Scale factor for progress
    reward += progress_reward
    
    # ==========================================================================
    # 2. OBSTACLE OVERLAP REWARD - Primary avoidance signal
    # ==========================================================================
    overlap_info = check_robot_overlap(sim)
    overlap_type = overlap_info['overlap_type']
    
    if overlap_type == 'none':
        # POSITIVE reward for staying in free space
        overlap_reward = 0.1
    elif overlap_type == 'person':
        # Penalty for being in person's proxemic zone
        overlap_reward = -1.0
    elif overlap_type == 'door':
        # Penalty for being in door's inflation zone
        overlap_reward = -1.0
    elif overlap_type == 'both':
        # Larger penalty for being in both zones
        overlap_reward = -2.0
    else:
        overlap_reward = 0.0
    
    reward += overlap_reward
    
    # ==========================================================================
    # 3. DISTANCE-BASED SHAPING - Continuous signal before overlap occurs
    # ==========================================================================
    min_person_dist, min_door_dist = compute_min_obstacle_distance(sim)
    
    # Shaping zone: give continuous reward based on proximity to obstacles
    # This helps the agent learn to keep distance BEFORE entering overlap zone
    shaping_threshold = 2.0  # Start shaping within 2m of obstacles
    
    shaping_reward = 0.0
    if min_person_dist < shaping_threshold and min_person_dist > 0:
        # Closer to person = more negative (linear shaping)
        shaping_reward += -0.1 * (1.0 - min_person_dist / shaping_threshold)
    
    if min_door_dist < shaping_threshold and min_door_dist > 0:
        # Closer to door = more negative
        shaping_reward += -0.1 * (1.0 - min_door_dist / shaping_threshold)
    
    reward += shaping_reward
    
    # ==========================================================================
    # 4. COLLISION PENALTY - Large penalty for actual physical collisions
    # ==========================================================================
    current_collision_count = sim.collision_count
    new_collisions = current_collision_count - prev_collision_count
    
    if new_collisions > 0:
        reward += -5.0 * new_collisions  # Large penalty per collision
    
    # ==========================================================================
    # 5. GOAL REACHED BONUS
    # ==========================================================================
    if dist < 1.0:  # Goal reached
        reward += 10.0  # Large positive reward for completing the task
    
    # Build info dict
    info = {
        'distance': dist,
        'collisions': current_collision_count,
        'new_collisions': new_collisions,
        'overlap_type': overlap_type,
        'person_overlap': overlap_info['person_overlap'],
        'door_overlap': overlap_info['door_overlap'],
        'min_person_dist': min_person_dist if min_person_dist != float('inf') else -1.0,
        'min_door_dist': min_door_dist if min_door_dist != float('inf') else -1.0,
        'offset': offset,
        'offset_abs': abs(offset),
        'progress_reward': progress_reward,
        'overlap_reward': overlap_reward,
        'shaping_reward': shaping_reward,
    }

    return reward, dist, current_collision_count, info


def train(config: Dict[str, Any], use_wandb: bool = True, run_name: Optional[str] = None) -> float:
    seed = int(config.get('seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Action space: single offset value for heading adjustment
    num_actions = 1

    # Environment / training hyperparameters (algorithm-agnostic where possible)
    gamma = float(config.get('gamma', 0.99))
    # PPO-specific (but genericizable via agent config)
    k_epochs = int(config.get('k_epochs', 10))
    eps_clip = float(config.get('eps_clip', 0.2))
    update_timestep = int(config.get('update_timestep', 2000))
    action_select_interval = int(config.get('action_select_interval', 1))  # select new action every N steps
    lr_actor = float(config.get('lr_actor', config.get('learning_rate', 1e-3)))
    lr_critic = float(config.get('lr_critic', config.get('learning_rate', 1e-3)))

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
        door_side='right',  # Fixed for initialization
        num_people=num_people,
        people_speeds=[random.uniform(people_speed_min, people_speed_max) for _ in range(10)],
        door_halo_radius=1.8,  # Fixed for initialization
        door_position_x=8.0,  # Fixed for initialization
    )
    _ = tmp_sim.step(dt)
    input_dim = int(len(extract_nav_features(tmp_sim)))

    # Build agent (PPO with discrete action space)
    hidden_size = int(config.get('hidden_size', 128))
    agent = PPO(
        state_dim=input_dim,
        action_dim=num_actions,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        K_epochs=k_epochs,
        eps_clip=eps_clip,
        has_continuous_action_space=True,
        action_std_init=0.6,
    )

    if use_wandb and wandb is not None:
        wandb.init(project=str(config.get('wandb_project', 'PredictiveDWA')),
                   name=run_name, config=config, allow_val_change=True)
        # Watch PPO networks if available
        try:
            wandb.watch(agent.policy.actor, log='all', log_freq=200)
            wandb.watch(agent.policy.critic, log='all', log_freq=200)
        except Exception:
            pass

    # PPO uses its own rollout buffer; updates are triggered by timesteps
    time_step = 0

    # Training loop (episodes of the headless simulation)
    episodes = int(config.get('episodes', 50))
    max_steps = int(config.get('max_steps', 3000))

    global_step = 0
    returns = []
    start_time = time.time()

    for ep in range(episodes):
        # Get curriculum parameters for this episode
        curriculum = get_curriculum_params(ep, episodes)
        
        # Sample door parameters from curriculum ranges
        door_halo_radius = random.uniform(*curriculum['door_halo_radius_range'])
        door_position_x = random.uniform(*curriculum['door_position_x_range'])
        door_side_param = None if curriculum['randomize_door_side'] else 'right'
        
        sim = Simulation(
            corridor_width=corridor_width,
            door_side=door_side_param,
            num_people=num_people,
            people_speeds=[random.uniform(people_speed_min, people_speed_max) for _ in range(10)],
            door_halo_radius=door_halo_radius,
            door_position_x=door_position_x,
        )

        # Reset progress tracker
        _, _, _ = sim.step(dt)  # advance once to initialize internal state
        robot_pos = sim.robot.position
        goal_pos = sim.robot.goal
        prev_dist = float(np.linalg.norm(goal_pos - robot_pos))
        prev_collision_count = 0  # Track collisions for delta calculation

        episode_return = 0.0
        overlap_counts = {'none': 0, 'person': 0, 'door': 0, 'both': 0}
        offset_history = []  # Track offsets for analysis
        reward_components = {'progress': 0.0, 'overlap': 0.0, 'shaping': 0.0}  # Track reward breakdown

        # Storage for action repetition
        prev_offset = None

        for t in range(max_steps):
            # Build state features
            state_feat = extract_nav_features(sim)

            # Decide whether to sample a new action or reuse previous one
            if (t % action_select_interval == 0) or (prev_offset is None):
                # Sample new action via policy (also records state/action/logprob/value in buffer)
                offset_normalized = agent.select_action(state_feat)  # Returns array
                prev_offset = float(offset_normalized[0])
                prev_offset = max(-1.0, min(1.0, prev_offset))
            else:
                # Reuse previous action but still log this step into PPO buffer
                offset_normalized_value = prev_offset
                # IMPORTANT: use policy_old for consistency with PPO's importance sampling
                policy_device = next(agent.policy_old.parameters()).device
                state_tensor_no_batch = torch.FloatTensor(state_feat).to(policy_device)
                state_tensor_batch = state_tensor_no_batch.unsqueeze(0)
                action_tensor_batch = torch.tensor([[offset_normalized_value]], dtype=torch.float32, device=policy_device)
                with torch.no_grad():
                    old_logprob, state_value, _ = agent.policy_old.evaluate(state_tensor_batch, action_tensor_batch)
                # Match shapes used by select_action: state (no batch), action ([1, action_dim])
                agent.buffer.states.append(state_tensor_no_batch)
                agent.buffer.actions.append(action_tensor_batch)
                agent.buffer.logprobs.append(old_logprob.detach())
                agent.buffer.state_values.append(state_value.squeeze(0).detach())

            # Scale offset from [-1, 1] to [-π/6, π/6] (±30 degrees)
            max_offset = math.pi / 3  # 60 degrees
            offset = prev_offset * max_offset
            
            # Set offset in the planner (agent's correction to path heading)
            sim.robot.nav.agent_offset = offset
            offset_history.append(abs(offset))

            # Step simulation
            _, _, done_flag = sim.step(dt)

            # Reward with all components
            reward, prev_dist, prev_collision_count, info = compute_reward(
                sim, prev_dist, offset, prev_collision_count
            )
            episode_return += reward

            # Track overlap statistics
            overlap_counts[info['overlap_type']] += 1
            
            # Track reward component breakdown
            reward_components['progress'] += info['progress_reward']
            reward_components['overlap'] += info['overlap_reward']
            reward_components['shaping'] += info['shaping_reward']

            # PPO bookkeeping
            agent.buffer.rewards.append(float(reward))
            agent.buffer.is_terminals.append(bool(done_flag))

            # Trigger update at fixed timesteps
            time_step += 1
            if time_step % update_timestep == 0:
                update_stats = agent.update()
                if use_wandb and wandb is not None and update_stats is not None:
                    wandb.log({
                        'ppo_loss': update_stats['loss'],
                        'ppo_policy_loss': update_stats['policy_loss'],
                        'ppo_value_loss': update_stats['value_loss'],
                        'ppo_entropy': update_stats['entropy'],
                        'train_time_step': time_step,
                    })

            if time_step % (10 * update_timestep) == 0:  # e.g. every 10 updates ####### ADD ACTION STD DECAY HERE #######
                agent.decay_action_std(action_std_decay_rate=0.01, min_action_std=0.05)

            global_step += 1
            if done_flag:
                # Update at episode boundary as well
                update_stats = agent.update()
                if use_wandb and wandb is not None and update_stats is not None:
                    wandb.log({
                        'ppo_loss': update_stats['loss'],
                        'ppo_policy_loss': update_stats['policy_loss'],
                        'ppo_value_loss': update_stats['value_loss'],
                        'ppo_entropy': update_stats['entropy'],
                        'train_time_step': time_step,
                    })
                break

        # Episode metrics
        total_steps = t + 1
        overlap_pct = {k: 100 * v / total_steps for k, v in overlap_counts.items()}
        returns.append(episode_return)
        
        # Offset statistics
        avg_abs_offset = np.mean(offset_history) if offset_history else 0.0
        max_abs_offset = np.max(offset_history) if offset_history else 0.0
        avg_abs_offset_deg = avg_abs_offset * 180 / math.pi
        max_abs_offset_deg = max_abs_offset * 180 / math.pi

        print(f"Episode {ep+1}/{episodes} | Return: {episode_return:.2f} | Steps: {total_steps}")
        print(f"  Curriculum Stage: {curriculum['stage'].upper()} | Door Radius: {door_halo_radius:.2f}m | Door Pos: {door_position_x:.1f}m | Side: {sim.door_side}")
        print(f"  Rewards - Progress: {reward_components['progress']:.2f} | Overlap: {reward_components['overlap']:.2f} | Shaping: {reward_components['shaping']:.2f}")
        print(f"  Overlaps - Free: {overlap_pct['none']:.1f}% | Person: {overlap_pct['person']:.1f}% | Door: {overlap_pct['door']:.1f}% | Both: {overlap_pct['both']:.1f}%")
        print(f"  Offsets - Avg: {avg_abs_offset_deg:.1f}° | Max: {max_abs_offset_deg:.1f}° | (range: ±60°)")

        if use_wandb and wandb is not None:
            wandb.log({
                'episode': ep + 1,
                'return': episode_return,
                'steps': total_steps,
                # Reward breakdown
                'reward_progress': reward_components['progress'],
                'reward_overlap': reward_components['overlap'],
                'reward_shaping': reward_components['shaping'],
                # Overlap statistics
                'overlap_free_pct': overlap_pct['none'],
                'overlap_person_pct': overlap_pct['person'],
                'overlap_door_pct': overlap_pct['door'],
                'overlap_both_pct': overlap_pct['both'],
                # Offset statistics
                'avg_abs_offset_deg': avg_abs_offset_deg,
                'max_abs_offset_deg': max_abs_offset_deg,
                'elapsed_min': (time.time() - start_time) / 60.0,
                # Curriculum parameters
                'curriculum_stage': curriculum['stage'],
                'door_halo_radius': door_halo_radius,
                'door_position_x': door_position_x,
                'door_side': sim.door_side,
            })

    avg_return = float(np.mean(returns)) if returns else 0.0

    # Save policy
    os.makedirs('checkpoints', exist_ok=True)
    agent.save(os.path.join('checkpoints', 'theta_qnet.pt'))
    print('Saved PPO policy to checkpoints/theta_qnet.pt')

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
            'lr_actor': trial.suggest_float('lr_actor', 1e-5, 5e-3, log=True),
            'lr_critic': trial.suggest_float('lr_critic', 1e-5, 5e-3, log=True),
            'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256, 384]),
            'gamma': trial.suggest_float('gamma', 0.90, 0.999),
            'k_epochs': trial.suggest_int('k_epochs', 5, 20),
            'eps_clip': trial.suggest_float('eps_clip', 0.1, 0.3),
            'update_timestep': trial.suggest_int('update_timestep', 1000, 5000, step=500),
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
    parser = argparse.ArgumentParser(description='Train navigation policy with PPO (agent-agnostic structure)')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr-actor', type=float, default=None)
    parser.add_argument('--lr-critic', type=float, default=None)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--k-epochs', type=int, default=10)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--update-timestep', type=int, default=2000)
    parser.add_argument('--action-select-interval', type=int, default=1, help='Select a new action every N steps (default 1 = every step)')
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
        'lr_actor': args.lr_actor if args.lr_actor is not None else args.lr,
        'lr_critic': args.lr_critic if args.lr_critic is not None else args.lr,
        'hidden_size': args.hidden,
        'gamma': args.gamma,
        'k_epochs': args.k_epochs,
        'eps_clip': args.eps_clip,
        'update_timestep': args.update_timestep,
        'action_select_interval': args.action_select_interval,
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


