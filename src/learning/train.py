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
from agents.td3 import TD3

# Optional third-party integrations

try:
    import wandb
except ImportError:
    wandb = None

try:
    import optuna
except ImportError:
    optuna = None

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
    
    # Stage 1: Easy (first 65% of training)
    if progress < 0.85:
        return {
            'door_halo_radius_range': (1.6, 2.0),  # Narrow range around known value
            'door_position_x_range': (7.0, 9.0),   # Fixed around 40% of 20m corridor
            'randomize_door_side': False,          # Keep door on right side
            'stage': 'easy'
        }
    
    # Stage 2: Medium (40% - 60% of training)
    elif progress < 1.1:
        return {
            'door_halo_radius_range': (1.5, 2.3),  # Expanded range
            'door_position_x_range': (6.0, 10.0),  # More variation in position
            'randomize_door_side': False,          # Still on right side
            'stage': 'medium'
        }
    
    # Stage 3: Hard (60% - 80% of training)
    elif progress < 1.1:
        return {
            'door_halo_radius_range': (1.3, 2.5),  # Full range
            'door_position_x_range': (6.0, 10.0),  # Wide variation
            'randomize_door_side': False,           # Randomize left/right
            'stage': 'hard'
        }
    
    # Stage 4: Expert (final 20% of training)
    else:
        return {
            'door_halo_radius_range': (1.2, 2.5),  # Full range
            'door_position_x_range': (6.0, 10.0),  # Full range (25%-65% of corridor)
            'randomize_door_side': False,           # Fully randomized
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


def compute_reward(sim, progress_prev_dist: float, action_value: float = 0.0, 
                   prev_collision_count: int = 0) -> tuple[float, float, int, dict]:
    """
    Reward function with proper credit assignment for obstacle avoidance.
    
    DESIGN PRINCIPLES:
    1. Zero-centered for neutral states: Free space should NOT be penalized
    2. Sparse terminal rewards dominate: Goal/failure rewards should be large
    3. Dense shaping rewards are small: Guide behavior without drowning signal
    4. Properly scaled: Progress should be meaningful relative to penalties
    
    Components:
    1. Progress reward: Encourage moving toward goal (dense, positive)
    2. Obstacle overlap: Penalty ONLY for violations (dense, negative)
    3. Proximity shaping: Small continuous signal near obstacles (dense, negative)
    4. Collision penalty: Large penalty for physical collisions (sparse, negative)
    5. Goal bonus: Large reward for task completion (sparse, positive)
    
    Returns (reward, new_distance, new_collision_count, info)
    """
    robot_pos = sim.robot.position
    goal_pos = sim.robot.goal
    dist = float(np.linalg.norm(goal_pos - robot_pos))

    reward = 0.0
    
    # ==========================================================================
    # 1. PROGRESS REWARD - Encourage moving toward goal
    # ==========================================================================
    # At 60 Hz with ~0.5 m/s speed, typical progress per step is ~0.008m
    # Over 1000 steps traveling 8m toward goal: 8 * 20 = 160 reward
    progress = progress_prev_dist - dist  # Positive if moving toward goal
    progress_reward = progress * 1.0  # Increased from 10.0 for better signal
    reward += progress_reward
    
    # ==========================================================================
    # 2. OBSTACLE OVERLAP PENALTY - Only penalize violations, NOT free space
    # ==========================================================================
    # KEY INSIGHT: Free space should be NEUTRAL (0.0), not negative.
    # Penalizing existence creates a negative baseline that drowns out progress.
    overlap_info = check_robot_overlap(sim)
    overlap_type = overlap_info['overlap_type']
    
    if overlap_type == 'none':
        # NEUTRAL - no penalty for being in free space!
        # This is critical: the robot shouldn't be punished for existing
        overlap_reward = 0.0
    elif overlap_type == 'person':
        # Penalty for violating person's proxemic zone (social cost)
        overlap_reward = -1.5
    elif overlap_type == 'door':
        # Penalty for blocking door area (navigation cost)
        overlap_reward = -5.0
    elif overlap_type == 'both':
        # Combined penalty for both violations
        overlap_reward = -15.0
    else:
        overlap_reward = 0.0
    
    reward += overlap_reward
    
    # ==========================================================================
    # 2b. SMALL TIME PENALTY - Encourage efficiency without drowning signal
    # ==========================================================================
    # Small per-step cost to encourage faster completion.
    # At 1800 steps: 1800 * 0.02 = 36 total, much smaller than progress (~160)
    # At 1000 steps: 1000 * 0.02 = 20 total
    # Difference: 16 points saved by being 800 steps faster
    time_penalty = -0.01
    reward += time_penalty
    
    # ==========================================================================
    # 3. PROXIMITY SHAPING - Small continuous signal near obstacles
    # ==========================================================================
    door_proximity_penalty = 0.0
    if hasattr(sim.robot, 'door_position') and hasattr(sim.robot, 'corridor_bounds'):
        door_pos = np.array(sim.robot.door_position, dtype=float)
        door_radius = float(getattr(sim.robot.global_planner, 'door_halo_radius', 1.0))
        dist_to_door = float(np.linalg.norm(robot_pos - door_pos))

        bounds = sim.robot.corridor_bounds
        corridor_mid_y = (bounds['y_min'] + bounds['y_max']) * 0.5
        door_side = "left" if door_pos[1] < corridor_mid_y else "right"
        n_world = np.array([0.0, 1.0]) if door_side == "left" else np.array([0.0, -1.0])
        v = robot_pos - door_pos
        on_inward_side = np.dot(n_world, v) > 0
        
        if on_inward_side:
            influence_radius = door_radius * 2.5
            if dist_to_door < influence_radius:
                normalized_dist = dist_to_door / influence_radius
                proximity_factor = (1.0 - normalized_dist) ** 2
                door_proximity_penalty = -4.0 * proximity_factor
                reward += door_proximity_penalty
    
    # ==========================================================================
    # 4. COLLISION PENALTY - Large penalty for actual physical collisions
    # ==========================================================================
    current_collision_count = sim.collision_count
    new_collisions = current_collision_count - prev_collision_count
    
    if new_collisions > 0:
        reward += -10.0 * new_collisions  # Increased from -8.0
    
    # ==========================================================================
    # 5. GOAL REACHED BONUS - Large sparse reward for success
    # ==========================================================================
    # Should be large enough to incentivize completion over safe wandering
    # At 1000 steps with good progress (~160 reward), goal bonus should be significant
    if dist < 1.0:  # Goal reached
        reward += 50.0  # Increased from 15.0 - 10x larger to dominate
    
    # Build info dict
    info = {
        'distance': dist,
        'collisions': current_collision_count,
        'new_collisions': new_collisions,
        'overlap_type': overlap_type,
        'person_overlap': overlap_info['person_overlap'],
        'door_overlap': overlap_info['door_overlap'],
        'action_value': action_value,
        'progress_reward': progress_reward,
        'overlap_reward': overlap_reward,
        'time_penalty': time_penalty,
    }

    return reward, dist, current_collision_count, info


def train(config: Dict[str, Any], use_wandb: bool = True, run_name: Optional[str] = None) -> float:
    seed = int(config.get('seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==========================================================================
    # CONTROL MODE SELECTION
    # ==========================================================================
    # 'offset' - Agent outputs heading offset for TS-DWA (theta_ph adjustment)
    # 'w_max'  - Agent outputs w_max ratio for DWA (angular velocity sampling limit)
    control_mode = str(config.get('control_mode', 'offset'))  # Default to offset for TS-DWA
    print(f"Control Mode: {control_mode.upper()}")
    # ==========================================================================

    # Action space: single value for heading adjustment or w_max control
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

    # Build agent based on agent_type
    hidden_size = int(config.get('hidden_size', 128))
    agent_type = str(config.get('agent_type', 'ppo')).lower()
    
    if agent_type == 'ppo':
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
    elif agent_type == 'td3':
        # TD3-specific hyperparameters
        tau = float(config.get('tau', 0.005))
        policy_noise = float(config.get('policy_noise', 0.2))
        noise_clip = float(config.get('noise_clip', 0.5))
        policy_delay = int(config.get('policy_delay', 2))
        buffer_size = int(config.get('buffer_size', 1_000_000))
        batch_size = int(config.get('batch_size', 256))
        
        agent = TD3(
            state_dim=input_dim,
            action_dim=num_actions,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            tau=tau,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            policy_delay=policy_delay,
            hidden_size=hidden_size,
            buffer_size=buffer_size,
            batch_size=batch_size,
            action_std_init=0.1,  # Lower initial exploration noise for TD3
        )
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}. Supported: 'ppo', 'td3'")

    if use_wandb and wandb is not None:
        wandb.init(project=str(config.get('wandb_project', 'PredictiveDWA')),
                   name=run_name, config=config, allow_val_change=True)
        # Watch networks based on agent type
        try:
            if agent_type == 'ppo':
                wandb.watch(agent.policy.actor, log='all', log_freq=200)
                wandb.watch(agent.policy.critic, log='all', log_freq=200)
            elif agent_type == 'td3':
                wandb.watch(agent.actor, log='all', log_freq=200)
                wandb.watch(agent.critic, log='all', log_freq=200)
        except Exception:
            pass

    # PPO uses its own rollout buffer; updates are triggered by timesteps
    time_step = 0

    # Training loop (episodes of the headless simulation)
    episodes = int(config.get('episodes', 50))
    max_steps = int(config.get('max_steps', 2000))
    timeout_steps = int(config.get('timeout_steps', 1600))  # Episode timeout (failure if exceeded)
    timeout_penalty = float(config.get('timeout_penalty', -100.0))  # Penalty for timeout failure

    global_step = 0
    returns = []
    start_time = time.time()

    use_curriculum = bool(config.get('use_curriculum', False))
    
    for ep in range(episodes):
        # Get curriculum parameters or use fixed defaults
        if use_curriculum:
            curriculum = get_curriculum_params(ep, episodes)
            door_halo_radius = random.uniform(*curriculum['door_halo_radius_range'])
            door_position_x = random.uniform(*curriculum['door_position_x_range'])
            door_side_param = None if curriculum['randomize_door_side'] else 'right'
        else:
            curriculum = {'stage': 'fixed'}
            door_halo_radius = 1.8
            door_position_x = 8.0
            door_side_param = 'right'
        
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
        action_history = []  # Track action values for analysis (offset or w_max depending on mode)
        reward_components = {'progress': 0.0, 'overlap': 0.0, 'shaping': 0.0, 'time': 0.0, 'timeout': 0.0}  # Track reward breakdown
        episode_timeout = False  # Track if episode ended due to timeout
        
        # Loss tracking for this episode
        episode_losses = {'total': [], 'policy': [], 'value': [], 'entropy': []}

        # Storage for action repetition
        prev_action = None  # Agent's action (offset or w_max depending on mode)

        for t in range(max_steps):
            # Build state features
            state_feat = extract_nav_features(sim)

            # Decide whether to sample a new action or reuse previous one
            if (t % action_select_interval == 0) or (prev_action is None):
                # Sample new action via policy (also records state/action/logprob/value in buffer)
                action_normalized = agent.select_action(state_feat)  # Returns array in [-1, 1]
                prev_action = float(action_normalized[0])
                prev_action = max(-1.0, min(1.0, prev_action))
            else:
                # Reuse previous action
                if agent_type == 'ppo':
                    # PPO: log this step into PPO buffer with logprob/value
                    action_value = prev_action
                    # IMPORTANT: use policy_old for consistency with PPO's importance sampling
                    policy_device = next(agent.policy_old.parameters()).device
                    state_tensor_no_batch = torch.FloatTensor(state_feat).to(policy_device)
                    state_tensor_batch = state_tensor_no_batch.unsqueeze(0)
                    action_tensor_batch = torch.tensor([[action_value]], dtype=torch.float32, device=policy_device)
                    with torch.no_grad():
                        old_logprob, state_value, _ = agent.policy_old.evaluate(state_tensor_batch, action_tensor_batch)
                    # Match shapes used by select_action: state (no batch), action ([1, action_dim])
                    agent.buffer.states.append(state_tensor_no_batch)
                    agent.buffer.actions.append(action_tensor_batch)
                    agent.buffer.logprobs.append(old_logprob.detach())
                    agent.buffer.state_values.append(state_value.squeeze(0).detach())
                elif agent_type == 'td3':
                    # TD3: just call select_action with the same state to record transition
                    action_normalized = agent.select_action(state_feat)
                    prev_action = float(action_normalized[0])
                    prev_action = max(-1.0, min(1.0, prev_action))

            # Apply action based on control mode
            if control_mode == 'offset':
                # OFFSET MODE: Agent outputs heading offset for TS-DWA
                # Scale from [-1, 1] to [-max_offset, max_offset]
                max_offset = math.pi / 2  # ±90 degrees
                offset = prev_action * max_offset
                sim.robot.nav.agent_offset = offset
                action_history.append(abs(offset))
                action_value_for_reward = offset
            else:  # control_mode == 'w_max'
                # W_MAX MODE: Agent outputs w_max ratio for DWA
                # Scale from [-1, 1] to [0, 1]
                w_max_ratio = (prev_action + 1.0) / 2.0
                sim.robot.nav.agent_w_max = w_max_ratio
                action_history.append(w_max_ratio)
                action_value_for_reward = w_max_ratio

            # Step simulation
            _, _, done_flag = sim.step(dt)

            # Reward with all components
            reward, prev_dist, prev_collision_count, info = compute_reward(
                sim, prev_dist, action_value_for_reward, prev_collision_count
            )
            
            # ==========================================================================
            # TIMEOUT TERMINATION - Failure if episode exceeds timeout_steps
            # ==========================================================================
            # Episodes lasting too long indicate failure to navigate efficiently.
            # At 60 Hz, 1800 steps = 30 seconds real-time.
            timeout_triggered = False
            if t >= timeout_steps - 1 and not done_flag:
                timeout_triggered = True
                done_flag = True  # Force episode termination
                # Add timeout penalty proportional to remaining distance
                # Closer to goal = smaller penalty (partial credit)
                remaining_dist = info['distance']
                initial_dist = 9.0  # Approximate initial distance to goal
                progress_ratio = max(0.0, 1.0 - remaining_dist / initial_dist)
                # Full penalty if no progress, reduced if close to goal
                actual_timeout_penalty = timeout_penalty * (1.0 - 0.5 * progress_ratio)
                reward += actual_timeout_penalty
                info['timeout'] = True
                info['timeout_penalty'] = actual_timeout_penalty
            else:
                info['timeout'] = False
                info['timeout_penalty'] = 0.0
            
            episode_return += reward

            # Track overlap statistics
            overlap_counts[info['overlap_type']] += 1
            
            # Track reward component breakdown
            reward_components['progress'] += info['progress_reward']
            reward_components['overlap'] += info['overlap_reward']
            reward_components['time'] += info['time_penalty']
            reward_components['timeout'] += info['timeout_penalty']
            
            # Track timeout
            if info['timeout']:
                episode_timeout = True

            # PPO bookkeeping
            agent.buffer.rewards.append(float(reward))
            agent.buffer.is_terminals.append(bool(done_flag))

            # Trigger updates based on agent type
            time_step += 1
            
            # PPO: update every update_timestep steps (on-policy batch updates)
            # TD3: update every step (off-policy with replay buffer)
            should_update = False
            if agent_type == 'ppo':
                should_update = (time_step % update_timestep == 0)
            elif agent_type == 'td3':
                should_update = True  # Update every step for TD3
            
            if should_update:
                update_stats = agent.update()
                if update_stats is not None:
                    # Track losses for episode-level averaging
                    episode_losses['total'].append(update_stats['loss'])
                    episode_losses['policy'].append(update_stats['policy_loss'])
                    episode_losses['value'].append(update_stats['value_loss'])
                    episode_losses['entropy'].append(update_stats['entropy'])
                    
                    if use_wandb and wandb is not None:
                        wandb.log({
                            'loss/total': update_stats['loss'],
                            'loss/policy': update_stats['policy_loss'],
                            'loss/value': update_stats['value_loss'],
                            'loss/entropy': update_stats['entropy'],
                            'train_step': time_step,
                        }, step=global_step)

            # Decay action std / exploration noise
            if agent_type == 'ppo':
                if time_step % (10 * update_timestep) == 0:
                    agent.decay_action_std(action_std_decay_rate=0.01, min_action_std=0.05)
            elif agent_type == 'td3':
                # TD3: decay exploration noise less frequently (every 5000 steps)
                if time_step % 5000 == 0:
                    agent.decay_action_std(action_std_decay_rate=0.005, min_action_std=0.01)

            global_step += 1
            if done_flag:
                # Handle final transition for TD3
                if agent_type == 'td3' and agent._prev_state is not None:
                    # Complete the final transition with terminal state
                    if len(agent.buffer.rewards) > 0:
                        reward = agent.buffer.rewards[-1]
                        agent.buffer.add(
                            agent._prev_state,
                            agent._prev_action,
                            reward,
                            state_feat,  # terminal state (doesn't matter much)
                            True  # done
                        )
                        agent.buffer.rewards.pop()
                        agent.buffer.is_terminals.pop()
                    agent._prev_state = None
                    agent._prev_action = None
                
                # Update at episode boundary as well
                update_stats = agent.update()
                if update_stats is not None:
                    # Track losses for episode-level averaging
                    episode_losses['total'].append(update_stats['loss'])
                    episode_losses['policy'].append(update_stats['policy_loss'])
                    episode_losses['value'].append(update_stats['value_loss'])
                    episode_losses['entropy'].append(update_stats['entropy'])
                    
                    if use_wandb and wandb is not None:
                        wandb.log({
                            'loss/total': update_stats['loss'],
                            'loss/policy': update_stats['policy_loss'],
                            'loss/value': update_stats['value_loss'],
                            'loss/entropy': update_stats['entropy'],
                            'train_step': time_step,
                        }, step=global_step)
                break
        
        # Handle TD3 pending transition if episode ended due to max_steps (not done_flag)
        if agent_type == 'td3' and agent._prev_state is not None:
            if len(agent.buffer.rewards) > 0:
                reward = agent.buffer.rewards[-1]
                agent.buffer.add(
                    agent._prev_state,
                    agent._prev_action,
                    reward,
                    state_feat,
                    False  # Not a terminal state, just truncated
                )
                agent.buffer.rewards.pop()
                agent.buffer.is_terminals.pop()
            agent._prev_state = None
            agent._prev_action = None

        # Episode metrics
        total_steps = t + 1
        overlap_pct = {k: 100 * v / total_steps for k, v in overlap_counts.items()}
        returns.append(episode_return)
        
        # Action statistics (different interpretation based on control mode)
        if control_mode == 'offset':
            # Offset mode: track absolute offset values (in radians)
            avg_action = np.mean(action_history) if action_history else 0.0
            min_action = np.min(action_history) if action_history else 0.0
            max_action = np.max(action_history) if action_history else 0.0
            # Convert to degrees for display
            avg_action_deg = avg_action * 180 / math.pi
            max_action_deg = max_action * 180 / math.pi
        else:  # w_max mode
            avg_action = np.mean(action_history) if action_history else 1.0
            min_action = np.min(action_history) if action_history else 1.0
            max_action = np.max(action_history) if action_history else 1.0

        # Calculate episode-level average losses
        avg_episode_losses = {
            'total': np.mean(episode_losses['total']) if episode_losses['total'] else 0.0,
            'policy': np.mean(episode_losses['policy']) if episode_losses['policy'] else 0.0,
            'value': np.mean(episode_losses['value']) if episode_losses['value'] else 0.0,
            'entropy': np.mean(episode_losses['entropy']) if episode_losses['entropy'] else 0.0,
        }

        # Determine episode outcome
        final_dist = info['distance']
        if final_dist < 1.0:
            outcome = "SUCCESS"
        elif episode_timeout:
            outcome = "TIMEOUT"
        else:
            outcome = "RUNNING"  # Shouldn't happen with timeout enabled
        
        print(f"Episode {ep+1}/{episodes} [{agent_type.upper()}] | Return: {episode_return:.2f} | Steps: {total_steps} | {outcome}")
        print(f"  Curriculum Stage: {curriculum['stage'].upper()} | Door Radius: {door_halo_radius:.2f}m | Door Pos: {door_position_x:.1f}m | Side: {sim.door_side}")
        print(f"  Final Distance: {final_dist:.2f}m | Timeout Penalty: {reward_components['timeout']:.2f}")
        print(f"  Rewards - Progress: {reward_components['progress']:.2f} | Overlap: {reward_components['overlap']:.2f} | Shaping: {reward_components['shaping']:.2f}")
        print(f"  Overlaps - Free: {overlap_pct['none']:.1f}% | Person: {overlap_pct['person']:.1f}% | Door: {overlap_pct['door']:.1f}% | Both: {overlap_pct['both']:.1f}%")
        if control_mode == 'offset':
            print(f"  Offset Control - Avg: {avg_action_deg:.1f}° | Max: {max_action_deg:.1f}° | (range: ±90°)")
        else:
            print(f"  w_max Control - Avg: {avg_action:.2f} | Min: {min_action:.2f} | Max: {max_action:.2f} | (range: 0-1)")
        if episode_losses['total']:
            print(f"  Losses - Total: {avg_episode_losses['total']:.4f} | Policy: {avg_episode_losses['policy']:.4f} | Value: {avg_episode_losses['value']:.4f} | Entropy: {avg_episode_losses['entropy']:.4f}")

        if use_wandb and wandb is not None:
            wandb.log({
                'episode': ep + 1,
                'return': episode_return,
                'steps': total_steps,
                # Episode outcome
                'outcome': outcome,
                'success': 1.0 if outcome == "SUCCESS" else 0.0,
                'timeout': 1.0 if episode_timeout else 0.0,
                'final_distance': final_dist,
                # Reward breakdown
                'reward_progress': reward_components['progress'],
                'reward_overlap': reward_components['overlap'],
                'reward_shaping': reward_components['shaping'],
                'reward_timeout': reward_components['timeout'],
                # Overlap statistics
                'overlap_free_pct': overlap_pct['none'],
                'overlap_person_pct': overlap_pct['person'],
                'overlap_door_pct': overlap_pct['door'],
                'overlap_both_pct': overlap_pct['both'],
                # Action statistics (mode-dependent)
                'avg_action': avg_action,
                'min_action': min_action,
                'max_action': max_action,
                'control_mode': control_mode,
                'elapsed_min': (time.time() - start_time) / 60.0,
                # Curriculum parameters
                'curriculum_stage': curriculum['stage'],
                'door_halo_radius': door_halo_radius,
                'door_position_x': door_position_x,
                'door_side': sim.door_side,
                # Episode-averaged losses (for loss graphs by episode)
                'loss_episode/total': avg_episode_losses['total'],
                'loss_episode/policy': avg_episode_losses['policy'],
                'loss_episode/value': avg_episode_losses['value'],
                'loss_episode/entropy': avg_episode_losses['entropy'],
            })

    avg_return = float(np.mean(returns)) if returns else 0.0

    # Save policy with agent type in filename
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_filename = f'{agent_type}_policy.pt'
    agent.save(os.path.join('checkpoints', checkpoint_filename))
    print(f'Saved {agent_type.upper()} policy to checkpoints/{checkpoint_filename}')

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
    parser = argparse.ArgumentParser(description='Train navigation policy with PPO or TD3')
    # Agent selection
    parser.add_argument('--agent-type', type=str, default='ppo', choices=['ppo', 'td3'],
                        help='RL algorithm to use: ppo or td3 (default: ppo)')
    # Control mode selection
    parser.add_argument('--control-mode', type=str, default='offset', choices=['offset', 'w_max'],
                        help='Control mode: offset (heading adjustment for TS-DWA) or w_max (angular velocity limit for DWA)')
    # Common parameters
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=3000)
    parser.add_argument('--timeout-steps', type=int, default=1800, 
                        help='Episode timeout in steps (1800 = 30s at 60Hz). Episodes exceeding this are failures.')
    parser.add_argument('--timeout-penalty', type=float, default=-100.0,
                        help='Penalty for timeout failure (scaled by remaining distance)')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr-actor', type=float, default=None)
    parser.add_argument('--lr-critic', type=float, default=None)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.99)
    # PPO-specific parameters
    parser.add_argument('--k-epochs', type=int, default=10, help='PPO: K epochs for policy update')
    parser.add_argument('--eps-clip', type=float, default=0.2, help='PPO: clipping parameter')
    parser.add_argument('--update-timestep', type=int, default=2000, help='PPO: update policy every N timesteps')
    # TD3-specific parameters
    parser.add_argument('--tau', type=float, default=0.005, help='TD3: soft update coefficient')
    parser.add_argument('--policy-noise', type=float, default=0.2, help='TD3: noise added to target policy')
    parser.add_argument('--noise-clip', type=float, default=0.5, help='TD3: range to clip target policy noise')
    parser.add_argument('--policy-delay', type=int, default=2, help='TD3: frequency of delayed policy updates')
    parser.add_argument('--buffer-size', type=int, default=1_000_000, help='TD3: replay buffer size')
    parser.add_argument('--batch-size', type=int, default=256, help='TD3: batch size for updates')
    # Common training parameters
    parser.add_argument('--action-select-interval', type=int, default=1, help='Select a new action every N steps (default 1 = every step)')
    parser.add_argument('--use-curriculum', action='store_true', help='Enable curriculum learning for door parameters')
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
        # Agent and control mode selection
        'agent_type': args.agent_type,
        'control_mode': args.control_mode,
        # Common parameters
        'episodes': args.episodes,
        'max_steps': args.max_steps,
        'timeout_steps': args.timeout_steps,
        'timeout_penalty': args.timeout_penalty,
        'learning_rate': args.lr,
        'lr_actor': args.lr_actor if args.lr_actor is not None else args.lr,
        'lr_critic': args.lr_critic if args.lr_critic is not None else args.lr,
        'hidden_size': args.hidden,
        'gamma': args.gamma,
        'action_select_interval': args.action_select_interval,
        'use_curriculum': args.use_curriculum,
        'seed': args.seed,
        'wandb_project': args.wandb_project,
        # PPO-specific parameters
        'k_epochs': args.k_epochs,
        'eps_clip': args.eps_clip,
        'update_timestep': args.update_timestep,
        # TD3-specific parameters
        'tau': args.tau,
        'policy_noise': args.policy_noise,
        'noise_clip': args.noise_clip,
        'policy_delay': args.policy_delay,
        'buffer_size': args.buffer_size,
        'batch_size': args.batch_size,
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


