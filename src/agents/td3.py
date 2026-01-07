"""
TD3 (Twin Delayed DDPG) Implementation

TD3 improves on DDPG with three key techniques:
1. Twin Critics: Two Q-networks, use minimum to reduce overestimation
2. Delayed Policy Updates: Update policy less frequently than critics
3. Target Policy Smoothing: Add noise to target actions for robustness

This is well-suited for continuous control with indirect action effects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"TD3 using device: {device}")


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.FloatTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device),
        )
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Deterministic policy network."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, 
                 max_action: float = 1.0):
        super(Actor, self).__init__()
        
        self.max_action = max_action
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Output in [-1, 1]
        )
    
    def forward(self, state):
        return self.max_action * self.net(state)


class Critic(nn.Module):
    """Twin Q-networks (Q1 and Q2)."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)
    
    def q1_forward(self, state, action):
        """Only compute Q1 (used for policy update)."""
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa)


class TD3:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3).
    
    Key features:
    - Off-policy with replay buffer (sample efficient)
    - Twin critics to reduce overestimation
    - Delayed policy updates for stability
    - Target policy smoothing for robustness
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        hidden_dim: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        buffer_size: int = 100000,
        batch_size: int = 256,
        exploration_noise: float = 0.1,
        warmup_steps: int = 1000,
    ):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            max_action: Maximum absolute action value
            hidden_dim: Hidden layer size
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            policy_noise: Noise added to target policy (smoothing)
            noise_clip: Range to clip target policy noise
            policy_delay: Update policy every N critic updates
            buffer_size: Replay buffer capacity
            batch_size: Minibatch size for updates
            exploration_noise: Std of Gaussian exploration noise
            warmup_steps: Random actions before learning starts
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        self.warmup_steps = warmup_steps
        
        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        # Counters
        self.total_steps = 0
        self.update_count = 0
    
    def select_action(self, state, add_noise: bool = True) -> np.ndarray:
        """
        Select action given state.
        
        Args:
            state: Current state (numpy array)
            add_noise: Whether to add exploration noise
            
        Returns:
            Action as numpy array
        """
        # Random actions during warmup
        if self.total_steps < self.warmup_steps:
            return np.random.uniform(-self.max_action, self.max_action, 
                                    size=(self.action_dim,))
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()
        
        if add_noise:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1
    
    def update(self) -> dict:
        """
        Perform one update step.
        
        Returns:
            Dictionary with loss statistics, or None if not enough samples.
        """
        if len(self.buffer) < self.batch_size:
            return None
        
        if self.total_steps < self.warmup_steps:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # ==================== Update Critics ====================
        with torch.no_grad():
            # Select action from target policy with smoothing noise
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )
            
            # Compute target Q values (use minimum of twin critics)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Current Q estimates
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        self.update_count += 1
        
        # ==================== Delayed Policy Update ====================
        actor_loss_val = 0.0
        if self.update_count % self.policy_delay == 0:
            # Compute actor loss (maximize Q1)
            actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            
            actor_loss_val = actor_loss.item()
            
            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss_val,
            'q1_mean': current_q1.mean().item(),
            'q2_mean': current_q2.mean().item(),
            'buffer_size': len(self.buffer),
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network: θ_target = τ*θ_source + (1-τ)*θ_target"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'update_count': self.update_count,
        }, path)
        print(f"TD3 model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.update_count = checkpoint.get('update_count', 0)
        print(f"TD3 model loaded from {path}")
    
    def set_exploration_noise(self, noise: float):
        """Set exploration noise level."""
        self.exploration_noise = noise
