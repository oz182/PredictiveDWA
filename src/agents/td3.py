"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient) Agent

Reference: "Addressing Function Approximation Error in Actor-Critic Methods" (Fujimoto et al., 2018)

Key improvements over DDPG:
1. Twin Q-networks to reduce overestimation bias
2. Delayed policy updates
3. Target policy smoothing with clipped noise
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

################################## set device ##################################
print("============================================================================================")
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## Replay Buffer ##################################
class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""
    
    def __init__(self, max_size: int = 1_000_000):
        self.storage = deque(maxlen=max_size)
        # Compatibility attributes for train.py interface
        self.rewards = []
        self.is_terminals = []
        self._pending_state = None
        self._pending_action = None
    
    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.storage.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample a batch of transitions."""
        batch = random.sample(self.storage, min(batch_size, len(self.storage)))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.FloatTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device),
        )
    
    def __len__(self):
        return len(self.storage)
    
    def clear(self):
        """Clear both the replay storage and the pending transition lists."""
        self.storage.clear()
        del self.rewards[:]
        del self.is_terminals[:]
        self._pending_state = None
        self._pending_action = None


################################## Actor Network ##################################
class Actor(nn.Module):
    """Deterministic policy network."""
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0, hidden_size: int = 256):
        super(Actor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        self.max_action = max_action
    
    def forward(self, state):
        return self.max_action * self.net(state)


################################## Critic Network ##################################
class Critic(nn.Module):
    """Twin Q-networks for TD3."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super(Critic, self).__init__()
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Q2 network (twin)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)
    
    def q1_forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


################################## TD3 Agent ##################################
class TD3:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.
    
    Compatible interface with PPO agent for use in train.py.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        max_action: float = 1.0,
        hidden_size: int = 256,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        # For API compatibility with PPO (these are exploration noise params)
        action_std_init: float = 0.1,
        **kwargs  # Absorb unused PPO params like K_epochs, eps_clip, has_continuous_action_space
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.max_action = max_action
        self.batch_size = batch_size
        
        # Exploration noise (equivalent to action_std in PPO)
        self.exploration_noise = action_std_init
        
        # Actor networks
        self.actor = Actor(state_dim, action_dim, max_action, hidden_size).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action, hidden_size).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic networks
        self.critic = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        # Update counter for delayed policy updates
        self.total_updates = 0
        
        # For compatibility: track if we have a pending state-action to store
        self._prev_state = None
        self._prev_action = None
    
    def set_action_std(self, new_action_std: float):
        """Set exploration noise (API compatibility with PPO)."""
        self.exploration_noise = new_action_std
    
    def decay_action_std(self, action_std_decay_rate: float, min_action_std: float):
        """Decay exploration noise over time (API compatibility with PPO)."""
        print("--------------------------------------------------------------------------------------------")
        self.exploration_noise = self.exploration_noise - action_std_decay_rate
        self.exploration_noise = round(self.exploration_noise, 4)
        if self.exploration_noise <= min_action_std:
            self.exploration_noise = min_action_std
            print("setting exploration noise to min: ", self.exploration_noise)
        else:
            print("setting exploration noise to: ", self.exploration_noise)
        print("--------------------------------------------------------------------------------------------")
    
    def select_action(self, state, add_noise: bool = True) -> np.ndarray:
        """
        Select action using current policy with optional exploration noise.
        
        Returns action as numpy array (flattened) for compatibility with PPO.
        """
        state_tensor = torch.FloatTensor(state).to(device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()
        
        if add_noise:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)
        
        # Store state-action for later when we receive reward
        # Complete the previous transition if we have one
        if self._prev_state is not None and len(self.buffer.rewards) > 0:
            reward = self.buffer.rewards[-1]
            done = self.buffer.is_terminals[-1]
            self.buffer.add(
                self._prev_state,
                self._prev_action,
                reward,
                state,  # current state is the next_state for previous transition
                done
            )
            # Clear the pending lists after adding to replay buffer
            self.buffer.rewards.pop()
            self.buffer.is_terminals.pop()
        
        # Store current state-action as pending
        self._prev_state = np.array(state, dtype=np.float32)
        self._prev_action = np.array(action, dtype=np.float32)
        
        return action
    
    def update(self):
        """
        Perform one update step using TD3 algorithm.
        
        Returns dict with loss statistics or None if buffer is too small.
        """
        # Need enough samples in replay buffer
        if len(self.buffer) < self.batch_size:
            return None
        
        self.total_updates += 1
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        with torch.no_grad():
            # Target policy smoothing: add clipped noise to next actions
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )
            
            # Twin Q-targets
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
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()
        
        # Delayed policy update
        actor_loss_val = 0.0
        if self.total_updates % self.policy_delay == 0:
            # Actor loss: maximize Q1
            actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            self.actor_optimizer.step()
            
            actor_loss_val = actor_loss.item()
            
            # Soft update target networks
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target, self.critic)
        
        # Return loss statistics in compatible format
        return {
            "loss": critic_loss.item() + actor_loss_val,
            "policy_loss": actor_loss_val,
            "value_loss": critic_loss.item(),
            "entropy": 0.0,  # TD3 doesn't have entropy bonus
        }
    
    def _soft_update(self, target_net: nn.Module, source_net: nn.Module):
        """Polyak averaging for target network update."""
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, checkpoint_path: str):
        """Save actor network weights."""
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, checkpoint_path)
    
    def load(self, checkpoint_path: str):
        """Load actor network weights."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle both old format (just actor state dict) and new format (full checkpoint)
        if 'actor' in checkpoint and isinstance(checkpoint['actor'], dict):
            # New format: {'actor': state_dict, 'actor_target': state_dict, ...}
            self.actor.load_state_dict(checkpoint['actor'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
        else:
            # Legacy format: direct state dict with 'actor.*' or 'net.*' keys
            # Check if we need to remap 'actor.*' keys to 'net.*' keys
            first_key = next(iter(checkpoint.keys()), '')
            if first_key.startswith('actor.'):
                # Remap 'actor.*' -> 'net.*' for actor, ignore 'critic.*' keys
                actor_state = {}
                for k, v in checkpoint.items():
                    if k.startswith('actor.'):
                        new_key = 'net.' + k[len('actor.'):]
                        actor_state[new_key] = v
                self.actor.load_state_dict(actor_state)
                self.actor_target.load_state_dict(actor_state)
            else:
                # Assume it's already in 'net.*' format
                self.actor.load_state_dict(checkpoint)
                self.actor_target.load_state_dict(checkpoint)
    
    # Compatibility properties for train.py that accesses agent.policy_old
    @property
    def policy_old(self):
        """Return actor for API compatibility with PPO."""
        return self.actor
    
    @property
    def policy(self):
        """Return actor for API compatibility with PPO."""
        return self.actor

