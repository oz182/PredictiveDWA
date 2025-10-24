import torch
import torch.nn as nn


class ThetaQNet(nn.Module):
    """
    Deeper MLP that maps a small navigation feature vector to Q-values
    over discrete theta_range choices with improved capacity.
    """
    def __init__(self, input_dim: int, num_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),  # Add normalization for stability
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


