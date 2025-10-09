import torch
import torch.nn as nn


class ThetaQNet(nn.Module):
    """
    Minimal MLP that maps a small navigation feature vector to Q-values
    over discrete theta_range choices.
    """
    def __init__(self, input_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


