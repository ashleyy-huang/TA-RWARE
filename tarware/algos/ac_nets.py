import numpy as np
import torch
import torch.nn as nn


def _ortho_init(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class Actor(nn.Module):
    """Two-hidden-layer MLP policy network; returns raw logits."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        sqrt2 = np.sqrt(2)
        self.net = nn.Sequential(
            _ortho_init(nn.Linear(obs_dim, 64), gain=sqrt2),
            nn.ReLU(),
            _ortho_init(nn.Linear(64, 64), gain=sqrt2),
            nn.ReLU(),
        )
        self.head = _ortho_init(nn.Linear(64, action_dim), gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(obs))


class Critic(nn.Module):
    """Two-hidden-layer MLP value network; returns scalar per sample."""

    def __init__(self, obs_dim: int):
        super().__init__()
        sqrt2 = np.sqrt(2)
        self.net = nn.Sequential(
            _ortho_init(nn.Linear(obs_dim, 64), gain=sqrt2),
            nn.ReLU(),
            _ortho_init(nn.Linear(64, 64), gain=sqrt2),
            nn.ReLU(),
        )
        self.head = _ortho_init(nn.Linear(64, 1), gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(obs)).squeeze(-1)
