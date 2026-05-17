from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from tarware.algos.ac_nets import Actor, Critic
from tarware.algos.rollout import NStepRollout


@dataclass
class HyperParams:
    lr: float = 3e-4
    adam_eps: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.96
    n_step: int = 5
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


class IACAgent:
    """Independent Actor-Critic agent for a single AGV."""

    def __init__(self, obs_dim: int, action_dim: int, hp: HyperParams):
        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(obs_dim)
        self.optim = Adam(
            [*self.actor.parameters(), *self.critic.parameters()],
            lr=hp.lr,
            eps=hp.adam_eps,
        )
        self.buffer = NStepRollout(hp.n_step)
        self.hp = hp

    def act(
        self, obs: np.ndarray, mask: np.ndarray
    ) -> tuple:
        """Select an action given observation and valid-action mask.

        Args:
            obs: 1-D float array of shape (obs_dim,).
            mask: 1-D float array of shape (action_dim,); 1=valid, 0=invalid.

        Returns:
            (action_int, log_prob, value, entropy)
        """
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)

        logits = self.actor(obs_t)
        # Mask invalid actions before sampling
        logits = logits.masked_fill(~mask_t, -1e9)
        dist = torch.distributions.Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        value = self.critic(obs_t)

        return (
            int(action.item()), log_prob.squeeze(0), value.squeeze(0), entropy.squeeze(0)
        )

    def store(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        entropy: torch.Tensor,
        reward: float,
        done: bool,
        k: int,
    ) -> None:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        action_t = torch.tensor(action)
        self.buffer.add(obs_t, action_t, log_prob, value, entropy, reward, done, k)

    def update(self, bootstrap_value: float) -> dict:
        """Compute GAE, run one gradient step, clear buffer.

        Returns a dict with scalar loss metrics (detached).
        """
        hp = self.hp
        advantages, returns = self.buffer.compute_gae(
            bootstrap_value, hp.gamma, hp.gae_lambda
        )

        # Normalise advantages (skip normalisation when only 1 transition)
        if len(self.buffer) > 1:
            adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            adv_norm = advantages

        log_probs = torch.stack([t.log_prob for t in self.buffer.transitions])
        entropies = torch.stack([t.entropy for t in self.buffer.transitions])
        values = torch.stack([t.value for t in self.buffer.transitions])

        policy_loss = (
            -(log_probs * adv_norm.detach()).mean()
            - hp.entropy_coef * entropies.mean()
        )
        value_loss = F.mse_loss(values, returns.detach())
        total_loss = policy_loss + hp.value_coef * value_loss

        self.optim.zero_grad()
        total_loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                [*self.actor.parameters(), *self.critic.parameters()],
                hp.max_grad_norm,
            ).item()
        )
        self.optim.step()
        self.buffer.clear()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropies.mean().item()),
            "grad_norm": grad_norm,
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "hp": self.hp,
            },
            path,
        )

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
