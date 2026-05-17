from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class Transition:
    obs: torch.Tensor
    action: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    reward: float
    done: bool
    k: int = 1


class NStepRollout:
    """Fixed-length rollout buffer with GAE computation."""

    def __init__(self, n_step: int):
        self.n_step = n_step
        self.transitions: List[Transition] = []

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        entropy: torch.Tensor,
        reward: float,
        done: bool,
        k: int,
    ) -> None:
        self.transitions.append(
            Transition(obs, action, log_prob, value, entropy, reward, done, k)
        )

    def compute_gae(
        self, bootstrap_value: float, gamma: float, lam: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute generalised advantage estimates and value targets.

        Args:
            bootstrap_value: V(s_{n+1}); pass 0.0 if the episode ended.
            gamma: discount factor.
            lam: GAE lambda.

        Returns:
            advantages: shape (n_step,)
            returns: shape (n_step,)  — A_t + V(s_t)
        """
        n = len(self.transitions)
        advantages = torch.zeros(n)
        gae = 0.0

        next_value = bootstrap_value
        for t in reversed(range(n)):
            tr = self.transitions[t]
            mask_done = 1.0 - float(tr.done)
            discount = gamma ** tr.k
            delta = tr.reward + discount * next_value * mask_done - tr.value.item()
            gae = delta + discount * lam * gae * mask_done
            advantages[t] = gae
            next_value = tr.value.item()

        values = torch.stack([tr.value for tr in self.transitions])
        returns = advantages + values
        return advantages, returns

    def __len__(self) -> int:
        return len(self.transitions)

    def clear(self) -> None:
        self.transitions = []
