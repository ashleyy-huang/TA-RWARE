from dataclasses import dataclass


@dataclass
class IACHyperParams:
    lr: float = 3e-4
    adam_eps: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.96
    n_step: int = 5
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
