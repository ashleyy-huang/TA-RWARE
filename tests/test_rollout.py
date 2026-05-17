import numpy as np
import torch
import pytest
import sys

sys.path.insert(0, "/mnt/sda/home/r147250250916/research/MARL/TA-RWARE")

from tarware.algos.rollout import NStepRollout, Transition


def _make_dummy_transition(reward, value_scalar, done):
    obs = torch.zeros(4)
    action = torch.tensor(0)
    value = torch.tensor(value_scalar)
    log_prob = torch.tensor(0.0)
    entropy = torch.tensor(0.0)
    return obs, action, log_prob, value, entropy, reward, done


def test_gae_correctness():
    """Numerically verify GAE against hand-computed values."""
    gamma = 0.99
    lam = 0.96
    bootstrap = 0.4

    # Sequence: (reward, value)
    seq = [(1.0, 0.5), (0.5, 0.3), (2.0, 0.8)]
    rollout = NStepRollout(n_step=3)
    for r, v in seq:
        obs, act, lp, val, ent, rw, dn = _make_dummy_transition(r, v, False)
        rollout.add(obs, act, lp, val, ent, rw, dn, k=1)

    adv, ret = rollout.compute_gae(bootstrap, gamma, lam)

    # Hand-compute backwards
    g = gamma
    l = lam

    # t=2
    delta_2 = 2.0 + g * bootstrap - 0.8
    A_2 = delta_2  # gae starts at 0

    # t=1
    delta_1 = 0.5 + g * 0.8 - 0.3
    A_1 = delta_1 + g * l * A_2

    # t=0
    delta_0 = 1.0 + g * 0.3 - 0.5
    A_0 = delta_0 + g * l * A_1

    expected_adv = torch.tensor([A_0, A_1, A_2], dtype=torch.float32)
    expected_ret = expected_adv + torch.tensor([0.5, 0.3, 0.8])

    assert torch.allclose(adv, expected_adv, atol=1e-5), \
        f"advantages mismatch: got {adv}, expected {expected_adv}"
    assert torch.allclose(ret, expected_ret, atol=1e-5), \
        f"returns mismatch: got {ret}, expected {expected_ret}"


def test_gae_episode_boundary():
    """done=True at last step → bootstrap is zero'd by mask."""
    gamma = 0.99
    lam = 0.96
    bootstrap = 999.0  # should be ignored because last step is done

    rollout = NStepRollout(n_step=2)
    # t=0: reward=1.0, value=0.5, done=False
    rollout.add(*_make_dummy_transition(1.0, 0.5, False), k=1)
    # t=1: reward=2.0, value=0.8, done=True
    rollout.add(*_make_dummy_transition(2.0, 0.8, True), k=1)

    adv, ret = rollout.compute_gae(bootstrap, gamma, lam)

    # t=1: done=True → mask=0 → next_value contribution = 0
    delta_1 = 2.0 + gamma * bootstrap * 0.0 - 0.8  # bootstrap masked out
    A_1 = delta_1

    # t=0: done=False → mask=1
    delta_0 = 1.0 + gamma * 0.8 * 1.0 - 0.5
    A_0 = delta_0 + gamma * lam * A_1 * 1.0

    expected_adv = torch.tensor([A_0, A_1], dtype=torch.float32)
    expected_ret = expected_adv + torch.tensor([0.5, 0.8])

    assert torch.allclose(adv, expected_adv, atol=1e-5)
    assert torch.allclose(ret, expected_ret, atol=1e-5)


def test_clear():
    rollout = NStepRollout(n_step=5)
    rollout.add(*_make_dummy_transition(1.0, 0.5, False), k=1)
    assert len(rollout) == 1
    rollout.clear()
    assert len(rollout) == 0


def test_gae_smdp_discount():
    """SMDP discount: each transition has variable option duration k.

    Verify gamma^k is applied correctly in bootstrap and GAE recursion.
    """
    rollout = NStepRollout(n_step=10)
    gamma = 0.99
    lam = 0.96

    transitions_spec = [
        (1.0, 0.5, 1),
        (0.5, 0.3, 3),
        (2.0, 0.8, 2),
    ]
    for r, v, k in transitions_spec:
        rollout.add(
            obs=torch.zeros(1), action=torch.tensor(0),
            log_prob=torch.tensor(0.0), value=torch.tensor(v),
            entropy=torch.tensor(0.0), reward=r, done=False, k=k,
        )

    bootstrap = 0.4
    advantages, returns = rollout.compute_gae(bootstrap, gamma, lam)

    expected_gae_2 = 2.0 + (gamma ** 2) * 0.4 - 0.8
    expected_gae_1 = (0.5 + (gamma ** 3) * 0.8 - 0.3) + (gamma ** 3) * lam * expected_gae_2
    expected_gae_0 = (1.0 + gamma * 0.3 - 0.5) + gamma * lam * expected_gae_1

    assert torch.allclose(advantages[2], torch.tensor(expected_gae_2), atol=1e-5)
    assert torch.allclose(advantages[1], torch.tensor(expected_gae_1), atol=1e-5)
    assert torch.allclose(advantages[0], torch.tensor(expected_gae_0), atol=1e-5)

    expected_returns = advantages + torch.tensor([0.5, 0.3, 0.8])
    assert torch.allclose(returns, expected_returns, atol=1e-5)
