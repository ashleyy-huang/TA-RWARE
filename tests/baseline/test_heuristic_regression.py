"""Pin heuristic baseline numerics. Regression guard against any
accidental modification of frozen env / heuristic logic.

Measured seed=0 episode on tarware-medium-8agvs-4pickers-partialobs-v1:
  pick_rate  = 54.72  (items/hour)
  clash_rate = 0.11   (clashes/step)
  stuck_rate = 0.006  (stucks/step)

Bounds:
  pick_rate  in [49.25, 60.19]  (measured ± 10%)
  clash_rate < 1.0              (max(0.11*5, 1.0))
  stuck_rate < 1.0              (max(0.006*5, 1.0))
"""

import time

import gymnasium as gym
import numpy as np

from tarware.heuristic import heuristic_episode


def _build_metrics(infos, global_episode_return, episode_returns, elapsed):
    total_deliveries = sum(i["shelf_deliveries"] for i in infos)
    total_clashes = sum(i["clashes"] for i in infos)
    total_stuck = sum(i["stucks"] for i in infos)
    episode_length = len(infos)
    pick_rate = total_deliveries * 3600 / (5 * episode_length)
    return {
        "pick_rate": pick_rate,
        "clash_rate": total_clashes / episode_length,
        "stuck_rate": total_stuck / episode_length,
    }


def test_heuristic_medium_seed0_pick_rate_within_band():
    env = gym.make("tarware-medium-8agvs-4pickers-partialobs-v1")
    try:
        start = time.time()
        infos, global_return, ep_returns = heuristic_episode(
            env.unwrapped, render=False, seed=0
        )
        elapsed = time.time() - start
        m = _build_metrics(infos, global_return, ep_returns, elapsed)

        lower, upper = 49.25, 60.19
        assert lower <= m["pick_rate"] <= upper, (
            f"pick_rate {m['pick_rate']:.4f} outside [{lower}, {upper}]"
        )

        clash_upper = 1.0
        assert m["clash_rate"] < clash_upper, (
            f"clash_rate {m['clash_rate']:.4f} >= {clash_upper}"
        )

        stuck_upper = 1.0
        assert m["stuck_rate"] < stuck_upper, (
            f"stuck_rate {m['stuck_rate']:.4f} >= {stuck_upper}"
        )
    finally:
        env.close()
