import json
import sys

sys.path.insert(0, "/mnt/sda/home/r147250250916/research/MARL/TA-RWARE")

import gymnasium as gym
import pytest

from tarware.algos.iac import HyperParams
from tarware.algos.trainer import IACTrainer
from tarware.utils.logger import Logger


def test_iac_10_episodes_no_crash(tmp_path):
    env = gym.make("tarware-tiny-2agvs-1pickers-partialobs-v1")
    hp = HyperParams(n_step=5)
    config = {
        "algo": "iac",
        "env_id": "tarware-tiny-2agvs-1pickers-partialobs-v1",
        "seed": 0,
    }
    logger = Logger(config, log_dir=str(tmp_path), use_wandb=False)
    ckpt_dir = str(tmp_path / "ckpt")

    trainer = IACTrainer(env, hp, logger, seed=0, checkpoint_dir=ckpt_dir)
    trainer.train(num_episodes=10)
    logger.close()
    env.close()

    # Find the run dir created inside tmp_path
    run_dirs = [p for p in tmp_path.iterdir() if p.is_dir() and p.name != "ckpt"]
    assert len(run_dirs) == 1, f"Expected 1 run dir, found: {run_dirs}"
    episodes_file = run_dirs[0] / "episodes.jsonl"

    rows = [json.loads(line) for line in episodes_file.read_text().splitlines()]
    assert len(rows) == 10, f"Expected 10 episode rows, got {len(rows)}"
    assert all("pick_rate" in r for r in rows)
    assert all("policy_entropy_mean" in r for r in rows)
    assert all("policy_loss_mean" in r for r in rows)
    assert all("value_loss_mean" in r for r in rows)
    assert all("grad_norm_mean" in r for r in rows)
    assert all("avg_decisions_per_agv" in r for r in rows)
