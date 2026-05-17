import os
import random
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import gymnasium as gym
import numpy as np
import torch
import yaml

from algos.base.registry import ALGO_REGISTRY
from algos.iac.hp import IACHyperParams
from tarware.utils.logger import Logger

parser = ArgumentParser(
    description="Train agents via a YAML config file",
    formatter_class=ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--config", required=True, help="Path to YAML config file")
parser.add_argument(
    "--num_episodes",
    default=None,
    type=int,
    help="Override num_episodes from YAML",
)
parser.add_argument(
    "--seed",
    default=None,
    type=int,
    help="Override seed from YAML",
)

if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.num_episodes is not None:
        cfg["num_episodes"] = args.num_episodes
    if args.seed is not None:
        cfg["seed"] = args.seed

    seed = cfg["seed"]
    num_episodes = cfg["num_episodes"]

    # Seed in the same order as run_iac.py — before creating env, hp, or trainer
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if cfg.get("picker_mode", "heuristic") == "train":
        raise NotImplementedError("picker_mode=train not implemented yet")

    hp = IACHyperParams(**cfg["hp"])

    env = gym.make(cfg["env_id"])

    # Build flat config dict for Logger — mirror run_iac.py's config dict structure
    # run_iac.py: config = vars(args) then adds algo and env_kwargs
    # We replicate the same keys so Logger produces equivalent W&B config
    env_kwargs = dict(gym.spec(cfg["env_id"]).kwargs)
    if hasattr(env_kwargs.get("reward_type"), "name"):
        env_kwargs["reward_type"] = env_kwargs["reward_type"].name

    log_config = {
        "algo": cfg["algo"],
        "env_id": cfg["env_id"],
        "num_episodes": num_episodes,
        "seed": seed,
        "log_dir": cfg["log_dir"],
        "wandb": cfg["wandb"]["enabled"],
        "wandb_project": cfg["wandb"]["project"],
        "episode_offset": 0,
        "lr": hp.lr,
        "adam_eps": hp.adam_eps,
        "gamma": hp.gamma,
        "gae_lambda": hp.gae_lambda,
        "n_step": hp.n_step,
        "entropy_coef": hp.entropy_coef,
        "value_coef": hp.value_coef,
        "max_grad_norm": hp.max_grad_norm,
        "picker_mode": cfg.get("picker_mode", "heuristic"),
        "device": cfg.get("device", "cpu"),
        "env_kwargs": env_kwargs,
    }

    logger = Logger(log_config, log_dir=cfg["log_dir"], use_wandb=cfg["wandb"]["enabled"])

    checkpoint_dir = str(logger.log_path / "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    trainer = ALGO_REGISTRY[cfg["algo"]](env, hp, logger, seed=seed, checkpoint_dir=checkpoint_dir)
    trainer.train(num_episodes=num_episodes)

    logger.close()
    env.close()
