import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import gymnasium as gym

from tarware.algos.iac import HyperParams
from tarware.algos.trainer import IACTrainer
from tarware.utils.logger import Logger

parser = ArgumentParser(
    description="Train IAC agents on TA-RWARE",
    formatter_class=ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--env_id", default="tarware-tiny-2agvs-1pickers-partialobs-v1")
parser.add_argument("--num_episodes", default=1500, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--log_dir", default="runs")
parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
parser.add_argument("--wandb_project", default="tarware")
parser.add_argument(
    "--episode_offset",
    default=0,
    type=int,
    help="Global episode offset for W&B logging",
)
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--gae_lambda", default=0.96, type=float)
parser.add_argument("--n_step", default=5, type=int)
parser.add_argument("--entropy_coef", default=0.01, type=float)
parser.add_argument("--value_coef", default=0.5, type=float)
parser.add_argument("--max_grad_norm", default=0.5, type=float)
parser.add_argument(
    "--picker_mode",
    default="heuristic",
    choices=["heuristic", "train"],
)
parser.add_argument("--device", default="cpu")

if __name__ == "__main__":
    args = parser.parse_args()

    if args.picker_mode == "train":
        raise NotImplementedError("picker_mode=train not implemented yet")

    config = vars(args)
    config["algo"] = "iac"

    env_kwargs = dict(gym.spec(args.env_id).kwargs)
    if hasattr(env_kwargs.get("reward_type"), "name"):
        env_kwargs["reward_type"] = env_kwargs["reward_type"].name
    config["env_kwargs"] = env_kwargs

    hp = HyperParams(
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        n_step=args.n_step,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
    )

    env = gym.make(args.env_id)
    logger = Logger(config, log_dir=args.log_dir, use_wandb=args.wandb)

    checkpoint_dir = str(logger.log_path / "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    trainer = IACTrainer(env, hp, logger, seed=args.seed, checkpoint_dir=checkpoint_dir)
    trainer.train(num_episodes=args.num_episodes)

    logger.close()
    env.close()
