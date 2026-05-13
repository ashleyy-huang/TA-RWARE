import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import gymnasium as gym
import numpy as np

from tarware.heuristic import heuristic_episode
from tarware.utils.logger import Logger

parser = ArgumentParser(
    description="Run heuristic baseline on TA-RWARE",
    formatter_class=ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--env_id", default="tarware-medium-8agvs-4pickers-partialobs-v1")
parser.add_argument("--num_episodes", default=100, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--render", action="store_true")
parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
parser.add_argument("--wandb_project", default="tarware")
parser.add_argument("--log_dir", default="runs")
parser.add_argument("--episode_offset", default=0, type=int,
                    help="Global episode offset for W&B logging (use when running parallel workers)")

args = parser.parse_args()


def build_episode_metrics(infos, global_episode_return, episode_returns, elapsed):
    total_deliveries = sum(i["shelf_deliveries"] for i in infos)
    total_clashes = sum(i["clashes"] for i in infos)
    total_stuck = sum(i["stucks"] for i in infos)
    episode_length = len(infos)
    pick_rate = total_deliveries * 3600 / (5 * episode_length)

    return {
        # Primary metric
        "pick_rate": pick_rate,
        # Returns
        "global_return": global_episode_return,
        "episode_returns": episode_returns.tolist(),
        # Delivery stats
        "total_deliveries": total_deliveries,
        "episode_length": episode_length,
        # Collision / stuck stats
        "total_clashes": total_clashes,
        "total_stuck": total_stuck,
        # Efficiency stats (averaged over steps)
        "avg_agvs_distance": np.mean([i["agvs_distance_travelled"] for i in infos]),
        "avg_pickers_distance": np.mean([i["pickers_distance_travelled"] for i in infos]),
        "avg_agvs_idle": np.mean([i["agvs_idle_time"] for i in infos]),
        "avg_pickers_idle": np.mean([i["pickers_idle_time"] for i in infos]),
        # Speed
        "fps": episode_length / elapsed,
    }


if __name__ == "__main__":
    config = vars(args)
    config["algo"] = "heuristic"

    logger = Logger(config, log_dir=args.log_dir, use_wandb=args.wandb)
    env = gym.make(args.env_id)

    pick_rates = []
    for i in range(args.num_episodes):
        start = time.time()
        infos, global_return, ep_returns = heuristic_episode(
            env.unwrapped, args.render, seed=args.seed + i
        )
        elapsed = time.time() - start

        metrics = build_episode_metrics(infos, global_return, ep_returns, elapsed)
        pick_rates.append(metrics["pick_rate"])
        logger.log_episode(args.episode_offset + i, metrics)

        print(
            f"Episode {i:4d} | "
            f"Pick Rate: {metrics['pick_rate']:6.2f} | "
            f"Deliveries: {metrics['total_deliveries']:3.0f} | "
            f"Clashes: {metrics['total_clashes']:4.0f} | "
            f"Stuck: {metrics['total_stuck']:3.0f} | "
            f"FPS: {metrics['fps']:5.1f}"
        )

    print(f"\n{'='*60}")
    print(f"Mean Pick Rate (all):    {np.mean(pick_rates):.2f}")
    print(f"Mean Pick Rate (last 50): {np.mean(pick_rates[-50:]):.2f}  ← paper metric")
    print(f"95% CI (last 50):        ±{1.96 * np.std(pick_rates[-50:]) / np.sqrt(50):.2f}")
    print(f"{'='*60}")

    logger.close()
    env.close()
