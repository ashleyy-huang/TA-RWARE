import json
import time
from pathlib import Path


class Logger:
    def __init__(self, config: dict, log_dir: str = "runs", use_wandb: bool = False):
        self.use_wandb = use_wandb
        self.run_name = self._make_run_name(config)

        self.log_path = Path(log_dir) / self.run_name
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.results_file = open(self.log_path / "episodes.jsonl", "w")
        with open(self.log_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        if use_wandb:
            import wandb
            wandb.init(
                project=config.get("wandb_project", "tarware"),
                name=self.run_name,
                config=config,
                group=f"{config.get('algo')}__{config.get('env_id')}",
                tags=[config.get("algo", "unknown"), config.get("env_id", "unknown")],
            )


        print(f"Logging to: {self.log_path}")

    def log_episode(self, episode: int, metrics: dict) -> None:
        row = {"episode": episode, **metrics}
        self.results_file.write(json.dumps(row) + "\n")
        self.results_file.flush()

        if self.use_wandb:
            import wandb
            wandb.log(row, step=episode)

    def close(self) -> None:
        self.results_file.close()
        if self.use_wandb:
            import wandb
            wandb.finish()

    @staticmethod
    def _make_run_name(config: dict) -> str:
        env_id = config.get("env_id", "tarware")
        algo = config.get("algo", "heuristic")
        seed = config.get("seed", 0)
        ts = time.strftime("%m%d_%H%M")
        return f"{algo}_{env_id}_{seed}_{ts}"
