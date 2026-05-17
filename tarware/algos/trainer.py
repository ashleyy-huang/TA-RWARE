import os
from collections import defaultdict

import numpy as np
import torch

from tarware.algos.iac import HyperParams, IACAgent
from tarware.algos.picker_policy import PickerHeuristicPolicy
from tarware.utils.logger import Logger
from tarware.warehouse import AgentType


class IACTrainer:
    """Trains one independent A2C agent per AGV with a heuristic picker policy."""

    def __init__(
        self,
        env,
        hp: HyperParams,
        logger: Logger,
        seed: int = 0,
        checkpoint_dir: str = None,
    ):
        self.env = env
        # All internal warehouse state is on the unwrapped env
        self._warehouse = env.unwrapped
        self.hp = hp
        self.logger = logger
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir

        # Populate env.agents via a throwaway reset (env returns obs tuple, no info)
        env.reset(seed=seed)

        # Identify AGV indices in the flat agent list
        self.agv_indices = [
            i for i, a in enumerate(self._warehouse.agents)
            if a.type == AgentType.AGV
        ]

        # One IACAgent per AGV
        obs_dim = env.observation_space[self.agv_indices[0]].shape[0]
        action_dim = env.action_space[self.agv_indices[0]].n
        self.iac_agents = [
            IACAgent(obs_dim, action_dim, hp) for _ in self.agv_indices
        ]

        self.picker_policy = PickerHeuristicPolicy(self._warehouse)

    def train(self, num_episodes: int) -> None:
        for ep in range(num_episodes):
            ep_metrics = self._run_episode(seed=self.seed + ep)
            self.logger.log_episode(ep, ep_metrics)
            if self._is_checkpoint_ep(ep):
                self._save_checkpoint(ep)
            print(
                f"Episode {ep:5d} | "
                f"Pick Rate: {ep_metrics['pick_rate']:6.2f} | "
                f"Deliveries: {ep_metrics['total_deliveries']:3d} | "
                f"Entropy: {ep_metrics['policy_entropy_mean']:.4f} | "
                f"PL: {ep_metrics['policy_loss_mean']:.4f} | "
                f"VL: {ep_metrics['value_loss_mean']:.4f}"
            )

    def _run_episode(self, seed: int) -> dict:
        # env.reset() returns a tuple of per-agent observations (no info dict)
        obs_tuple = self.env.reset(seed=seed)
        self.picker_policy.reset()

        done = False
        episode_returns = np.zeros(self._warehouse.num_agents)
        all_infos = []
        rollout_step = 0
        total_losses = defaultdict(list)

        while not done:
            masks = self._warehouse.compute_valid_action_masks()

            agv_actions = []
            for i, agent in enumerate(self.iac_agents):
                obs = obs_tuple[self.agv_indices[i]]
                mask = masks[self.agv_indices[i]]
                action, log_prob, value, entropy = agent.act(obs, mask)
                agv_actions.append(action)
                # Store with placeholder reward/done; filled in after step
                agent.store(obs, action, log_prob, value, entropy, 0.0, False)

            picker_actions = self.picker_policy.act(agv_actions)
            full_actions = [int(a) for a in agv_actions] + picker_actions

            obs_tuple, rewards, terms, truncs, info = self.env.step(full_actions)
            done = all(terms) or all(truncs)

            # Back-fill reward and done into the just-stored transition
            for i, agent in enumerate(self.iac_agents):
                agent.buffer.transitions[-1].reward = float(rewards[self.agv_indices[i]])
                agent.buffer.transitions[-1].done = done

            episode_returns += np.array(rewards, dtype=np.float64)
            all_infos.append(info)
            rollout_step += 1

            # n-step update or episode end
            if rollout_step >= self.hp.n_step or done:
                for i, agent in enumerate(self.iac_agents):
                    if done:
                        bootstrap = 0.0
                    else:
                        with torch.no_grad():
                            obs_t = torch.tensor(
                                obs_tuple[self.agv_indices[i]], dtype=torch.float32
                            ).unsqueeze(0)
                            bootstrap = float(agent.critic(obs_t).item())
                    if len(agent.buffer) > 0:
                        losses = agent.update(bootstrap)
                        for k, v in losses.items():
                            total_losses[k].append(v)
                rollout_step = 0

        return self._build_episode_metrics(all_infos, episode_returns, total_losses, masks)

    def _build_episode_metrics(self, all_infos, episode_returns, total_losses, masks_last) -> dict:
        total_deliveries = sum(i["shelf_deliveries"] for i in all_infos)
        total_clashes = sum(i["clashes"] for i in all_infos)
        total_stuck = sum(i["stucks"] for i in all_infos)
        episode_length = len(all_infos)
        pick_rate = total_deliveries * 3600 / (5 * episode_length)

        all_travel_times = [t for i in all_infos for t in i.get("delivery_travel_times", [])]
        travel_time_metrics = {}
        if all_travel_times:
            travel_time_metrics = {
                "travel_time_mean": float(np.mean(all_travel_times)),
                "travel_time_p95": float(np.percentile(all_travel_times, 95)),
                "delivery_travel_times": all_travel_times,
            }

        wh = self._warehouse
        agv_busy = [
            sum(i["vehicles_busy"][j] for i in all_infos) / episode_length
            for j in range(wh.num_agvs)
        ]
        picker_busy = [
            sum(i["vehicles_busy"][wh.num_agvs + j] for i in all_infos) / episode_length
            for j in range(wh.num_pickers)
        ]

        mask_valid_counts = [float(masks_last[j].sum()) for j in range(wh.num_agvs)]

        return {
            "pick_rate": pick_rate,
            "global_return": float(episode_returns.sum()),
            "episode_returns": episode_returns.tolist(),
            "total_deliveries": int(total_deliveries),
            "episode_length": episode_length,
            "total_clashes": int(total_clashes),
            "total_stuck": int(total_stuck),
            "clash_rate": total_clashes / episode_length,
            "stuck_rate": total_stuck / episode_length,
            "agv_busy_ratio": float(np.mean(agv_busy)),
            "picker_busy_ratio": float(np.mean(picker_busy)),
            "mask_avg_valid_count_agv": float(np.mean(mask_valid_counts)),
            "mask_avg_valid_count_per_agv": mask_valid_counts,
            **travel_time_metrics,
            "policy_loss_mean": float(np.mean(total_losses["policy_loss"])) if total_losses["policy_loss"] else 0.0,
            "value_loss_mean": float(np.mean(total_losses["value_loss"])) if total_losses["value_loss"] else 0.0,
            "policy_entropy_mean": float(np.mean(total_losses["entropy"])) if total_losses["entropy"] else 0.0,
            "grad_norm_mean": float(np.mean(total_losses["grad_norm"])) if total_losses["grad_norm"] else 0.0,
        }

    def _is_checkpoint_ep(self, ep: int) -> bool:
        return ep >= 1000 and (ep % 500 == 0)

    def _save_checkpoint(self, ep: int) -> None:
        if self.checkpoint_dir is None:
            return
        path = os.path.join(self.checkpoint_dir, f"ep_{ep:05d}.pt")
        state = {
            "episode": ep,
            "agents": [
                {"actor": a.actor.state_dict(), "critic": a.critic.state_dict()}
                for a in self.iac_agents
            ],
            "hp": vars(self.hp),
        }
        torch.save(state, path)
