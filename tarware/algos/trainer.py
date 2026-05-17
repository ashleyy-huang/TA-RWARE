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
        self._warehouse = env.unwrapped
        self.hp = hp
        self.logger = logger
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir

        env.reset(seed=seed)

        self.agv_indices = [
            i for i, a in enumerate(self._warehouse.agents)
            if a.type == AgentType.AGV
        ]

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
            avg_dec = ep_metrics.get("avg_decisions_per_agv", 0.0)
            print(
                f"Episode {ep:5d} | "
                f"Pick: {ep_metrics['pick_rate']:6.2f} | "
                f"Deliv: {ep_metrics['total_deliveries']:3d} | "
                f"Decisions: {avg_dec:5.1f} | "
                f"Entropy: {ep_metrics['policy_entropy_mean']:.4f} | "
                f"PL: {ep_metrics['policy_loss_mean']:+.4f} | "
                f"VL: {ep_metrics['value_loss_mean']:.4f} | "
                f"Bsy: {ep_metrics['agv_busy_ratio']:.2f}"
            )

    def _run_episode(self, seed: int) -> dict:
        obs_tuple = self.env.reset(seed=seed)
        self.picker_policy.reset()

        num_agvs = len(self.agv_indices)
        # Per-AGV SMDP state: pending decision and accumulated reward
        smdp = [
            {"pending": None, "accumulated_reward": 0.0, "decisions": 0}
            for _ in range(num_agvs)
        ]

        done = False
        episode_returns = np.zeros(self._warehouse.num_agents)
        all_infos = []
        total_losses = defaultdict(list)

        # Mask aggregation accumulators (Bug 3 fix)
        mask_valid_count_sum = np.zeros(num_agvs)
        mask_step_count = 0

        while not done:
            masks = self._warehouse.compute_valid_action_masks()

            # Accumulate mask stats every step (for comparable metric across runs)
            for i in range(num_agvs):
                mask_valid_count_sum[i] += float(masks[self.agv_indices[i]].sum())
            mask_step_count += 1

            agv_actions = []
            for i, agent in enumerate(self.iac_agents):
                wh_agent = self._warehouse.agents[self.agv_indices[i]]
                if not wh_agent.busy:
                    # Finalize previous pending decision if any
                    if smdp[i]["pending"] is not None:
                        obs_p, act_p, lp_p, val_p, ent_p = smdp[i]["pending"]
                        agent.store(
                            obs_p, act_p, lp_p, val_p, ent_p,
                            smdp[i]["accumulated_reward"], False,
                        )
                        smdp[i]["accumulated_reward"] = 0.0

                    # Make a new decision
                    obs_i = obs_tuple[self.agv_indices[i]]
                    mask_i = masks[self.agv_indices[i]]
                    action, log_prob, value, entropy = agent.act(obs_i, mask_i)
                    smdp[i]["pending"] = (obs_i, action, log_prob, value, entropy)
                    smdp[i]["decisions"] += 1
                    agv_actions.append(action)
                else:
                    # AGV busy: submit NOOP, do not call policy
                    agv_actions.append(0)

            picker_actions = self.picker_policy.act(agv_actions)
            full_actions = [int(a) for a in agv_actions] + picker_actions

            obs_tuple, rewards, terms, truncs, info = self.env.step(full_actions)
            done = all(terms) or all(truncs)

            episode_returns += np.array(rewards, dtype=np.float64)
            all_infos.append(info)

            # Accumulate per-AGV rewards into SMDP state
            for i in range(num_agvs):
                smdp[i]["accumulated_reward"] += float(rewards[self.agv_indices[i]])

        # Finalize all pending decisions at episode end with done=True
        for i, agent in enumerate(self.iac_agents):
            if smdp[i]["pending"] is not None:
                obs_p, act_p, lp_p, val_p, ent_p = smdp[i]["pending"]
                agent.store(
                    obs_p, act_p, lp_p, val_p, ent_p,
                    smdp[i]["accumulated_reward"], True,
                )
                smdp[i]["pending"] = None
                smdp[i]["accumulated_reward"] = 0.0

        # Episode-end update for each AGV
        for i, agent in enumerate(self.iac_agents):
            if len(agent.buffer) > 0:
                losses = agent.update(bootstrap_value=0.0)
                for k, v in losses.items():
                    total_losses[k].append(v)

        avg_decisions = float(np.mean([s["decisions"] for s in smdp]))
        mask_valid_counts = (mask_valid_count_sum / max(mask_step_count, 1)).tolist()

        return self._build_episode_metrics(
            all_infos, episode_returns, total_losses,
            mask_valid_counts, avg_decisions,
        )

    def _build_episode_metrics(
        self, all_infos, episode_returns, total_losses,
        mask_valid_counts, avg_decisions,
    ) -> dict:
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
            "avg_decisions_per_agv": avg_decisions,
            **travel_time_metrics,
            "policy_loss_mean": (
                float(np.mean(total_losses["policy_loss"]))
                if total_losses["policy_loss"] else 0.0
            ),
            "value_loss_mean": (
                float(np.mean(total_losses["value_loss"]))
                if total_losses["value_loss"] else 0.0
            ),
            "policy_entropy_mean": (
                float(np.mean(total_losses["entropy"]))
                if total_losses["entropy"] else 0.0
            ),
            "grad_norm_mean": (
                float(np.mean(total_losses["grad_norm"]))
                if total_losses["grad_norm"] else 0.0
            ),
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
