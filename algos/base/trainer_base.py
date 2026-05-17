import numpy as np

from tarware.warehouse import AgentType


class TrainerBase:
    """Base class providing SMDP scaffolding and AGV mask helpers."""

    def __init__(
        self,
        env,
        hp,
        logger,
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

    def _compute_agv_mask(self, agv) -> np.ndarray:
        """Three-state mask for a single AGV.

        Not carrying → requested shelves only;
        carrying + shelf is requested → goals only;
        carrying + shelf not requested (delivered) → empty slots only.
        NOOP (action 0) kept valid to preserve env convention.
        """
        env = self._warehouse
        n_goals = len(env.goals)
        mask = np.zeros(env.action_size, dtype=np.float32)
        mask[0] = 1.0

        if agv.carrying_shelf is None:
            mask[1 + n_goals:] = env.get_shelf_request_information()
        elif agv.carrying_shelf in env.request_queue:
            mask[1: 1 + n_goals] = 1.0
        else:
            mask[1 + n_goals:] = env.get_empty_shelf_information()

        return mask

    def _finalize_smdp(self, i, agent, smdp, env_step_counter, done):
        """Store the completed option for AGV i and reset accumulated reward."""
        obs_p, act_p, lp_p, val_p, ent_p = smdp[i]["pending"]
        k_opt = env_step_counter - smdp[i]["option_start_step"]
        agent.store(
            obs_p, act_p, lp_p, val_p, ent_p,
            smdp[i]["accumulated_reward"], done,
            k=max(k_opt, 1),
        )
        smdp[i]["accumulated_reward"] = 0.0

    def _decide_agv_action(self, i, agent, smdp, obs_tuple, env_step_counter):
        """Make a new decision for idle AGV i; return selected action."""
        wh_agent = self._warehouse.agents[self.agv_indices[i]]
        obs_i = obs_tuple[self.agv_indices[i]]
        mask_i = self._compute_agv_mask(wh_agent)
        action, log_prob, value, entropy = agent.act(obs_i, mask_i)
        smdp[i]["pending"] = (obs_i, action, log_prob, value, entropy)
        smdp[i]["option_start_step"] = env_step_counter
        smdp[i]["decisions"] += 1
        return action

    def _collect_agv_actions(self, smdp, obs_tuple, env_step_counter):
        """Return per-AGV actions; finalize/start options for idle AGVs."""
        agv_actions = []
        for i, agent in enumerate(self.iac_agents):
            wh_agent = self._warehouse.agents[self.agv_indices[i]]
            if not wh_agent.busy:
                if smdp[i]["pending"] is not None:
                    self._finalize_smdp(i, agent, smdp, env_step_counter, False)
                act = self._decide_agv_action(
                    i, agent, smdp, obs_tuple, env_step_counter
                )
                agv_actions.append(act)
            else:
                agv_actions.append(0)
        return agv_actions

    def _episode_end_update(self, smdp, env_step_counter, total_losses):
        """Finalize pending options, run gradient updates, populate total_losses."""
        for i, agent in enumerate(self.iac_agents):
            if smdp[i]["pending"] is not None:
                self._finalize_smdp(i, agent, smdp, env_step_counter, True)
                smdp[i]["pending"] = None
        for i, agent in enumerate(self.iac_agents):
            if len(agent.buffer) > 0:
                losses = agent.update(bootstrap_value=0.0)
                for k, v in losses.items():
                    total_losses[k].append(v)
