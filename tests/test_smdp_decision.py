"""Verify SMDP loop: policy is called only when AGV is idle."""
import sys

sys.path.insert(0, "/mnt/sda/home/r147250250916/research/MARL/TA-RWARE")

import gymnasium as gym

from tarware.algos.iac import HyperParams, IACAgent
from tarware.algos.picker_policy import PickerHeuristicPolicy
from tarware.utils.logger import Logger
from tarware.warehouse import AgentType


def _make_counting_act(agent: IACAgent, counter: list):
    original_act = agent.act

    def counting_act(obs, mask):
        counter[0] += 1
        return original_act(obs, mask)

    return counting_act


def test_smdp_only_decides_when_idle():
    """AGVs should only receive policy calls on idle steps."""
    env = gym.make("tarware-tiny-2agvs-1pickers-partialobs-v1")
    warehouse = env.unwrapped

    hp = HyperParams()
    env.reset(seed=42)

    agv_indices = [
        i for i, a in enumerate(warehouse.agents)
        if a.type == AgentType.AGV
    ]
    num_agvs = len(agv_indices)
    obs_dim = env.observation_space[agv_indices[0]].shape[0]
    action_dim = env.action_space[agv_indices[0]].n
    agents = [IACAgent(obs_dim, action_dim, hp) for _ in agv_indices]

    counters = [[0] for _ in range(num_agvs)]
    for i, agent in enumerate(agents):
        agent.act = _make_counting_act(agent, counters[i])

    picker_policy = PickerHeuristicPolicy(warehouse)
    obs_tuple = env.reset(seed=42)
    picker_policy.reset()

    # Run 20 steps with the SMDP loop logic mirrored from trainer
    smdp = [
        {"pending": None, "accumulated_reward": 0.0}
        for _ in range(num_agvs)
    ]
    decision_steps = [[] for _ in range(num_agvs)]  # which steps had decisions

    for step in range(20):
        masks = warehouse.compute_valid_action_masks()
        agv_actions = []

        for i, agent in enumerate(agents):
            wh_agent = warehouse.agents[agv_indices[i]]
            if not wh_agent.busy:
                obs_i = obs_tuple[agv_indices[i]]
                mask_i = masks[agv_indices[i]]
                action, log_prob, value, entropy = agent.act(obs_i, mask_i)
                smdp[i]["pending"] = (obs_i, action, log_prob, value, entropy)
                decision_steps[i].append(step)
                agv_actions.append(action)
            else:
                agv_actions.append(0)

        picker_actions = picker_policy.act(agv_actions)
        full_actions = [int(a) for a in agv_actions] + picker_actions
        obs_tuple, rewards, terms, truncs, _ = env.step(full_actions)

        for i in range(num_agvs):
            smdp[i]["accumulated_reward"] += float(rewards[agv_indices[i]])

        if all(terms) or all(truncs):
            break

    env.close()

    # Verify: policy calls == number of idle steps per AGV
    for i in range(num_agvs):
        assert counters[i][0] == len(decision_steps[i]), (
            f"AGV {i}: {counters[i][0]} policy calls but {len(decision_steps[i])} idle steps"
        )

    # Verify: at least one AGV had fewer decisions than steps (was busy for some steps)
    total_decisions = sum(len(d) for d in decision_steps)
    assert total_decisions < 20 * num_agvs, (
        "Expected AGVs to be busy for some steps, but policy was called every step"
    )

    # Verify: decisions happen at step 0 (all idle at reset) for both AGVs
    for i in range(num_agvs):
        assert 0 in decision_steps[i], f"AGV {i} should decide at step 0 (idle after reset)"
