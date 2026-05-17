"""
Parity test: PickerHeuristicPolicy must produce identical picker actions to
heuristic_episode when AGV actions come from heuristic_episode.
"""
import sys
sys.path.insert(0, "/mnt/sda/home/r147250250916/research/MARL/TA-RWARE")

from collections import OrderedDict

import gymnasium as gym
import numpy as np
import pytest

from tarware.algos.picker_policy import PickerHeuristicPolicy
from tarware.heuristic import Mission, MissionType
from tarware.utils.utils import flatten_list, split_list
from tarware.warehouse import AgentType

SEEDS = [0, 1, 2, 3, 4]
ENV_ID = "tarware-tiny-2agvs-1pickers-partialobs-v1"


def collect_actions_from_heuristic(env, seed):
    """Run one episode of heuristic and record (agv_actions, picker_actions) per step.

    Full heuristic loop copied verbatim from heuristic.py so the reference
    behaviour is unambiguous.
    """
    non_goal_location_ids = []
    for id_, coords in env.action_id_to_coords_map.items():
        if (coords[1], coords[0]) not in env.goals:
            non_goal_location_ids.append(id_)
    non_goal_location_ids = np.array(non_goal_location_ids)
    location_map = env.action_id_to_coords_map

    env.reset(seed=seed)

    agents = env.agents
    agvs = [a for a in agents if a.type == AgentType.AGV]
    pickers = [a for a in agents if a.type == AgentType.PICKER]
    coords_original_loc_map = {v: k for k, v in env.action_id_to_coords_map.items()}
    sections = env.rack_groups
    picker_sections = split_list(sections, len(pickers))
    picker_sections = [flatten_list(l) for l in picker_sections]

    assigned_agvs = OrderedDict()
    assigned_pickers = OrderedDict()
    assigned_items = OrderedDict()

    step_actions = []
    done = False
    timestep = 0

    while not done:
        request_queue = env.request_queue
        goal_locations = env.goals
        actions = {k: 0 for k in agents}

        # [AGV None -> AGV PICKING]
        for item in request_queue:
            if item.id in assigned_items.values():
                continue
            available_agvs = [a for a in agvs if not a.busy and not a.carrying_shelf]
            available_agvs = [a for a in available_agvs if a not in assigned_agvs]
            if not available_agvs:
                continue
            agv_shortest_paths = [
                env.find_path((a.y, a.x), (item.y, item.x), a, care_for_agents=False)
                for a in available_agvs
            ]
            agv_distances = [len(p) for p in agv_shortest_paths]
            closest_agv = available_agvs[np.argmin(agv_distances)]
            item_location_id = coords_original_loc_map[(item.y, item.x)]
            if closest_agv:
                assigned_agvs[closest_agv] = Mission(
                    MissionType.PICKING, item_location_id, item.x, item.y, timestep
                )
                assigned_items[closest_agv] = item.id

        for agv in agvs:
            if agv in assigned_agvs and (
                agv.x == assigned_agvs[agv].location_x
                and agv.y == assigned_agvs[agv].location_y
            ):
                assigned_agvs[agv].at_location = True

            if agv not in assigned_agvs or agv.busy:
                continue

            # [AGV PICKING -> AGV DELIVERING]
            if (
                assigned_agvs[agv].mission_type == MissionType.PICKING
                and assigned_agvs[agv].at_location
                and agv.carrying_shelf
            ):
                goal_shortest_paths = [
                    env.find_path(
                        (agv.y, agv.x), (y, x), agv, care_for_agents=False
                    )
                    for (x, y) in goal_locations
                ]
                goal_distances = [len(p) for p in goal_shortest_paths]
                closest_goal = goal_locations[np.argmin(goal_distances)]
                goal_location_id = coords_original_loc_map[
                    (closest_goal[1], closest_goal[0])
                ]
                assigned_agvs.pop(agv)
                assigned_agvs[agv] = Mission(
                    MissionType.DELIVERING,
                    goal_location_id,
                    closest_goal[0],
                    closest_goal[1],
                    timestep,
                )

            # [AGV DELIVERING -> AGV RETURNING]
            if (
                assigned_agvs[agv].mission_type == MissionType.DELIVERING
                and assigned_agvs[agv].at_location
                and agv.carrying_shelf
            ):
                empty_shelves = env.get_empty_shelf_information()
                empty_location_ids = list(non_goal_location_ids[empty_shelves > 0])
                assigned_item_loc_agvs = [
                    mission.location_id for mission in assigned_agvs.values()
                ]
                empty_location_ids = [
                    loc_id
                    for loc_id in empty_location_ids
                    if loc_id not in assigned_item_loc_agvs
                ]
                if not empty_location_ids:
                    continue
                empty_location_yx = [location_map[i] for i in empty_location_ids]
                closest_empty_location_paths = [
                    env.find_path(
                        (agv.y, agv.x), (y, x), agv, care_for_agents=False
                    )
                    for (y, x) in empty_location_yx
                ]
                closest_empty_location_distances = [
                    len(p) for p in closest_empty_location_paths
                ]
                closest_location_id = empty_location_ids[
                    np.argmin(closest_empty_location_distances)
                ]
                closest_location_yx = location_map[closest_location_id]
                assigned_agvs.pop(agv)
                assigned_agvs[agv] = Mission(
                    MissionType.RETURNING,
                    closest_location_id,
                    closest_location_yx[1],
                    closest_location_yx[0],
                    timestep,
                )

            # [AGV RETURNING -> AGV None]
            if (
                assigned_agvs[agv].mission_type == MissionType.RETURNING
                and assigned_agvs[agv].at_location
                and not agv.carrying_shelf
            ):
                assigned_agvs.pop(agv)
                assigned_items.pop(agv)

        # Picker assignment (heuristic.py lines 117-124)
        for agv, mission in assigned_agvs.items():
            if mission.mission_type in [MissionType.PICKING, MissionType.RETURNING]:
                in_pickers_zone = [
                    (mission.location_y, mission.location_x) in p
                    for p in picker_sections
                ]
                relevant_picker = pickers[in_pickers_zone.index(True)]
                if relevant_picker not in assigned_pickers.keys():
                    assigned_pickers[relevant_picker] = Mission(
                        MissionType.PICKING,
                        mission.location_id,
                        mission.location_x,
                        mission.location_y,
                        timestep,
                    )

        # Pop pickers that arrived (heuristic.py lines 125-129)
        for picker in pickers:
            if picker in assigned_pickers and (
                picker.x == assigned_pickers[picker].location_x
                and picker.y == assigned_pickers[picker].location_y
            ):
                assigned_pickers[picker].at_location = True
                assigned_pickers.pop(picker)

        # Build actions
        for agv, mission in assigned_agvs.items():
            actions[agv] = mission.location_id if not agv.busy else 0
        for picker, mission in assigned_pickers.items():
            actions[picker] = mission.location_id

        agv_acts = [actions[agv] for agv in agvs]
        picker_acts = [actions[picker] for picker in pickers]
        step_actions.append((agv_acts, picker_acts))

        _, _, terminated, truncated, _ = env.step(list(actions.values()))
        done = all(terminated) or all(truncated)
        timestep += 1

    return step_actions


def test_picker_parity():
    env = gym.make(ENV_ID)
    env = env.unwrapped

    for seed in SEEDS:
        ref_actions = collect_actions_from_heuristic(env, seed)

        env.reset(seed=seed)
        picker_policy = PickerHeuristicPolicy(env)

        for step_idx, (agv_acts, ref_picker_acts) in enumerate(ref_actions):
            pred_picker_acts = picker_policy.act(agv_acts)
            assert pred_picker_acts == ref_picker_acts, (
                f"seed={seed} step={step_idx}: "
                f"expected {ref_picker_acts}, got {pred_picker_acts}"
            )

            full_acts = agv_acts + pred_picker_acts
            _, _, terminated, truncated, _ = env.step(full_acts)
            if all(terminated) or all(truncated):
                break

    env.close()


def test_picker_policy_cross_episode_reuse():
    env = gym.make(ENV_ID).unwrapped
    env.reset(seed=99)
    picker_policy = PickerHeuristicPolicy(env)

    for seed in [0, 1, 2]:
        ref_actions = collect_actions_from_heuristic(env, seed)

        env.reset(seed=seed)
        picker_policy.reset()

        for step_idx, (agv_acts, ref_picker_acts) in enumerate(ref_actions):
            pred = picker_policy.act(agv_acts)
            assert pred == ref_picker_acts, (
                f"seed={seed} step={step_idx}: expected {ref_picker_acts}, got {pred}"
            )
            full_acts = agv_acts + pred
            _, _, terminated, truncated, _ = env.step(full_acts)
            if all(terminated) or all(truncated):
                break

    env.close()
