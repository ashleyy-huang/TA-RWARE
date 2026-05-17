from collections import OrderedDict

from tarware.heuristic import Mission, MissionType
from tarware.utils.utils import flatten_list, split_list
from tarware.warehouse import Agent, AgentType


class PickerHeuristicPolicy:
    """Heuristic picker controller for use during AGV RL training.

    Replicates the picker assignment logic from heuristic_episode so that
    RL-trained AGVs can be paired with a deterministic picker baseline.

    Key insight: the heuristic assigns pickers by iterating over the full
    assigned_agvs dict each step — not just the actions issued this step.
    We mirror this by inspecting each AGV's current target + carrying_shelf
    state to infer whether its active mission needs a picker.
    """

    def __init__(self, env):
        self.env = env
        self.assigned_pickers: OrderedDict[Agent, Mission] = OrderedDict()
        self._capture_agents()
        sections = env.rack_groups
        raw = split_list(sections, len(self.pickers))
        self.picker_sections = [flatten_list(l) for l in raw]

    def _capture_agents(self) -> None:
        self.agvs = [a for a in self.env.agents if a.type == AgentType.AGV]
        self.pickers = [a for a in self.env.agents if a.type == AgentType.PICKER]

    def reset(self) -> None:
        self._capture_agents()
        self.assigned_pickers.clear()

    def _mission_type_for_agv(self, agv, target_id: int):
        """Infer mission type from AGV state and target.

        Returns MissionType or None if this AGV needs no picker.
        """
        env = self.env
        goals_set = set(env.goals)  # (y, x) tuples
        coord = env.action_id_to_coords_map[target_id]  # (y, x)
        coord_xy = (coord[1], coord[0])  # (x, y)

        if agv.carrying_shelf:
            if coord_xy in goals_set:
                return None  # DELIVERING — no picker needed
            else:
                return MissionType.RETURNING
        else:
            return MissionType.PICKING

    def act(self, agv_macro_actions: list) -> list:
        """Return picker macro actions given the AGV actions for this step.

        Mirrors heuristic.py execution order (lines 117-135):
          1. Assign pickers for active AGV PICKING / RETURNING missions.
          2. Pop pickers that have arrived (they become available again).
          3. Return per-picker action (location_id or 0 for NOOP).
        """
        env = self.env

        # Step 1: replicate heuristic.py lines 117-124.
        # For each AGV that has an active PICKING or RETURNING mission,
        # assign the relevant section picker if not already assigned.
        for i, agv in enumerate(self.agvs):
            # Determine the effective target this step
            action = agv_macro_actions[i]
            if action != 0:
                # AGV is receiving a new assignment this step
                target_id = action
            elif agv.busy and agv.target != 0:
                # AGV is already working towards a previously assigned target
                target_id = agv.target
            else:
                continue

            mission_type = self._mission_type_for_agv(agv, target_id)
            if mission_type is None:
                continue

            coord = env.action_id_to_coords_map[target_id]  # (y, x)
            coord_yx = (coord[0], coord[1])

            in_section = [coord_yx in sec for sec in self.picker_sections]
            if True not in in_section:
                continue
            section_idx = in_section.index(True)
            relevant_picker = self.pickers[section_idx]

            if relevant_picker not in self.assigned_pickers:
                self.assigned_pickers[relevant_picker] = Mission(
                    mission_type,
                    target_id,
                    coord[1],  # location_x
                    coord[0],  # location_y
                    assigned_time=0,
                )

        # Step 2: pop pickers that have arrived (heuristic.py lines 125-129).
        # Must happen AFTER assignment so that a picker arriving exactly at the
        # newly-assigned location this step is immediately cleared (action = 0).
        arrived = [
            p for p, m in self.assigned_pickers.items()
            if p.x == m.location_x and p.y == m.location_y
        ]
        for p in arrived:
            del self.assigned_pickers[p]

        # Step 3: build action list for all pickers
        return [
            self.assigned_pickers[p].location_id if p in self.assigned_pickers else 0
            for p in self.pickers
        ]
