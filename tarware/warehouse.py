import random
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import networkx as nx
import numpy as np
import pyastar2d
from gymnasium import spaces
from tarware.definitions import (Action, AgentType, Direction,
                                 RewardType, CollisionLayers)
from tarware.spaces import observation_map
from tarware.utils import find_sections, get_next_micro_action

_FIXING_CLASH_TIME = 4
_STUCK_THRESHOLD = 5

class Entity:
    def __init__(self, id_: int, x: int, y: int):
        self.id = id_
        self.prev_x = None
        self.prev_y = None
        self.x = x
        self.y = y

class Agent(Entity):
    counter = 0

    def __init__(self, x: int, y: int, dir_: Direction, agent_type: AgentType):
        Agent.counter += 1
        super().__init__(Agent.counter, x, y)
        self.dir = dir_
        self.req_action: Optional[Action] = None
        self.carrying_shelf: Optional[Shelf] = None
        self.canceled_action = None
        self.has_delivered = False
        self.path = None
        self.busy = False
        self.fixing_clash = 0
        self.type = agent_type
        self.target = 0
        self.task_start_step: Optional[int] = None

    def req_location(self, grid_size) -> Tuple[int, int]:
        if self.req_action != Action.FORWARD:
            return self.x, self.y
        elif self.dir == Direction.UP:
            return self.x, max(0, self.y - 1)
        elif self.dir == Direction.DOWN:
            return self.x, min(grid_size[0] - 1, self.y + 1)
        elif self.dir == Direction.LEFT:
            return max(0, self.x - 1), self.y
        elif self.dir == Direction.RIGHT:
            return min(grid_size[1] - 1, self.x + 1), self.y

        raise ValueError(
            f"Direction is {self.dir}. Should be one of {[v for v in Direction]}"
        )

    def req_direction(self) -> Direction:
        wraplist = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        if self.req_action == Action.RIGHT:
            return wraplist[(wraplist.index(self.dir) + 1) % len(wraplist)]
        elif self.req_action == Action.LEFT:
            return wraplist[(wraplist.index(self.dir) - 1) % len(wraplist)]
        else:
            return self.dir

class Shelf(Entity):
    counter = 0

    def __init__(self, x, y):
        Shelf.counter += 1
        super().__init__(Shelf.counter, x, y)

class StuckCounter:
    def __init__(self, position: Tuple[int, int]):
        self.position = position
        self.count = 0

    def update(self, new_position: Tuple[int, int]):
        if new_position == self.position:
            self.count += 1
        else:
            self.count = 0
            self.position = new_position

    def reset(self, position=None):
        self.count = 0
        if position:
            self.position = position

class Warehouse(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        shelf_columns: int,
        column_height: int,
        shelf_rows: int,
        num_agvs: int,
        num_pickers: int,
        request_queue_size: int,
        max_inactivity_steps: Optional[int],
        max_steps: Optional[int],
        reward_type: RewardType,
        normalised_coordinates: bool=False,
        observation_type: str = "global",
    ):
        """The robotic warehouse environment

        Creates a grid world where multiple agents (robots)
        are supposed to collect shelfs, bring them to a goal
        and then return them.
        .. note:
            The grid looks like this:

            shelf
            columns
                vv
            ----------
            -XX-XX-XX-        ^
            -XX-XX-XX-  Column Height
            -XX-XX-XX-        v
            ----------
            -XX----XX-   <\
            -XX----XX-   <- Shelf Rows
            -XX----XX-   </
            ----------
            ----GG----

            G: is the goal positions where agents are rewarded if
            they bring the correct shelfs.

            The final grid size will be
            height: (column_height + 1) * shelf_rows + 2
            width: (2 + 1) * shelf_columns + 1

            The bottom-middle column will be removed to allow for
            robot queuing next to the goal locations

        :param shelf_columns: Number of columns in the warehouse
        :type shelf_columns: int
        :param column_height: Column height in the warehouse
        :type column_height: int
        :param shelf_rows: Number of columns in the warehouse
        :type shelf_rows: int
        :param num_agvs: Number of spawned and controlled agv
        :type num_agvs: int
        :param num_pickers: Number of spawned and controlled pickers
        :type num_pickers: int
        :param request_queue_size: How many shelfs are simultaneously requested
        :type request_queue_size: int
        :param max_inactivity: Number of steps without a delivered shelf until environment finishes
        :type max_inactivity: Optional[int]
        :param reward_type: Specifies if agents are rewarded individually or globally
        :type reward_type: RewardType
        :param observation_type: Specifies type of observations
        :type normalised_coordinates: str
        :param normalised_coordinates: Specifies whether absolute coordinates should be normalised
            with respect to total warehouse size
        :type normalised_coordinates: bool
        """

        self.goals: List[Tuple[int, int]] = []

        self.num_agvs = num_agvs
        self.num_pickers = num_pickers
        self.num_agents = num_agvs + num_pickers

        self._make_layout_from_params(shelf_columns, shelf_rows, column_height)
        # If no Pickers are generated, AGVs can perform picks independently
        if num_pickers > 0:
            self._agent_types = [AgentType.AGV for _ in range(num_agvs)] + [AgentType.PICKER for _ in range(num_pickers)]
        else:
            self._agent_types = [AgentType.AGENT for _ in range(self.num_agents)]

        self.max_inactivity_steps: Optional[int] = max_inactivity_steps
        self.reward_type = reward_type
        self._cur_inactive_steps = None
        self._cur_steps = 0
        self.max_steps = max_steps

        self.action_size = len(self.action_id_to_coords_map) + 1
        self.action_space = spaces.Tuple(tuple(self.num_agents * [spaces.Discrete(self.action_size)]))

        self.observation_space_mapper = observation_map[observation_type](
            self.num_agvs,
            self.num_pickers,
            self.grid_size,
            len(self.action_id_to_coords_map)-len(self.goals),
            normalised_coordinates,
        )
        self.observation_space = spaces.Tuple(tuple(self.observation_space_mapper.ma_spaces))

        self.request_queue_size = request_queue_size
        self.request_queue = []
        self.rack_groups = find_sections(list([loc for loc in self.action_id_to_coords_map.values() if (loc[1], loc[0]) not in self.goals]))
        self.agents: List[Agent] = []
        self.stuck_counters = []
        self.renderer = None

    @property
    def targets_agvs(self):
        return [agent.target for agent in self.agents[:self.num_agvs]]

    @property
    def targets_pickers(self):
        return [agent.target for agent in self.agents[self.num_agvs:]]

    def _make_layout_from_params(self, shelf_columns: int, shelf_rows: int, column_height: int) -> None:
        """
        根據 shelf_columns、shelf_rows、column_height 自動建立 warehouse 的地圖 layout。做幾件事：
        計算整個 warehouse 的 grid_size
        初始化 self.grid
        決定哪些 row / column 是 highway lane
        建立 self.highways，標記每個格子是不是通道
        建立 self.goals，也就是底部可交付/目標位置
        建立 self.action_id_to_coords_map，把 action id 對應到地圖座標
        """
        assert shelf_columns % 2 == 1, "Only odd number of shelf columns is supported"
        self._bottom_rows = 2
        self._highway_lanes = 2
        self.column_width = 2
        self.column_height = column_height
        self.grid_size = (
            self._highway_lanes + (self.column_height + self._highway_lanes) * shelf_rows  + self._bottom_rows + 1,
            self._highway_lanes + (self.column_width  + self._highway_lanes) * shelf_columns,
        )
        self.grid = np.zeros((len(CollisionLayers), *self.grid_size), dtype=np.int32)

        def get_highway_lanes_indices(axis_size, step):
            """計算哪些 row/ column 會是通道位置"""
            return [
                i + j
                for i in range(
                    0, axis_size, step + self._highway_lanes
                )
                for j in range(self._highway_lanes)
            ]

        highway_ys = get_highway_lanes_indices(self.grid_size[0], self.column_height)
        highway_xs = get_highway_lanes_indices(self.grid_size[1], self.column_width)

        def highway_func(x, y):
            """ 
            建立 layout 時即時計算
            判斷某個座標 (x, y) 是不是 highway 
            """
            return x in highway_xs or y in highway_ys or y >= self.grid_size[0] - 1 - self._bottom_rows

        self.goals = [
            (i, self.grid_size[0] - 1)
            for i in range(self.grid_size[1]) if not i in highway_xs
        ]
        self.num_goals = len(self.goals)

        self.highways = np.zeros(self.grid_size, dtype=np.int32)
        self.action_id_to_coords_map = {i+1: (x, y) for i, (y, x) in enumerate(self.goals)}
        item_loc_index=len(self.action_id_to_coords_map)+1
        for x in range(self.grid_size[1]):
            for y in range(self.grid_size[0]):
                self.highways[y, x] = highway_func(x, y)
                if not highway_func(x, y) and (x, y) not in self.goals:
                    self.action_id_to_coords_map[item_loc_index] = (y, x)
                    item_loc_index+=1

    def _is_highway(self, x: int, y: int) -> bool:
        """ layout 建好後，直接查已經存好的 array """
        return self.highways[y, x]

    def find_path(self, start, goal: Tuple[int], agent: Tuple[int], care_for_agents: bool = True) -> List[Tuple[int]]:
        """
        Constructs a path from start to goal using the A* algorithm contidioned on a grid that integrates both the
        environment specific (shelf) and the presence of other agents as obstacles.

        If `care_for_agents` is True, the grid is adjusted to consider other agents as obtacles. However, we avoid
        situatiosns where the paths is invalidated by agents of other types waiting to cooperate with the current.
        For Pickers, the grid is further modified to ensure they can only travel through designated highways and
        access goal locations.

        Parameters:
        - care_for_agents (bool): Whether to consider other agents in the grid.
        - agent (Agent): The agent for which the path is being calculated.
        - start (tuple): The starting coordinates (x, y) of the agent.
        - goal (tuple): The goal coordinates (x, y) for the agent.

        Returns:
        - List of tuples representing the path from start to goal, or an empty list if no path is found.
        """
        grid = np.zeros(self.grid_size)
        if care_for_agents:
            # 如果 care_for_agents=True，就把目前 AGV 和 Picker 佔據的位置也加進 grid，當成障礙物
            grid += self.grid[CollisionLayers.AGVS]
            grid += self.grid[CollisionLayers.PICKERS]
        # Agents should start a path regardless if some others are waiting around the target location
        grid[goal[0], goal[1]] = 0

        if agent.type == AgentType.PICKER:
            # Pickers can only travel through the highway, but can access goal locations
            # highway = 1，非 highway = 0
            # picker 主要只能走 highway，不能像 AGV 一樣在 rack/shelf 自由穿梭
            grid += (1-self.highways)
            grid[goal[0], goal[1]] -= not self._is_highway(goal[1], goal[0])
            for i in range(self.grid_size[1]):
                grid[self.grid_size[0] - 1, i] = 1

        # If a Picker starts inside a non-highway rack location and its target is adjacent,
        # force the path to re-enter the highway first instead of cutting directly through racks.
        """
        不想要：
        Picker 在 shelf A -> 直接走到隔壁 shelf B

        想要：
        Picker 先回 highway -> 沿 highway 繞路 -> 再進入目標 shelf 位置
        """
        start_fix = (0, 0)
        if agent.type == AgentType.PICKER and  ( 
            (not self._is_highway(start[1], start[0])) and # Picker 在 rack 裡
            goal[0] == start[0] and # 同一 row
            abs(goal[1] - start[1]) == 1 # 緊鄰（1格）
            ): 
            if self._is_highway(start[1] - 1, start[0]): # 左邊是 highway?
                start_fix = (0, - 1)                     # 起點往左移一格
            if self._is_highway(start[1] + 1, start[0]): # 右邊是 highway?
                start_fix = (0, 1)                       # 起點往右移一格
            grid[start[0], start[1]] = 1                 # 封鎖原始起點

        grid[start[0]+start_fix[0], start[1]+start_fix[1]] = 0 # 確保新起點是可通行的

        """
        grid 轉換成 A* 看得懂的 -> 從有沒有東西轉換成走這一格要花幾步
        grid != 0     →  有東西的格子=True，空格=False
        map(int, ...) →  True=1（障礙）, False=0（可通行）
        """
        grid = [list(map(int, l)) for l in (grid!=0)] 
        grid = np.array(grid, dtype=np.float32)
        grid[np.where(grid == 1)] = np.inf # 障礙物 = 無限成本（走不過去）
        grid[np.where(grid == 0)] = 1      # 空格   = 成本 1（走一格算一步）

        """
        第二段：路徑裁切
        """
        astar_path = pyastar2d.astar_path(grid, np.add(start, start_fix), goal, allow_diagonal=False) # returns None if cant find path
        # astar_path = astar_path[int(np.where(astar_path == np.add(start, start_fix))[0][0]):]
        if astar_path is not None:
            astar_path = [tuple(x) for x in list(astar_path)] # convert back to other format
             # A* 路徑包含起點。正常情況跳過（agent 已在起點）；
             # start_fix 情況起點被封鎖（inf），第一格是 agent 要走去的 highway，保留
            
            astar_path = astar_path[1 - int(grid[start[0], start[1]] > 1):]
             # astar_path[1:]   ← 跳過第一格（起點，agent 已在那裡）
             # astar_path[0:]   ← 保留全部（第一格是要走去的 highway）

        if astar_path:
            return [(x, y) for y, x in astar_path]
        else:
            return []

    def _recalc_grid(self) -> None:
        """
        重新計算目前 warehouse 裡每一層 grid 的佔用狀態
        """
        self.grid.fill(0)

        carried_shelf_ids = {agent.carrying_shelf.id for agent in self.agents if agent.carrying_shelf}
        for shelf in self.shelfs:
            if shelf.id not in carried_shelf_ids:
                self.grid[CollisionLayers.SHELVES, shelf.y, shelf.x] = shelf.id
        for agent in self.agents:
            layer = CollisionLayers.PICKERS if agent.type == AgentType.PICKER else CollisionLayers.AGVS
            self.grid[layer, agent.y, agent.x] = agent.id
            if agent.carrying_shelf:
                 self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = agent.carrying_shelf.id

    def get_carrying_shelf_information(self):
        return [agent.carrying_shelf != None for agent in self.agents[:self.num_agvs]]

    def get_shelf_request_information(self) -> np.ndarray[int]:
        request_item_map = np.zeros(len(self.shelfs))
        requested_shelf_ids = [shelf.id for shelf in self.request_queue]
        for id_, coords in self.action_id_to_coords_map.items():
            if (coords[1], coords[0]) not in self.goals:
                if self.grid[CollisionLayers.SHELVES, coords[0], coords[1]] in requested_shelf_ids:
                    request_item_map[id_ - len(self.goals) - 1] = 1
        return request_item_map

    def get_empty_shelf_information(self) -> np.ndarray[int]:
        empty_item_map = np.zeros(len(self.shelfs))
        for id_, coords in self.action_id_to_coords_map.items():
            if (coords[1], coords[0]) not in self.goals:
                if self.grid[CollisionLayers.SHELVES, coords[0], coords[1]] == 0 and (
                    self.grid[CollisionLayers.CARRIED_SHELVES, coords[0], coords[1]] == 0
                    or self.agents[
                        self.grid[CollisionLayers.AGVS, coords[0], coords[1]] - 1
                    ].req_action
                    not in [Action.NOOP, Action.TOGGLE_LOAD]
                ):
                    empty_item_map[id_ - len(self.goals) - 1] = 1
        return empty_item_map

    # TODO : add to note
    def attribute_macro_actions(self, macro_actions: List[int]) -> Tuple[int, int]:
        """
        這個 function 不是真的移動 agent，
        它只是決定每個 agent 這一步的 agent.req_action，
        真正執行移動是在後面的 execute_micro_actions()
        """

        # Count how many AGV / Picker movement steps are planned in this environment step.
        agvs_distance_travelled = 0
        pickrs_distance_travelled = 0

        """
        # Logic for Macro Actions
        # Each agent receives one macro action.
        # A macro action is a target location id, not a direct movement command.
        """
        for agent, macro_action in zip(self.agents, macro_actions):
            # Initialize action for step
            ## Default action for every step is doing nothing.
            ## Later logic may overwrite this with FORWARD / LEFT / RIGHT / TOGGLE_LOAD.
            agent.req_action = Action.NOOP

            # Collision avoidance logic
            ## If this agent is currently in a collision-fixing cooldown,
            ## decrease the remaining cooldown time.
            if agent.fixing_clash > 0:
                agent.fixing_clash -= 1

            # Case 1: agent is idle, so it can accept a new macro action.
            if not agent.busy:
                agent.target = 0 # Clear previous target because the agent is no longer working on it.
                
                # macro_action == 0 means no new target is assigned.
                if macro_action != 0: # 代表有新 target
                    # Convert the macro action id into a target coordinate,
                    # then find a path from current position to that target.
                    # care_for_agents=False means path planning ignores current agent positions.
                    agent.path = self.find_path((agent.y, agent.x), self.action_id_to_coords_map[macro_action], agent, care_for_agents=False)
                   
                    # If pathfinding succeeds, mark the agent as busy
                    # and prepare its first low-level movement action.
                    if agent.path:
                        agent.busy = True
                        agent.target = macro_action

                        # If AGV is sent to a shelf/item location,
                        # record the start time for later delivery-time statistics.
                        if agent.type == AgentType.AGV and macro_action > self.num_goals:
                            agent.task_start_step = self._cur_steps
                        
                        # Convert the first coordinate in the path into a micro action.
                        # Example: turn left, turn right, or move forward.
                        # agent.path[0] : path 裡面的下一個座標 e.g agent.path[0] = (5,3)
                        # agent 不能直接走過去(5,3)，而是要透過執行一步的 micro action 來做到 -> 執行 get_next_micro_action()
                        agent.req_action = get_next_micro_action(agent.x, agent.y, agent.dir, agent.path[0])
                        
                        # stuck_counters: 用來偵測 agent 有沒有一直停在同一個地方
                        # 因為 agent pathfinding success，代表 agent 拿到新的任務 -> stuck_counters reset
                        self.stuck_counters[agent.id - 1].reset((agent.x, agent.y))
        
            # Case 2: agent is already busy, so it should continue its existing path.
            else:
                # Check if agent finished the given path, if not continue the path
                if agent.path == []: # Path is empty, agent arrived target.
                    
                    # AGV / single AGENT 
                    ## performs TOGGLE_LOAD after reaching the target.
                    ## This means load or unload shelf depending on current carrying state.
                    if agent.type in [AgentType.AGV, AgentType.AGENT]:
                        agent.req_action = Action.TOGGLE_LOAD
                    
                    # Picker 
                    ## does not load/unload directly here.
                    ## Once it reaches its target, it becomes idle again.
                    if agent.type == AgentType.PICKER:
                        agent.busy = False
                
                # If path still has coordinates, continue moving along the path.
                else:
                    # Convert the next path coordinate into this step's micro action.
                    agent.req_action = get_next_micro_action(agent.x, agent.y, agent.dir, agent.path[0])
                    
                    # Count planned movement distance by agent type.
                    agvs_distance_travelled += int(agent.type == AgentType.AGV)
                    pickrs_distance_travelled += int(agent.type == AgentType.PICKER)
                
                # If only one coordinate remains, the agent is about to reach the final target.    
                if len(agent.path) == 1:
                    
                    # If agent is at the end of a path and carrying a shelf and the target location is already occupied, restart agent
                    ## If an agent is carrying a shelf but the target location already has a shelf,
                    ## stop and mark it as not busy so it can be reassigned later.
                    if agent.carrying_shelf and self.grid[CollisionLayers.SHELVES, agent.path[-1][1], agent.path[-1][0]]:
                        agent.req_action = Action.NOOP
                        agent.busy = False
                    
                    # Logic for Pickers to load shelves if AGV is present at location or wait otherwise
                    # Picker-specific cooperation logic:
                    # Picker 只有在 AGV=TOGGLE_LOAD + at the target location 時才進行 interaction
                    if agent.type == AgentType.PICKER:
                        
                        # If no AGV is at the target, or the AGV is not loading/unloading,
                        # the Picker waits instead of moving/finishing.
                        if (
                            self.grid[CollisionLayers.AGVS, agent.path[-1][1], agent.path[-1][0]] == 0
                            or self.agents[self.grid[CollisionLayers.AGVS, agent.path[-1][1], agent.path[-1][0]]- 1].req_action != Action.TOGGLE_LOAD
                        ):
                            agent.req_action = Action.NOOP
                        
                        # If an AGV is at the target and is TOGGLE_LOAD,
                        # reset the Picker stuck counter because this waiting is intentional cooperation.
                        elif (
                            self.grid[CollisionLayers.AGVS, agent.path[-1][1], agent.path[-1][0]] != 0
                            and self.agents[self.grid[CollisionLayers.AGVS, agent.path[-1][1], agent.path[-1][0]] - 1].req_action == Action.TOGGLE_LOAD
                        ):
                            self.stuck_counters[agent.id - 1].reset((agent.x, agent.y))
        
        # Return movement statistics for this step.
        return agvs_distance_travelled, pickrs_distance_travelled

    # TODO : Add to note
    def resolve_move_conflict(self, agent_list):
        """
        StuckCounter 追蹤每個 agent 有沒有在原地不動。
        超過 _STUCK_THRESHOLD = 5 步就強制重新規劃路徑，再超過就直接取消任務（agent.busy = False）
        ---
        commited_agents 代表：這一個 step 被允許執行原本 req_action 的 agent
        在 resolve_move_conflict() 裡，每個 agent 先在 attribute_macro_actions() 得到一個想做的動作：agent.req_action
        -> 但不是每個動作都可以真的做，要看會不會撞車
        """
        
        # 建立「每個 agent 從哪裡想移到哪裡」的 directed graph。
        ## Store the ids of agents whose requested movement is allowed to execute.
        commited_agents = set()

        ## Build a directed graph of intended movements for this step.
        ## Each edge means: current position -> requested next position.
        G = nx.DiGraph()

        ## Add one movement edge for each agent.
        ## If the agent is not moving forward, req_location() returns its current location.
        for agent in agent_list:
            start = agent.x, agent.y
            target = agent.req_location(self.grid_size)
            G.add_edge(start, target)
        
        # Split the movement graph into independent connected components.
        # Agents in different components do not directly affect each other's movement
        # 把 graph 分成幾個互相有關的區塊，不相關的 agent 可以分開處理
        wcomps = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
        for comp in wcomps:
            try:
                # 如果有 cycle，代表一群 agent 可能在輪流讓位
                # if we find a cycle in this component we have to
                # commit all nodes in that cycle, and nothing else
                cycle = nx.algorithms.find_cycle(comp)
                
                # 2 Agent Cycle
                if len(cycle) == 2:
                    # we have a situation like this: [A] <-> [B]
                    # which is physically impossible.會互撞 -> so skip
                    continue
                
                # 三人以上的 cycle 是可以一起動的 -> 把 cycle 裡的 agent 加進 commited_agents
                # A -> B 的位置
                # B -> C 的位置
                # C -> A 的位置
                for edge in cycle:
                    start_node = edge[0]
                    
                    # Check whether an AGV is currently located at this start node.
                    agent_id = self.grid[CollisionLayers.AGVS, start_node[1], start_node[0]] # 把「graph 裡的座標」轉回「實際站在那個座標上的 agent id」。
                    # action = self.agents[agent_id - 1].req_action
                    # print(f"{agent_id}: C {cycle} {action}")

                    # If an AGV is found, allow AGV to execute its requested action.
                    if agent_id > 0:
                        commited_agents.add(agent_id)
                        continue
                    
                    # If no AGV is found, check whether a Picker is located there.
                    picker_id = self.grid[CollisionLayers.PICKERS, start_node[1], start_node[0]]

                    # If a Picker is found, allow that Picker to execute its requested action.
                    if picker_id > 0:
                        commited_agents.add(picker_id)
                        continue
            
            except nx.NetworkXNoCycle:
                # If the component has no cycle, it is a directed acyclic movement chain.
                # Choose the longest chain of movements to execute.
                # 找 DAG 裡最長的一串移動鏈，讓這串 agent 優先動，其他的後面會被設成 NOOP
                longest_path = nx.algorithms.dag_longest_path(comp)

                # Commit every agent currently located on that longest movement chain.
                for x, y in longest_path:
                    """
                    這邊處理的方法和上面有cycle一樣
                    """
                    # Check whether an AGV is located at this graph node
                    agent_id = self.grid[CollisionLayers.AGVS, y, x]
                    
                    # If an AGV is found, allow it to execute its requested action.
                    if agent_id:
                        commited_agents.add(agent_id)
                        continue
                    picker_id = self.grid[CollisionLayers.PICKERS, y, x]
                    if picker_id:
                        commited_agents.add(picker_id)

        # Count how many movement clashes are detected in this step.
        clashes = 0

        # Compare every pair of agents to catch direct movement conflicts.
        for agent in agent_list:
            for other in agent_list:

                # Skip comparing the agent with itself.
                if agent.id != other.id:
                    
                    # Compute where each agent wants to be after this step
                    agent_new_x, agent_new_y = agent.req_location(self.grid_size)
                    other_new_x, other_new_y = other.req_location(self.grid_size)
                    
                    # Clash fixing logic
                    ## Only agents with a path need clash handling here.
                    ## A clash happens if this agent moves into:
                    ## 1. the other agent's current position, or
                    ## 2. the other agent's requested next position.
                    if agent.path and ((agent_new_x, agent_new_y) in [(other.x, other.y), (other_new_x, other_new_y)]):

                        # 特殊情況: AGV/Picker 在 rack 協作 -> 位置不是 highway，且一個是 AGV、一個是 picker
                        # e.g picker 在 shelf 旁邊，AGV 過來 load/unload
                        # If we are in a rack and one of the agents is a picker we ignore clashses, assumed behaviour is Picker is loading
                        if not self._is_highway(agent_new_x, agent_new_y) and (agent.type == AgentType.PICKER or other.type == AgentType.PICKER) and agent.type != other.type:
                            # Allow Pickers to step over AGVs (if no other Picker at that shelf location) or AGVs to step over Pickers (if no other AGV at that shelf location)
                            if ((agent.type == AgentType.PICKER and self.grid[CollisionLayers.PICKERS, agent_new_y, agent_new_x] in [0, agent.id])
                                or (agent.type == AgentType.AGV and self.grid[CollisionLayers.AGVS, agent_new_y, agent_new_x] in [0, agent.id])):
                                commited_agents.add(agent.id)
                                continue
                        
                        # Case 1:
                        # Agent's next action wants to move into other agent's current cell.
                        if (agent_new_x, agent_new_y) == (other.x, other.y):
                            agent.req_action = Action.NOOP # Stop this agent for now to avoid moving into an occupied cell.

                            # 檢查 other 是否真的會讓開
                            # If the other agent stays, swaps, or also targets this cell,
                            # the conflict is not naturally resolved.
                            if (other_new_x, other_new_y) in [(agent.x, agent.y), (agent_new_x, agent_new_y)] and not other.req_action in (Action.LEFT, Action.RIGHT):
                                
                                # 如果 other 沒有安全離開，且不是只是在原地轉向，就視為 clash
                                if other.fixing_clash == 0: 
                                    clashes+=1
                                    agent.fixing_clash = _FIXING_CLASH_TIME # Agent start time for clash fixing
                                    new_path = self.find_path((agent.y, agent.x), (agent.path[-1][1] ,agent.path[-1][0]), agent) # 開始找路
                                    if new_path != []: 
                                        # If the agent can find an alternative path, assign it if not let the other solve the clash
                                        agent.path = new_path
                                    else:
                                    # If no alternative path exists, cancel the fixing cooldown.
                                    # The agent will simply remain stopped for this step.
                                        agent.fixing_clash = 0
                        # Case 2:
                        # 兩個 agent 想去同一格 -> 讓兩個 agent 都停下
                        elif (agent_new_x, agent_new_y) == (other_new_x, other_new_y) and (agent_new_x, agent_new_y) != (agent.x, agent.y):
                            if agent.fixing_clash == 0 and other.fixing_clash == 0:
                                # 如果兩個 agent 都沒有在 handle clash，stop this agent and mark it as fixing collision
                                agent.req_action = Action.NOOP # If the agent's actions leads them in the position of another STOP
                                agent.fixing_clash = _FIXING_CLASH_TIME  # Agent wait one step while the other moves into place
        
        # Convert committed agent ids into actual Agent objects.
        commited_agents = set([self.agents[id_ - 1] for id_ in commited_agents])
        
        # Agents not committed by the graph logic are not allowed to move this step.
        failed_agents = set(agent_list) - commited_agents

        # Force all failed agents to stay still.
        for agent in failed_agents:
            agent.req_action = Action.NOOP
        
        # Return the number of detected clashes for logging/statistics.
        return clashes

    def resolve_stuck_agents(self) -> None:
        """
        偵測「busy 但一直卡在同一個位置」的 agent。
        如果 agent 卡太久，先嘗試重新找路；如果還是卡住，就讓它停止目前任務。
        ---
        # This can happen when their goal is occupied after reaching their last step/re-calculating a path
        ---
        1. 找出需要檢查 stuck 的 busy agents
        2. 更新每個 agent 的 stuck counter
        3. 如果剛開始卡住，先嘗試重新 find_path()
        4. 如果卡太久，放棄目前任務，agent.busy = False
        """
        
        overall_stucks = 0 # Count how many agents are finally considered stuck in this step.

        # Select agents that should be checked for being stuck.
        # 只有 busy agents 才要被 checked -> 因為 idle agent 不會移動
        # NOTE : 如果之後設計要讓 agent explore 的話，那應該要更改這邊的設計 -> 待討論
        moving_agents = [
            agent
            for agent in self.agents
            if agent.busy
            and agent.req_action not in (Action.LEFT, Action.RIGHT) # Turning left/right does not change position, 不算 stuck.
            and (agent.req_action!=Action.TOGGLE_LOAD or (agent.x, agent.y) in self.goals) # Don't count loading or changing directions / if at goal
        ]

        # Check each moving agent's stuck counter.
        for agent in moving_agents:

            agent_stuck_count = self.stuck_counters[agent.id - 1] # Get this agent's stuck counter. agent.id starts from 1, so use id - 1 as list index.
            
            # Update stuck counter with the agent's current position.
            # If the position is unchanged, the counter increases.
            # If the position changed, the counter resets to 0.
            agent_stuck_count.update((agent.x, agent.y))
            
            # Case 1: stuck_counter 還在 middle range, 可接受
            # no.1 條件式 : If the agent has stayed in the same position for more than _STUCK_THRESHOLD,
            # no.2 條件式 : 沒有卡到真的太久, try to recover by replanning.
            # NOTE: 這裡的 hyperparameter 大小會影響 stuck 的程度
            if _STUCK_THRESHOLD < agent_stuck_count.count < _STUCK_THRESHOLD + self.column_height + 2:  # Time to get out of aisle
                
                # Stop the agent for this step before trying a new path.
                agent.req_action = Action.NOOP

                # Case 1: 
                # Agent still have path, try to find a new path (from 現在位置 to target)
                if agent.path:
                    new_path = self.find_path((agent.y, agent.x), (agent.path[-1][1], agent.path[-1][0]), agent)
                    
                    
                    # If a new path is found, replace the old path.
                    # Picker should wait for AGV to arrive at destination regardless of stuck count
                    if new_path:
                        agent.path = new_path
                        if len(agent.path) == 1: # 如果只剩下 1 step -> very close to the target -> 有可能只是在等協作條件成立 -> skip resetting the stuck_counter and just wait
                            # 可能是 picker 已到但 AGV 還沒抵達，所以 picker 這輪會先被設為 NOOP
                            # 但 picler 並不是永遠等待，要在上面 if 的 stuck count 處在 middle range 時才繼續等
                            continue 
                        agent_stuck_count.reset((agent.x, agent.y)) # multiple step left(get useful path) -> reset stuck_counter from the current position.
                        continue
                
                # Case 2:
                # Agent has no path, it cannot recover by replanning.
                # Mark it as stuck and stop its current task. -> 只是使 AGV 變成 idle，之後他還是可以被重新指派新的 macro action
                else:
                    overall_stucks += 1
                    agent.busy = False
                    agent_stuck_count.reset()
            
            # Case 2: Agent Stuck_counter 卡太久了
            # give up current task, and make it idle
            if agent_stuck_count.count > _STUCK_THRESHOLD + self.column_height + 2:  # Time to get out of aisle
                overall_stucks += 1
                agent_stuck_count.reset((agent.x, agent.y)) 
                agent.req_action = Action.NOOP
                agent.busy = False

        # Return how many agents were considered stuck in this step.
        return overall_stucks

    def _execute_forward(self, agent: Agent) -> None:
        agent.x, agent.y = agent.req_location(self.grid_size)
        agent.path = agent.path[1:]
        if agent.carrying_shelf:
            agent.carrying_shelf.x, agent.carrying_shelf.y = agent.x, agent.y

    def _execute_rotation(self, agent: Agent) -> None:
        agent.dir = agent.req_direction()

    def _execute_load(self, agent: Agent, rewards: np.ndarray[int]) -> np.ndarray[int]:
        shelf_id = self.grid[CollisionLayers.SHELVES, agent.y, agent.x]
        picker_id = self.grid[CollisionLayers.PICKERS, agent.y, agent.x]
        if shelf_id:
            if (
                (agent.type == AgentType.AGV and picker_id)
                or agent.type == AgentType.AGENT
            ):
                agent.carrying_shelf = self.shelfs[shelf_id - 1]
                self.grid[CollisionLayers.SHELVES, agent.y, agent.x] = 0
                self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = shelf_id
                agent.busy = False
                # Reward picker for loading
                if self.reward_type == RewardType.GLOBAL:
                    rewards += 0.5
                elif self.reward_type == RewardType.INDIVIDUAL:
                    if agent.type == AgentType.AGENT:
                        rewards[agent.id - 1] += 0.1
                    else:
                        rewards[picker_id - 1] += 0.1
        else:
            agent.busy = False
        return rewards

    def _execute_unload(self, agent: Agent, rewards: np.ndarray[int]) -> np.ndarray[int]:
        if (agent.x, agent.y) in self.goals or self.grid[CollisionLayers.SHELVES, agent.y, agent.x] != 0:
            agent.busy = False
            return rewards
        picker_id = self.grid[CollisionLayers.PICKERS, agent.y, agent.x]
        if not self._is_highway(agent.x, agent.y):
            if (
                (agent.type == AgentType.AGV and picker_id)
                or agent.type == AgentType.AGENT
            ):
                self.grid[CollisionLayers.SHELVES, agent.y, agent.x] = agent.carrying_shelf.id
                self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = 0
                agent.carrying_shelf = None
                agent.busy = False
                agent.has_delivered = False
                # Reward picker for unloading
                if self.reward_type == RewardType.GLOBAL:
                    rewards += 0.5
                elif self.reward_type == RewardType.INDIVIDUAL:
                    if agent.type == AgentType.AGENT:
                        rewards[agent.id - 1] += 0.1
                    else:
                        rewards[picker_id - 1] += 0.1
        return rewards

    def execute_micro_actions(self, rewards: np.ndarray[int]) -> np.ndarray[int]:
        """
        FORWARD: 移動一格
        LEFT/RIGHT: 轉向
        TOGGLE_LOAD: pickup 或 unload
        """
        
        for agent in self.agents:
            if agent.req_action == Action.FORWARD:
                self._execute_forward(agent)
            elif agent.req_action in [Action.LEFT, Action.RIGHT]:
                self._execute_rotation(agent)
            elif agent.req_action == Action.TOGGLE_LOAD:
                if not agent.carrying_shelf:
                    rewards = self._execute_load(agent, rewards)
                else:
                    rewards = self._execute_unload(agent, rewards)
        return rewards

    def process_shelf_deliveries(self, rewards: np.ndarray[int]) -> np.ndarray[int]:
        """
        檢查有沒有 AGV 把「被 request 的 shelf」送到 goal
        如果有，就給 reward、更新 request queue、記錄 delivery time，並重設 inactive counter。
        """
        shelf_deliveries = 0 # 在這個 step 有幾個 shelf delivered
        delivery_travel_times = [] # Store travel times for shelves delivered in this step.

        # Check every goal / delivery location.
        # Note: lf.goals stores coordinates as (x, y),
        # but this loop names them as (y, x), so the later grid indexing looks swapped.
        for y, x in self.goals:

            # Check whether there is a carried shelf at this goal location.
            # self.grid uses [layer, row, col] = [layer, y, x],
            # so here it uses [x, y] because the loop variable names are swapped.
            shelf_id = self.grid[CollisionLayers.CARRIED_SHELVES, x, y]

            # 條件式 1: no carried shelf here, skip
            # 條件式 2: shelf is not requested, not counted as delivery
            if not shelf_id  or self.shelfs[shelf_id - 1] not in self.request_queue:
                continue
            
            # Delivered shelf is requested
            # Remove shelf from request queue and add replacement
            carried_shels = [agent.carrying_shelf for agent in self.agents if agent.carrying_shelf]

            # New request candidates are shelves that are:
            # 1. not already requested
            # 2. not currently being carried
            new_shelf_candidates = list(set(self.shelfs) - set(self.request_queue) - set(carried_shels))
            
            new_shelf_candidates.sort(key = lambda x: x.id) # Sort candidates so random choice is reproducible when using a fixed seed.

            new_request = np.random.choice(new_shelf_candidates) # Choose a new shelf request.
            
            self.request_queue[self.request_queue.index(self.shelfs[shelf_id - 1])] = new_request # Replace the delivered shelf in the request queue with the new request.


            """ 紀錄把 shelf 送達的 agent id 要給 reward """
            # Find the AGV currently at the goal location.
            # Grid stores agent ids, and agent ids start from 1,
            # so subtract 1 to index self.agents
            agent = self.agents[self.grid[CollisionLayers.AGVS, x, y] - 1]

            # NOTE: Delivery Reward Design 
            # Avoid giving reward repeatedly while the AGV remains on the goal. -> 怕 agent 一直站在原地，使得env 以為他一直重複送達
            if not agent.has_delivered:
                agent.has_delivered = True
                if self.reward_type == RewardType.GLOBAL:
                    rewards += 1
                elif self.reward_type == RewardType.INDIVIDUAL:
                    rewards[agent.id - 1] += 1
                
                # 紀錄 how many environment steps the AGV took to finish a delivery.
                # task_start_step is set at the *start* of the step where the macro
                # action was assigned (before _cur_steps is incremented at end-of-step).
                # Delivery is also detected before that increment, so the raw difference
                # counts step *boundaries* — off by 1 vs the inclusive step count.
                # +1 corrects it: an AGV that delivers in the same step as assignment
                # consumed 1 environment step, not 0.
                if agent.task_start_step is not None:
                    delivery_travel_times.append(
                        self._cur_steps - agent.task_start_step + 1
                    )
                    agent.task_start_step = None
            
            # INDICATOR: 紀錄有幾個 shelf 了
            shelf_deliveries += 1

        # INDICATOR: 紀錄已經連續幾個 step 沒有成功 delivery
        if shelf_deliveries:
            self._cur_inactive_steps = 0
        else: # If no shelf was delivered, increase inactivity counter.
            self._cur_inactive_steps += 1

        return rewards, shelf_deliveries, delivery_travel_times

    def reset(self, seed=None, options=None)-> Tuple:
        # Reset counters
        Shelf.counter = 0
        Agent.counter = 0
        self._cur_inactive_steps = 0
        self._cur_steps = 0

        # Set seed
        self.seed(seed)

        # Make the shelfs
        self.shelfs = [
            Shelf(x, y)
            for y, x in zip(
                np.indices(self.grid_size)[0].reshape(-1),
                np.indices(self.grid_size)[1].reshape(-1),
            )
            if not self._is_highway(x, y)
        ]
        self._higway_locs = np.array([(y, x) for y, x in zip(
                np.indices(self.grid_size)[0].reshape(-1),
                np.indices(self.grid_size)[1].reshape(-1),
            ) if self._is_highway(x, y)])

        # Spawn agents on higwahy locations
        agent_loc_ids = np.random.choice(
            np.arange(len(self._higway_locs)),
            size=self.num_agents,
            replace=False,
        )
        agent_locs = [self._higway_locs[agent_loc_ids, 0], self._higway_locs[agent_loc_ids, 1]]
        # and direction
        agent_dirs = np.random.choice([d for d in Direction], size=self.num_agents)
        self.agents = [
            Agent(x, y, dir_, agent_type = agent_type)
            for y, x, dir_, agent_type in zip(*agent_locs, agent_dirs, self._agent_types)
        ]

        self.stuck_counters = [StuckCounter((agent.x, agent.y)) for agent in self.agents]
        self._recalc_grid()

        self.request_queue = list(
            np.random.choice(self.shelfs, size=self.request_queue_size, replace=False)
        )
        self.observation_space_mapper.extract_environment_info(self)
        return tuple([self.observation_space_mapper.observation(agent) for agent in self.agents])

    def step(
        self, macro_actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], List[bool], Dict]:
        """
        Execute one environment step.

        macro_actions 是 policy 給 environment 的高階動作。
        每個 macro action 代表一個 agent 的目標位置 ID，
        不是直接的 LEFT / RIGHT / FORWARD 這種低階移動指令。

        這個 function 會完成一次 environment step：
        1. 把 macro action 轉成 micro action
        2. 處理碰撞
        3. 處理卡住的 agent
        4. 執行移動 / 轉向 / 搬貨
        5. 計算 reward
        6. 回傳新的 observation、reward、done flags 和 info
        """

        # Attribute macro actions to agents and resolve conflicts
        # 把每個 agent 的 macro action 轉成實際要執行的 micro action -> 只設定 agent.req_action，不會真的移動 agent
        agvs_distance_travelled, pickers_distance_travelled = self.attribute_macro_actions(macro_actions)

        # 在真正執行動作前，先檢查 agent 之間會不會發生 clashes
        # 如果某些 agent 會撞在一起， resolve_move_conflict 會把 req_action 改成 NOOP，或是重新規劃 path
        clashes_count = self.resolve_move_conflict(self.agents)

        # 檢查 busy agent 是否卡在同一個位置太久。
        # 如果 agent 卡住，這裡可能會重新規劃 path，取消目前任務，讓 agent 變回 idle
        stucks_count = self.resolve_stuck_agents()

        # NOTE: Reward Array 設計
        # Initialized Reward Array，every agent 一個 reward -> 初始都是 0
        rewards = np.zeros(self.num_agents)
        # Apply penalty for inactivity
        rewards -= 0.001
        # Execute micro actions
        rewards = self.execute_micro_actions(rewards)
        # Process shelf deliveries
        rewards, shelf_deliveries, delivery_travel_times = self.process_shelf_deliveries(rewards)

        self._recalc_grid() # 重新計算 warehouse grid 每層的狀態
        self._cur_steps += 1 # environment step counter 加 1。
        
        # 判斷 episode 是否結束，條件有兩種:
        # 1. 太久沒有成功 delivery
        # 2. 已經達到最大 step 數
        if (
            self.max_inactivity_steps
            and self._cur_inactive_steps >= self.max_inactivity_steps
        ) or (self.max_steps and self._cur_steps >= self.max_steps):
            terminateds = truncateds = self.num_agents * [True]
        else:
            terminateds = truncateds =  self.num_agents * [False]

        # 先更新 observation mapper 裡的環境資訊，再依照每個 agent 產生新的 observation。
        self.observation_space_mapper.extract_environment_info(self)
        new_obs = tuple([self.observation_space_mapper.observation(agent) for agent in self.agents])

        # 把這個 step 的資訊整理成 info dictionary
        # 這些資料通常用來 logging、debug 或分析訓練結果
        info = self._build_info(
            agvs_distance_travelled,
            pickers_distance_travelled,
            clashes_count,
            stucks_count,
            shelf_deliveries,
            delivery_travel_times,
        )

        # 回傳 Gymnasium step 格式：
        # observation, reward, terminated, truncated, info
        return new_obs, list(rewards), terminateds, terminateds, info

    def _build_info(
        self,
        agvs_distance_travelled: int,
        pickers_distance_travelled: int,
        clashes_count: int,
        stucks_count:  int,
        shelf_deliveries: int,
        delivery_travel_times: List[int],
    ) -> Dict[str, np.ndarray]:
        info = {}
        agvs_idle_time = sum([int(agent.req_action in (Action.NOOP, Action.TOGGLE_LOAD)) for agent in self.agents[:self.num_agvs]]) 
        pickers_idle_time = sum([int(agent.req_action in (Action.NOOP, Action.TOGGLE_LOAD)) for agent in self.agents[self.num_agvs:]])
        info["vehicles_busy"] = [agent.busy for agent in self.agents]
        info["shelf_deliveries"] = shelf_deliveries
        info["clashes"] = clashes_count
        info["stucks"] = stucks_count
        info["agvs_distance_travelled"] = agvs_distance_travelled
        info["pickers_distance_travelled"] = pickers_distance_travelled
        info["agvs_idle_time"] = agvs_idle_time
        info["pickers_idle_time"] = pickers_idle_time
        info["delivery_travel_times"] = delivery_travel_times
        return info

    def compute_valid_action_masks(self, pickers_to_agvs=True, block_conflicting_actions=True):
        requested_items = self.get_shelf_request_information()
        empty_items = self.get_empty_shelf_information()
        carrying_shelf_info = self.get_carrying_shelf_information()
        targets_agvs = [target - len(self.goals) - 1 for target in self.targets_agvs if target > len(self.goals)]
        targets_pickers = [target - len(self.goals) - 1 for target in self.targets_pickers if target > len(self.goals)]
        # Compute valid location list for AGVs
        valid_location_list_agvs = np.array([
            empty_items if carrying_shelf else requested_items for carrying_shelf in carrying_shelf_info
        ])
        # Compute valid location list for Pickers
        if pickers_to_agvs:
            valid_location_list_pickers = np.zeros(len(self.action_id_to_coords_map) - len(self.goals))
            valid_location_list_pickers[targets_agvs] = 1
        else:
            valid_location_list_pickers = requested_items
        # Mask out conflicting actions for agents of the same type
        if block_conflicting_actions:
            valid_location_list_agvs[:, targets_agvs] = 0
            valid_location_list_pickers[targets_pickers] = 0
        valid_action_masks = np.ones((self.num_agents, self.action_size))
        valid_action_masks[:self.num_agvs,  1 + len(self.goals):] = valid_location_list_agvs
        valid_action_masks[:self.num_agvs,  1 : 1 + len(self.goals)] = np.repeat(np.expand_dims(np.array(carrying_shelf_info), 1), len(self.goals), axis=1)
        valid_action_masks[self.num_agvs:,  1 + len(self.goals):] = valid_location_list_pickers
        valid_action_masks[self.num_agvs:,  1 : 1 + len(self.goals)] = 0
        return valid_action_masks

    def render(self, mode="human"):
        if not self.renderer:
            from tarware.rendering import Viewer
            self.renderer = Viewer(self.grid_size)
        return self.renderer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.renderer:
            self.renderer.close()

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
