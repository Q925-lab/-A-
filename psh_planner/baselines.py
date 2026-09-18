"""Fair online baselines sharing the same observation stream as PSH."""

import heapq
from itertools import count
from typing import Sequence, Tuple

import numpy as np

from config import DIRECTIONS
from d_star_lite import DStarLite

Cell = Tuple[int, int]
State = Tuple[int, int, int]


def _route_action(state: State, next_cell: Cell) -> str:
    r, c, direction = state
    dr, dc = next_cell[0] - r, next_cell[1] - c
    desired = next(k for k, delta in DIRECTIONS.items() if delta == (dr, dc))
    diff = (desired - direction) % 4
    if diff == 0:
        return "forward"
    if diff == 2:
        return "backward"
    return "turn_right" if diff == 1 else "turn_left"


def _astar_cells(grid, start: Cell, goal: Cell):
    open_heap = []
    serial = count()
    heapq.heappush(open_heap, (0, next(serial), start))
    cost = {start: 0}
    parent = {}
    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current == goal:
            path = [current]
            while current in parent:
                current = parent[current]
                path.append(current)
            return path[::-1]
        for dr, dc in DIRECTIONS.values():
            nxt = current[0] + dr, current[1] + dc
            if not (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1]):
                continue
            if grid[nxt] == 1:
                continue
            new_cost = cost[current] + 1
            if new_cost < cost.get(nxt, 10**9):
                cost[nxt] = new_cost
                parent[nxt] = current
                h = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
                heapq.heappush(open_heap, (new_cost + h, next(serial), nxt))
    return []


class ReplanningAStarPlanner:
    name = "Replanning A*"

    def reset(self, static_grid, start: Cell, goal: Cell, start_dir: int = 1):
        self.static_grid = np.array(static_grid, dtype=int).copy()
        self.goal = goal
        self.shield_interventions = 0

    def act(self, state: State, observed_dynamic: Sequence[Cell]) -> str:
        grid = self.static_grid.copy()
        for cell in observed_dynamic:
            if cell not in (state[:2], self.goal):
                grid[cell] = 1
        path = _astar_cells(grid, state[:2], self.goal)
        return _route_action(state, path[1]) if len(path) > 1 else "turn_left"


class OnlineDStarPlanner:
    name = "Online D* Lite"

    def reset(self, static_grid, start: Cell, goal: Cell, start_dir: int = 1):
        self.static_grid = np.array(static_grid, dtype=int).copy()
        self.goal = goal
        self.planner = DStarLite(self.static_grid, start, goal)
        self.path = self.planner.plan()
        self.dynamic_cells = set()
        self.shield_interventions = 0

    def act(self, state: State, observed_dynamic: Sequence[Cell]) -> str:
        observed = set(observed_dynamic) - {state[:2], self.goal}
        for cell in self.dynamic_cells - observed:
            if self.static_grid[cell] != 1:
                self.planner.remove_obstacle(*cell)
        for cell in observed - self.dynamic_cells:
            self.planner.set_obstacle(*cell)
        self.dynamic_cells = observed
        self.planner.update_start(state[:2])
        self.path = self.planner.replan()
        return _route_action(state, self.path[1]) if len(self.path) > 1 else "turn_left"
