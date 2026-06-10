# astar.py

import heapq
import time
from itertools import count

from config import (
    ACTIONS,
    MOVE_COST,
    BACKWARD_COST,
    TURN_COST,
    DANGER_COST,
    EDGE_COST,
    EDGE_MARGIN,
)


def manhattan_distance(pos, goal):
    """
    曼哈顿距离启发函数。
    pos 和 goal 都是 (row, col)
    """
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def get_action_cost(env, action, next_state, info):
    """
    A* 中的动作代价：
    - 前进：正常移动代价
    - 后退：略高代价
    - 左转/右转：转向代价
    - 危险区域：额外安全距离代价
    - 靠近地图边界：额外边界惩罚，避免贴边绕行
    """
    if action == "forward":
        cost = MOVE_COST
    elif action == "backward":
        cost = BACKWARD_COST
    elif action in ["turn_left", "turn_right"]:
        cost = TURN_COST
    else:
        cost = MOVE_COST

    row, col, _ = next_state

    # 障碍物附近危险区域代价
    if env.is_danger(row, col):
        cost += DANGER_COST

    # 边界软约束：不禁止靠边，但靠边更贵
    distance_to_edge = min(
        row,
        col,
        env.height - 1 - row,
        env.width - 1 - col,
    )

    if distance_to_edge <= EDGE_MARGIN:
        cost += EDGE_COST

    return cost


def reconstruct_path(came_from, end_state):
    """
    从终点状态反向回溯路径。
    """
    path = [end_state]
    current = end_state

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()
    return path


def astar_search(env):
    """
    在带有朝向的状态空间中执行 A* 搜索。

    返回：
    {
        "success": bool,
        "path_states": list,
        "expanded_nodes": int,
        "running_time": float,
        "total_cost": float
    }
    """
    start_time = time.time()

    start_state = env.start_state
    goal = env.goal

    open_heap = []
    tie_breaker = count()

    g_score = {start_state: 0.0}
    came_from = {}

    start_h = manhattan_distance((start_state[0], start_state[1]), goal)
    heapq.heappush(open_heap, (start_h, next(tie_breaker), start_state))

    closed_set = set()
    expanded_nodes = 0

    while open_heap:
        _, _, current_state = heapq.heappop(open_heap)

        if current_state in closed_set:
            continue

        closed_set.add(current_state)
        expanded_nodes += 1

        row, col, direction = current_state

        if (row, col) == goal:
            running_time = time.time() - start_time
            path_states = reconstruct_path(came_from, current_state)

            return {
                "success": True,
                "path_states": path_states,
                "expanded_nodes": expanded_nodes,
                "running_time": running_time,
                "total_cost": g_score[current_state],
            }

        for action in ACTIONS:
            next_state, valid, info = env.get_next_state(current_state, action)

            if not valid:
                continue

            step_cost = get_action_cost(env, action, next_state, info)
            tentative_g = g_score[current_state] + step_cost

            if tentative_g < g_score.get(next_state, float("inf")):
                came_from[next_state] = current_state
                g_score[next_state] = tentative_g

                next_row, next_col, _ = next_state
                h = manhattan_distance((next_row, next_col), goal)
                f = tentative_g + h

                heapq.heappush(open_heap, (f, next(tie_breaker), next_state))

    running_time = time.time() - start_time

    return {
        "success": False,
        "path_states": [],
        "expanded_nodes": expanded_nodes,
        "running_time": running_time,
        "total_cost": float("inf"),
    }


def summarize_path(env, path_states, running_time, expanded_nodes, total_cost):
    """
    统计路径指标。
    """
    if not path_states:
        return {
            "path_length": 0,
            "turn_count": 0,
            "danger_cells": 0,
            "running_time": running_time,
            "expanded_nodes": expanded_nodes,
            "total_cost": total_cost,
        }

    path_length = 0
    turn_count = 0
    danger_positions = set()

    for i in range(1, len(path_states)):
        prev = path_states[i - 1]
        curr = path_states[i]

        prev_pos = (prev[0], prev[1])
        curr_pos = (curr[0], curr[1])

        if prev_pos != curr_pos:
            path_length += 1

        if prev[2] != curr[2]:
            turn_count += 1

        if env.is_danger(curr[0], curr[1]):
            danger_positions.add(curr_pos)

    return {
        "path_length": path_length,
        "turn_count": turn_count,
        "danger_cells": len(danger_positions),
        "running_time": running_time,
        "expanded_nodes": expanded_nodes,
        "total_cost": total_cost,
    }