# evaluate.py

import csv
import os
from typing import Dict, List, Optional


def summarize_path_basic(env, path_states):
    """
    对任意状态路径进行基础统计。
    适用于 A* 和 Q-learning。
    """
    if not path_states:
        return {
            "path_length": 0,
            "turn_count": 0,
            "danger_cells": 0,
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
    }


def write_comparison_csv(rows: List[Dict], save_path: str):
    """
    写出算法对比表。不同算法缺失的字段用空值保存。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fieldnames = [
        "seed",
        "algorithm",
        "success",
        "path_length",
        "turn_count",
        "danger_cells",
        "collision",
        "running_time",
        "expanded_nodes",
        "total_cost",
        "total_reward",
        "final_success_rate",
        "avg_reward_last_100",
        "episodes",
        "max_steps",
    ]

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"对比表已保存到：{save_path}")


def append_summary_csv(rows: List[Dict], save_path: str):
    """
    写出跨地图总表。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fieldnames = [
        "seed",
        "algorithm",
        "success",
        "path_length",
        "turn_count",
        "danger_cells",
        "collision",
        "running_time",
        "expanded_nodes",
        "total_cost",
        "total_reward",
        "final_success_rate",
        "avg_reward_last_100",
        "episodes",
        "max_steps",
    ]

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"跨地图总表已保存到：{save_path}")
