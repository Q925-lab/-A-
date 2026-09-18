"""Reproducible benchmark and ablation study for the PSH planner."""

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

from grid_env import create_dynamic_env
from psh_planner import OnlineDStarPlanner, PredictiveSafeHybridPlanner, ReplanningAStarPlanner


def observe(env, state, radius):
    r, c, _ = state
    return sorted(
        cell for cell in env.moving_obstacles
        if abs(cell[0] - r) <= radius and abs(cell[1] - c) <= radius
    )


def clearance(cell, obstacles):
    if not obstacles:
        return float("inf")
    return min(math.hypot(cell[0] - r, cell[1] - c) for r, c in obstacles)


def run_episode(env, planner, observation_radius=5, max_steps=350, trace=False):
    state = env.reset()
    planner.reset(env._static_grid, env.start, env.goal, env.start_dir)
    positions = [state[:2]]
    obstacle_trace = [list(env.moving_obstacles)]
    latencies = []
    turns = 0
    min_clearance = clearance(state[:2], env.moving_obstacles)
    collision = False

    for _ in range(max_steps):
        visible = observe(env, state, observation_radius)
        started = time.perf_counter()
        action = planner.act(state, visible)
        latencies.append((time.perf_counter() - started) * 1000.0)
        if action in ("turn_left", "turn_right"):
            turns += 1
        state, _, done, info = env.step(action)
        positions.append(state[:2])
        obstacle_trace.append(list(env.moving_obstacles))
        min_clearance = min(min_clearance, clearance(state[:2], env.moving_obstacles))
        collision = bool(info.get("collision", False))
        if done:
            break

    success = env.is_goal(state[0], state[1]) and not collision
    sorted_latency = sorted(latencies)
    p95_index = max(0, math.ceil(0.95 * len(sorted_latency)) - 1)
    result = {
        "success": success,
        "collision": collision,
        "steps": len(positions) - 1,
        "path_length": sum(a != b for a, b in zip(positions, positions[1:])),
        "turns": turns,
        "min_clearance": min_clearance,
        "latency_mean_ms": statistics.fmean(latencies) if latencies else 0.0,
        "latency_p95_ms": sorted_latency[p95_index] if sorted_latency else 0.0,
        "shield_interventions": getattr(planner, "shield_interventions", 0),
    }
    if trace:
        result["trace"] = {"robot": positions, "obstacles": obstacle_trace}
    return result


def planner_factories():
    return {
        "Replanning A*": ReplanningAStarPlanner,
        "Online D* Lite": OnlineDStarPlanner,
        "PSH - prediction": lambda: PredictiveSafeHybridPlanner(use_prediction=False),
        "PSH - shield": lambda: PredictiveSafeHybridPlanner(use_shield=False),
        "PSH (full)": PredictiveSafeHybridPlanner,
    }


def benchmark(seeds, output_dir, quick=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, cols = (18, 24) if quick else (24, 32)
    obstacle_count = 18 if quick else 34
    max_steps = 220 if quick else 350
    records = []

    for seed in seeds:
        for name, factory in planner_factories().items():
            env = create_dynamic_env(
                seed=seed, rows=rows, cols=cols, obstacle_count=obstacle_count,
                num_moving=4,
                obstacle_patterns=["horizontal", "vertical", "random", "horizontal"],
            )
            metrics = run_episode(env, factory(), max_steps=max_steps, trace=False)
            records.append({"seed": seed, "algorithm": name, **metrics})
            print(f"seed={seed:>3}  {name:<18} success={metrics['success']} "
                  f"collision={metrics['collision']} steps={metrics['steps']}")

    fields = list(records[0].keys())
    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    summary = []
    for name in planner_factories():
        rows_for_name = [r for r in records if r["algorithm"] == name]
        summary.append({
            "algorithm": name,
            "episodes": len(rows_for_name),
            "success_rate": statistics.fmean(r["success"] for r in rows_for_name),
            "collision_rate": statistics.fmean(r["collision"] for r in rows_for_name),
            "mean_steps": statistics.fmean(r["steps"] for r in rows_for_name),
            "mean_path_length": statistics.fmean(r["path_length"] for r in rows_for_name),
            "mean_turns": statistics.fmean(r["turns"] for r in rows_for_name),
            "mean_min_clearance": statistics.fmean(r["min_clearance"] for r in rows_for_name),
            "latency_mean_ms": statistics.fmean(r["latency_mean_ms"] for r in rows_for_name),
            "latency_p95_ms": statistics.fmean(r["latency_p95_ms"] for r in rows_for_name),
            "mean_shield_interventions": statistics.fmean(r["shield_interventions"] for r in rows_for_name),
        })
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return records, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20, 40)))
    parser.add_argument("--output", default="results/psh_benchmark")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    benchmark(args.seeds, args.output, args.quick)
