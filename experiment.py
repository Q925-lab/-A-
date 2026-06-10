# experiment.py
"""
系统实验：在动态障碍物环境下对比 A*、Q-table、DQN

实验设计：
- 测试集：10张未见地图（不同seed）
- 每个地图运行 A*（全局已知，动态重规划）、DQN（局部观测+训练策略）
- 记录：成功率、路径长度、转弯次数、危险区经过数、运行时间
"""

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    RESULTS_DIR, ACTIONS, DIRECTIONS,
    MAP_ROWS, MAP_COLS,
    OBSTACLE_COUNT, OBSTACLE_MIN_LEN, OBSTACLE_MAX_LEN,
    DANGER_RADIUS,
    NUM_MOVING_OBSTACLES, MOVING_OBSTACLE_PATTERNS,
    LOCAL_OBS_WINDOW,
)
from grid_env import (
    create_dynamic_env, create_demo_env,
    get_local_obs, get_obs_shape,
    DynamicGridEnv, GridEnv,
)
from astar import astar_search, summarize_path
from q_learning import QLearningAgent
from train import train_q_learning, extract_qlearning_path
from dqn_agent import DQNAgent
from dqn_train import extract_dqn_path
from evaluate import summarize_path_basic, write_comparison_csv


def run_astar_dynamic(env, max_steps_per_plan=20):
    """
    在动态环境中运行A*：每N步重新规划一次。
    模拟"A*在动态环境下需要频繁重规划"的场景。
    """
    state = env.reset()
    path_states = [state]
    total_cost = 0.0
    all_expanded = 0

    t0 = time.time()

    for step in range(500):
        # 每N步或在路径走完时重新规划
        if step % max_steps_per_plan == 0:
            # 用当前静态网格+当前位置规划
            from astar import astar_search as astar_impl
            # 创建一个临时静态环境
            temp_grid = env._static_grid.copy()
            # 标记当前位置为起点
            orig_start = temp_grid[env.start]
            orig_goal = temp_grid[env.goal]
            temp_grid[env.start] = 0  # FREE
            temp_grid[env.goal] = 4   # GOAL

            rr, cc, dd = state
            temp_grid[rr, cc] = 3  # 标记当前位置

            temp_env = GridEnv(
                grid=temp_grid,
                start=(rr, cc),
                goal=env.goal,
                start_dir=dd,
            )

            result = astar_impl(temp_env)
            all_expanded += result.get("expanded_nodes", 0)

            if result["success"] and len(result["path_states"]) > 1:
                local_path = result["path_states"]
                # 只取前面几步
                local_path = local_path[:max_steps_per_plan + 1]
            else:
                local_path = None

            path_idx = 0

        # 沿着规划的路径走一步
        if local_path and path_idx + 1 < len(local_path):
            target_state = local_path[path_idx + 1]
            # 执行动作达到target_state
            action = _action_to_reach(state, target_state, env)
            path_idx += 1
        else:
            # 无可用路径，随机尝试向目标方向移动
            action = _greedy_toward_goal(state, env)

        next_state, reward, done, info = env.step(action)
        path_states.append(next_state)
        state = next_state

        if done:
            break

    elapsed = time.time() - t0

    row, col, _ = state
    success = env.is_goal(row, col)
    path_metrics = summarize_path_basic(env, path_states)

    return {
        "success": success,
        "path_states": path_states,
        "path_length": path_metrics["path_length"],
        "turn_count": path_metrics["turn_count"],
        "danger_cells": path_metrics["danger_cells"],
        "running_time": elapsed,
        "expanded_nodes": all_expanded,
    }


def _action_to_reach(state, target_state, env):
    """计算从state到相邻target_state需要执行的动作"""
    r, c, d = state
    tr, tc, td = target_state

    if (r, c) == (tr, tc):
        # 只需要转向
        if d != td:
            diff = (td - d) % 4
            if diff == 1:
                return "turn_right"
            elif diff == 3:
                return "turn_left"
            elif diff == 2:
                return "turn_left"  # 两次左转等同于掉头
        return "forward"  # fallback

    dr, dc = DIRECTIONS[d]
    if (r + dr, c + dc) == (tr, tc):
        return "forward"
    if (r - dr, c - dc) == (tr, tc):
        return "backward"

    # 需要先转向
    for nd in range(4):
        ndr, ndc = DIRECTIONS[nd]
        if (r + ndr, c + ndc) == (tr, tc):
            diff = (nd - d) % 4
            if diff == 1:
                return "turn_right"
            elif diff == 3:
                return "turn_left"
    return "forward"


def _greedy_toward_goal(state, env):
    """贪心朝目标方向走"""
    r, c, d = state
    gr, gc = env.goal
    # 朝目标方向转向
    if r < gr and d != 2:
        return "turn_right" if (2 - d) % 4 == 1 else "turn_left"
    if r > gr and d != 0:
        return "turn_right" if (0 - d) % 4 == 1 else "turn_left"
    if c < gc and d != 1:
        return "turn_right" if (1 - d) % 4 == 1 else "turn_left"
    if c > gc and d != 3:
        return "turn_right" if (3 - d) % 4 == 1 else "turn_left"
    return "forward"


def run_q_table_static(env, episodes=5000, max_steps=500):
    """在静态地图上训练Q-table（原有方法）"""
    from config import Q_ALPHA, Q_GAMMA, Q_EPSILON_START, Q_EPSILON_MIN, Q_EPSILON_DECAY

    # 创建静态环境（无移动障碍物）
    static_env = GridEnv(
        grid=env._static_grid,
        start=env.start,
        goal=env.goal,
        start_dir=env.start_dir,
    )

    agent, history = train_q_learning(
        env=static_env,
        episodes=episodes,
        max_steps=max_steps,
        alpha=Q_ALPHA,
        gamma=Q_GAMMA,
        epsilon_start=Q_EPSILON_START,
        epsilon_min=Q_EPSILON_MIN,
        epsilon_decay=Q_EPSILON_DECAY,
    )

    path_result = extract_qlearning_path(
        env=static_env, agent=agent, max_steps=max_steps,
    )

    path_metrics = summarize_path_basic(static_env, path_result["path_states"])

    return {
        "success": path_result["success"],
        "path_length": path_metrics["path_length"],
        "turn_count": path_metrics["turn_count"],
        "danger_cells": path_metrics["danger_cells"],
        "collision": path_result["collision"],
        "total_reward": history["rewards"][-1] if history["rewards"] else 0,
        "final_success_rate": history["success_rates"][-1] if history["success_rates"] else 0,
    }


def run_experiment(test_seeds, dqn_model_path=None):
    """
    系统实验主函数。
    """
    results = []

    # 加载DQN模型
    dqn_agent = None
    if dqn_model_path and os.path.exists(dqn_model_path):
        dqn_agent = DQNAgent(
            input_dim=get_obs_shape(LOCAL_OBS_WINDOW),
            hidden_dim=128,
        )
        dqn_agent.load(dqn_model_path)
        dqn_agent.epsilon = 0.0
        print(f"已加载DQN模型：{dqn_model_path}")

    print(f"\n{'='*60}")
    print(f"系统实验：{len(test_seeds)} 张测试地图")
    print(f"{'='*60}")

    for si, seed in enumerate(test_seeds):
        print(f"\n--- Seed {seed} ({si+1}/{len(test_seeds)}) ---")

        # 创建动态环境
        dyn_env = create_dynamic_env(
            seed=seed,
            rows=MAP_ROWS, cols=MAP_COLS,
            obstacle_count=OBSTACLE_COUNT,
            danger_radius=DANGER_RADIUS,
            num_moving=NUM_MOVING_OBSTACLES,
            obstacle_patterns=MOVING_OBSTACLE_PATTERNS[:NUM_MOVING_OBSTACLES],
        )

        # 1. A*（静态全局）
        print("  运行 A*（静态全局）...")
        static_env = GridEnv(
            grid=dyn_env._static_grid,
            start=dyn_env.start,
            goal=dyn_env.goal,
            start_dir=dyn_env.start_dir,
        )
        astar_result = astar_search(static_env)
        if astar_result["success"]:
            metrics = summarize_path(
                static_env, astar_result["path_states"],
                astar_result["running_time"], astar_result["expanded_nodes"],
                astar_result["total_cost"],
            )
            results.append({
                "seed": seed, "algorithm": "A* (static)",
                "success": True,
                "path_length": metrics["path_length"],
                "turn_count": metrics["turn_count"],
                "danger_cells": metrics["danger_cells"],
                "running_time": metrics["running_time"],
                "total_cost": metrics["total_cost"],
                "expanded_nodes": metrics["expanded_nodes"],
            })
            print(f"    OK: path={metrics['path_length']}, cost={metrics['total_cost']:.1f}")
        else:
            results.append({
                "seed": seed, "algorithm": "A* (static)",
                "success": False,
                "path_length": 0, "turn_count": 0, "danger_cells": 0,
                "running_time": astar_result["running_time"],
                "total_cost": float("inf"), "expanded_nodes": astar_result["expanded_nodes"],
            })
            print("    FAILED")

        # 2. A*（动态重规划）
        print("  运行 A*（动态重规划）...")
        dyn_astar = run_astar_dynamic(dyn_env, max_steps_per_plan=10)
        results.append({
            "seed": seed, "algorithm": "A* (replan)",
            "success": dyn_astar["success"],
            "path_length": dyn_astar["path_length"],
            "turn_count": dyn_astar["turn_count"],
            "danger_cells": dyn_astar["danger_cells"],
            "running_time": dyn_astar["running_time"],
            "total_cost": "",
            "expanded_nodes": dyn_astar["expanded_nodes"],
        })
        print(f"    {'OK' if dyn_astar['success'] else 'FAILED'}: path={dyn_astar['path_length']}")

        # 3. Q-table（静态地图，原方法）
        print("  运行 Q-table（静态地图）...")
        q_result = run_q_table_static(dyn_env, episodes=Q_TABLE_EPISODES, max_steps=500)
        results.append({
            "seed": seed, "algorithm": "Q-table (static)",
            "success": q_result["success"],
            "path_length": q_result["path_length"],
            "turn_count": q_result["turn_count"],
            "danger_cells": q_result["danger_cells"],
            "running_time": 0,
            "total_cost": "",
            "expanded_nodes": "",
            "total_reward": q_result.get("total_reward", 0),
        })
        print(f"    {'OK' if q_result['success'] else 'FAILED'}: path={q_result['path_length']}")

        # 4. DQN（动态地图+局部观测）
        if dqn_agent:
            print("  运行 DQN（动态地图+局部观测）...")
            dqn_path = extract_dqn_path(dqn_agent, dyn_env, max_steps=500)
            dqn_metrics = summarize_path_basic(dyn_env, dqn_path["path_states"])
            results.append({
                "seed": seed, "algorithm": "DQN (dynamic)",
                "success": dqn_path["success"],
                "collision": dqn_path["collision"],
                "path_length": dqn_metrics["path_length"],
                "turn_count": dqn_metrics["turn_count"],
                "danger_cells": dqn_metrics["danger_cells"],
                "running_time": 0,
                "total_cost": "",
                "expanded_nodes": "",
            })
            print(f"    {'OK' if dqn_path['success'] else 'FAILED'}: path={dqn_metrics['path_length']}")
        else:
            print("  跳过 DQN（未提供模型）")

        # 打印对比
        print(f"  --- Seed {seed} 对比 ---")
        for r in results:
            if r["seed"] == seed:
                print(f"    {r['algorithm']:20s} | succ={r['success']} | "
                      f"len={r['path_length']:3d} | turns={r['turn_count']:2d} | "
                      f"danger={r['danger_cells']:2d}")

    return results


def plot_experiment_results(results, save_dir):
    """绘制实验结果对比图"""
    os.makedirs(save_dir, exist_ok=True)

    df = {r["algorithm"]: [] for r in results}
    seeds = sorted(set(r["seed"] for r in results))

    # 按算法聚合
    algo_data = {}
    for r in results:
        algo = r["algorithm"]
        if algo not in algo_data:
            algo_data[algo] = {"path_length": [], "success": [], "turn_count": [],
                               "danger_cells": [], "running_time": []}
        algo_data[algo]["path_length"].append(r["path_length"])
        algo_data[algo]["success"].append(1 if r["success"] else 0)
        algo_data[algo]["turn_count"].append(r["turn_count"])
        algo_data[algo]["danger_cells"].append(r["danger_cells"])
        if isinstance(r.get("running_time"), (int, float)):
            algo_data[algo]["running_time"].append(r["running_time"])

    algos = list(algo_data.keys())

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 成功率
    ax = axes[0, 0]
    x = np.arange(len(algos))
    rates = [np.mean(algo_data[a]["success"]) for a in algos]
    bars = ax.bar(x, rates, color=["#4CAF50", "#FF9800", "#2196F3", "#9C27B0"][:len(algos)])
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate by Algorithm")
    ax.set_ylim(0, 1.15)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{rate:.1%}", ha="center", fontsize=10)

    # 路径长度
    ax = axes[0, 1]
    means = [np.mean(algo_data[a]["path_length"]) for a in algos]
    stds = [np.std(algo_data[a]["path_length"]) for a in algos]
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=["#4CAF50", "#FF9800", "#2196F3", "#9C27B0"][:len(algos)])
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Path Length")
    ax.set_title("Avg Path Length (±std)")

    # 转弯次数
    ax = axes[0, 2]
    means = [np.mean(algo_data[a]["turn_count"]) for a in algos]
    bars = ax.bar(x, means, color=["#4CAF50", "#FF9800", "#2196F3", "#9C27B0"][:len(algos)])
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Turn Count")
    ax.set_title("Avg Turn Count")

    # 危险区经过数
    ax = axes[1, 0]
    means = [np.mean(algo_data[a]["danger_cells"]) for a in algos]
    bars = ax.bar(x, means, color=["#4CAF50", "#FF9800", "#2196F3", "#9C27B0"][:len(algos)])
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Danger Cells")
    ax.set_title("Avg Danger Cells Traversed")

    # 运行时间
    ax = axes[1, 1]
    if any(algo_data[a]["running_time"] for a in algos):
        means = [np.mean(algo_data[a]["running_time"]) * 1000 for a in algos]  # ms
        bars = ax.bar(x, means, color=["#4CAF50", "#FF9800", "#2196F3", "#9C27B0"][:len(algos)])
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Time (ms)")
        ax.set_title("Avg Running Time")
        ax.set_yscale("log")

    # Per-seed 对比
    ax = axes[1, 2]
    for i, algo in enumerate(algos):
        vals = algo_data[algo]["path_length"]
        ax.plot(seeds, vals, marker="o", label=algo, linewidth=2)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Path Length")
    ax.set_title("Path Length per Seed")
    ax.legend(fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "experiment_results.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"实验结果图已保存到：{save_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test-seeds", type=str, default="7,11,21,42,66,99,156,202,303,404")
    parser.add_argument("--dqn-model", type=str, default=None,
                        help="训练好的DQN模型路径")
    parser.add_argument("--q-episodes", type=int, default=3000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    global Q_TABLE_EPISODES
    Q_TABLE_EPISODES = args.q_episodes

    test_seeds = [int(s.strip()) for s in args.test_seeds.split(",")]

    out_dir = args.output if args.output else os.path.join(RESULTS_DIR, "experiment")
    os.makedirs(out_dir, exist_ok=True)

    # 查找DQN模型
    dqn_path = args.dqn_model
    if not dqn_path:
        default = os.path.join(RESULTS_DIR, "dqn_train", "dqn_model.pt")
        if os.path.exists(default):
            dqn_path = default

    results = run_experiment(test_seeds, dqn_model_path=dqn_path)

    # 保存CSV
    csv_path = os.path.join(out_dir, "comparison.csv")
    write_comparison_csv(results, csv_path)

    # 保存JSON
    json_path = os.path.join(out_dir, "experiment_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 绘图
    plot_experiment_results(results, out_dir)

    # 打印汇总
    print(f"\n{'='*60}")
    print("实验汇总")
    print(f"{'='*60}")
    algos = sorted(set(r["algorithm"] for r in results))
    for algo in algos:
        algo_results = [r for r in results if r["algorithm"] == algo]
        succ = sum(1 for r in algo_results if r["success"])
        avg_len = np.mean([r["path_length"] for r in algo_results if r["success"]])
        avg_danger = np.mean([r["danger_cells"] for r in algo_results])
        print(f"  {algo:20s}: success={succ}/{len(algo_results)} "
              f"avg_len={avg_len:.1f} avg_danger={avg_danger:.1f}")

    print(f"\n结果已保存到：{out_dir}")


if __name__ == "__main__":
    main()
