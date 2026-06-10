# main.py

import json
import os
import time

from config import (
    RESULTS_DIR,
    EXPERIMENT_SEEDS,
    MAP_ROWS,
    MAP_COLS,
    OBSTACLE_COUNT,
    OBSTACLE_MIN_LEN,
    OBSTACLE_MAX_LEN,
    DANGER_RADIUS,
    Q_EPISODES,
    Q_MAX_STEPS,
    Q_ALPHA,
    Q_GAMMA,
    Q_EPSILON_START,
    Q_EPSILON_MIN,
    Q_EPSILON_DECAY,
)
from grid_env import create_demo_env
from astar import astar_search, summarize_path
from train import train_q_learning, extract_qlearning_path
from visualize import plot_grid_path, plot_training_curves
from evaluate import summarize_path_basic, write_comparison_csv, append_summary_csv


def make_seed_dir(seed):
    """
    为每个地图种子创建单独结果目录：results/seed_xx/
    """
    out_dir = os.path.join(RESULTS_DIR, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_map_info(seed, out_dir):
    """
    保存当前地图参数，保证实验可复现。
    """
    info = {
        "seed": seed,
        "rows": MAP_ROWS,
        "cols": MAP_COLS,
        "obstacle_count": OBSTACLE_COUNT,
        "obstacle_min_len": OBSTACLE_MIN_LEN,
        "obstacle_max_len": OBSTACLE_MAX_LEN,
        "danger_radius": DANGER_RADIUS,
        "q_episodes": Q_EPISODES,
        "q_max_steps": Q_MAX_STEPS,
        "q_alpha": Q_ALPHA,
        "q_gamma": Q_GAMMA,
        "q_epsilon_start": Q_EPSILON_START,
        "q_epsilon_min": Q_EPSILON_MIN,
        "q_epsilon_decay": Q_EPSILON_DECAY,
    }
    save_path = os.path.join(out_dir, "experiment_config.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"实验配置已保存到：{save_path}")


def run_astar(env, seed, out_dir):
    """
    运行 A* 路径规划，并把结果保存到当前 seed 目录。
    """
    print("\n==============================")
    print(f"开始运行 A* 路径规划 | seed={seed}")
    print("==============================")

    result = astar_search(env)

    if result["success"]:
        print("A* 成功找到路径！")

        metrics = summarize_path(
            env=env,
            path_states=result["path_states"],
            running_time=result["running_time"],
            expanded_nodes=result["expanded_nodes"],
            total_cost=result["total_cost"],
        )

        print("\n===== A* 路径规划结果 =====")
        print(f"路径长度：{metrics['path_length']}")
        print(f"转弯次数：{metrics['turn_count']}")
        print(f"经过危险区域格子数：{metrics['danger_cells']}")
        print(f"搜索节点数量：{metrics['expanded_nodes']}")
        print(f"总代价：{metrics['total_cost']:.2f}")
        print(f"运行时间：{metrics['running_time']:.6f} 秒")

        save_path = os.path.join(out_dir, "astar_path.png")
        plot_grid_path(
            env=env,
            path_states=result["path_states"],
            title=f"A* Path Planning | Seed {seed}",
            save_path=save_path,
        )

        row = {
            "seed": seed,
            "algorithm": "A*",
            "success": True,
            "path_length": metrics["path_length"],
            "turn_count": metrics["turn_count"],
            "danger_cells": metrics["danger_cells"],
            "collision": False,
            "running_time": metrics["running_time"],
            "expanded_nodes": metrics["expanded_nodes"],
            "total_cost": metrics["total_cost"],
            "episodes": "",
            "max_steps": "",
        }
        return row

    print("A* 没有找到可行路径。")

    save_path = os.path.join(out_dir, "astar_failed.png")
    plot_grid_path(
        env=env,
        path_states=None,
        title=f"A* Failed | Seed {seed}",
        save_path=save_path,
    )

    return {
        "seed": seed,
        "algorithm": "A*",
        "success": False,
        "collision": False,
        "path_length": 0,
        "turn_count": 0,
        "danger_cells": 0,
        "running_time": result["running_time"],
        "expanded_nodes": result["expanded_nodes"],
        "total_cost": result["total_cost"],
        "episodes": "",
        "max_steps": "",
    }


def run_q_learning(env, seed, out_dir):
    """
    运行 Q-learning 训练与路径提取，并把结果保存到当前 seed 目录。
    """
    print("\n==============================")
    print(f"开始训练 Q-learning | seed={seed}")
    print("==============================")

    q_table_path = os.path.join(out_dir, "q_table.npy")

    train_start = time.time()
    agent, history = train_q_learning(
        env=env,
        episodes=Q_EPISODES,
        max_steps=Q_MAX_STEPS,
        alpha=Q_ALPHA,
        gamma=Q_GAMMA,
        epsilon_start=Q_EPSILON_START,
        epsilon_min=Q_EPSILON_MIN,
        epsilon_decay=Q_EPSILON_DECAY,
        save_path=q_table_path,
    )
    training_time = time.time() - train_start

    reward_curve_path = os.path.join(out_dir, "reward_curve.png")
    smooth_reward_curve_path = os.path.join(out_dir, "reward_curve_smooth.png")
    success_curve_path = os.path.join(out_dir, "success_rate.png")

    plot_training_curves(
        history=history,
        reward_save_path=reward_curve_path,
        smooth_reward_save_path=smooth_reward_curve_path,
        success_save_path=success_curve_path,
    )

    print("\n开始提取 Q-learning 训练后路径...")
    extract_start = time.time()
    path_result = extract_qlearning_path(
        env=env,
        agent=agent,
        max_steps=Q_MAX_STEPS,
    )
    extract_time = time.time() - extract_start

    if path_result["success"]:
        print("Q-learning 成功到达终点！")
    elif path_result["collision"]:
        print("Q-learning 路径发生碰撞，说明训练还不充分。")
    else:
        print("Q-learning 未能到达终点，可能陷入循环或训练不足。")

    q_path_save_path = os.path.join(out_dir, "qlearning_path.png")
    plot_grid_path(
        env=env,
        path_states=path_result["path_states"],
        title=f"Q-learning Learned Path | Seed {seed}",
        save_path=q_path_save_path,
    )

    final_success_rate = history["success_rates"][-1]
    avg_reward_last_100 = sum(history["rewards"][-100:]) / min(100, len(history["rewards"]))
    final_total_reward = history["rewards"][-1]

    path_metrics = summarize_path_basic(env, path_result["path_states"])

    print("\n===== Q-learning 训练结果 =====")
    print(f"训练总时间：{training_time:.6f} 秒")
    print(f"最终近 100 轮成功率：{final_success_rate:.2%}")
    print(f"最后 100 轮平均累计奖励：{avg_reward_last_100:.2f}")
    print(f"最终路径是否成功：{path_result['success']}")

    row = {
        "seed": seed,
        "algorithm": "Q-learning",
        "success": path_result["success"],
        "path_length": path_metrics["path_length"],
        "turn_count": path_metrics["turn_count"],
        "danger_cells": path_metrics["danger_cells"],
        "collision": path_result["collision"],
        "running_time": training_time + extract_time,
        "expanded_nodes": "",
        "total_cost": "",
        "total_reward": final_total_reward,
        "final_success_rate": final_success_rate,
        "avg_reward_last_100": avg_reward_last_100,
        "episodes": Q_EPISODES,
        "max_steps": Q_MAX_STEPS,
    }
    return row


def run_single_seed(seed):
    """
    对单个随机地图种子运行 A* 与 Q-learning，并输出该种子的 comparison.csv。
    """
    out_dir = make_seed_dir(seed)
    save_map_info(seed, out_dir)

    env = create_demo_env(
        seed=seed,
        rows=MAP_ROWS,
        cols=MAP_COLS,
        obstacle_count=OBSTACLE_COUNT,
        min_len=OBSTACLE_MIN_LEN,
        max_len=OBSTACLE_MAX_LEN,
        danger_radius=DANGER_RADIUS,
    )

    # 第一部分：A* 基准算法
    astar_row = run_astar(env, seed, out_dir)

    # 第二部分：Q-learning 强化学习
    q_row = run_q_learning(env, seed, out_dir)

    rows = [astar_row, q_row]
    write_comparison_csv(rows, os.path.join(out_dir, "comparison.csv"))

    print("\n==============================")
    print(f"seed={seed} 当前地图实验完成")
    print("==============================")
    print(f"结果文件已保存到：{out_dir}")
    print("1. astar_path.png")
    print("2. reward_curve.png")
    print("3. reward_curve_smooth.png")
    print("4. success_rate.png")
    print("5. qlearning_path.png")
    print("6. q_table.npy")
    print("7. comparison.csv")

    return rows


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_rows = []
    for seed in EXPERIMENT_SEEDS:
        rows = run_single_seed(seed)
        all_rows.extend(rows)

    append_summary_csv(all_rows, os.path.join(RESULTS_DIR, "all_seeds_comparison.csv"))

    print("\n==============================")
    print("全部地图种子实验运行完成")
    print("==============================")
    print(f"总表已保存到：{os.path.join(RESULTS_DIR, 'all_seeds_comparison.csv')}")


if __name__ == "__main__":
    main()
