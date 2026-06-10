# visualize.py

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 批量实验时不弹窗，避免 plt.show() 阻塞程序
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def compress_positions(path_states):
    """
    将状态路径转成位置路径，并去掉连续重复的位置。
    因为机器人原地转向时，位置不会变。
    """
    positions = []

    for state in path_states:
        row, col, _ = state
        pos = (row, col)

        if not positions or positions[-1] != pos:
            positions.append(pos)

    return positions


def moving_average(values, window=100):
    """
    简单滑动平均，用于生成更适合报告/PPT展示的平滑奖励曲线。
    """
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_grid_path(env, path_states=None, title="Path Planning Result", save_path=None):
    """
    绘制地图和路径。
    """
    grid = env.grid.copy()

    cmap = ListedColormap([
        "white",       # 0 free
        "black",       # 1 obstacle
        "orange",      # 2 danger
        "green",       # 3 start
        "red",         # 4 goal
    ])

    fig = plt.figure(figsize=(14, 10))
    plt.imshow(grid, cmap=cmap, vmin=0, vmax=4)

    # 绘制网格线
    plt.xticks(np.arange(-0.5, env.width, 1), [])
    plt.yticks(np.arange(-0.5, env.height, 1), [])
    plt.grid(color="gray", linestyle="-", linewidth=0.5)

    # 标注起点终点
    start_row, start_col = env.start
    goal_row, goal_col = env.goal

    plt.text(start_col, start_row, "S", ha="center", va="center",
             fontsize=14, color="white", weight="bold")
    plt.text(goal_col, goal_row, "G", ha="center", va="center",
             fontsize=14, color="white", weight="bold")

    # 绘制路径
    if path_states:
        positions = compress_positions(path_states)
        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        plt.plot(cols, rows, marker="o", linewidth=2, markersize=4)

    plt.title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"路径图已保存到：{save_path}")

    plt.close(fig)


def plot_training_curves(history, reward_save_path=None, success_save_path=None, smooth_reward_save_path=None):
    """
    绘制 Q-learning 训练过程中的累计奖励曲线和成功率曲线。
    """
    rewards = history["rewards"]
    success_rates = history["success_rates"]

    # 奖励曲线
    fig = plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-learning Reward Curve")
    plt.grid(True)

    if reward_save_path:
        os.makedirs(os.path.dirname(reward_save_path), exist_ok=True)
        plt.savefig(reward_save_path, dpi=200)
        print(f"奖励曲线已保存到：{reward_save_path}")

    plt.close(fig)

    # 平滑奖励曲线，更适合报告展示
    if smooth_reward_save_path:
        smoothed = moving_average(rewards, window=100)
        fig = plt.figure(figsize=(10, 5))
        x = np.arange(len(smoothed)) + 100
        plt.plot(x, smoothed)
        plt.xlabel("Episode")
        plt.ylabel("Smoothed Total Reward")
        plt.title("Q-learning Smoothed Reward Curve (Window=100)")
        plt.grid(True)
        os.makedirs(os.path.dirname(smooth_reward_save_path), exist_ok=True)
        plt.savefig(smooth_reward_save_path, dpi=200)
        print(f"平滑奖励曲线已保存到：{smooth_reward_save_path}")
        plt.close(fig)

    # 成功率曲线
    fig = plt.figure(figsize=(10, 5))
    plt.plot(success_rates)
    plt.xlabel("Episode")
    plt.ylabel("Success Rate in Recent 100 Episodes")
    plt.title("Q-learning Success Rate Curve")
    plt.ylim(0, 1.05)
    plt.grid(True)

    if success_save_path:
        os.makedirs(os.path.dirname(success_save_path), exist_ok=True)
        plt.savefig(success_save_path, dpi=200)
        print(f"成功率曲线已保存到：{success_save_path}")

    plt.close(fig)
