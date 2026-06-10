# dqn_train.py
"""
DQN训练脚本：在动态障碍物环境中训练智能体。
支持断点续训（checkpoint）。

训练策略：
- 预缓存训练环境，避免每轮重新生成地图
- 定期保存checkpoint，中断后可恢复
- 在验证集上评估，保留最佳模型
"""

import os
import json
import pickle
import time
import numpy as np
import torch
from tqdm import tqdm

from config import (
    RESULTS_DIR, ACTIONS,
    MAP_ROWS, MAP_COLS,
    OBSTACLE_COUNT, OBSTACLE_MIN_LEN, OBSTACLE_MAX_LEN,
    DANGER_RADIUS,
    NUM_MOVING_OBSTACLES, MOVING_OBSTACLE_PATTERNS,
    LOCAL_OBS_WINDOW,
    DEFAULT_SEED,
)
from grid_env import create_dynamic_env, get_local_obs, get_obs_shape
from dqn_agent import DQNAgent
from visualize import plot_training_curves


def _cache_envs(seeds, map_rows, map_cols, obstacle_count, num_moving):
    """预创建环境，避免每轮重复生成。返回到表 [(seed, env), ...]"""
    envs = []
    for seed in seeds:
        env = create_dynamic_env(
            seed=seed, rows=map_rows, cols=map_cols,
            obstacle_count=obstacle_count,
            danger_radius=DANGER_RADIUS,
            num_moving=num_moving,
            obstacle_patterns=MOVING_OBSTACLE_PATTERNS[:num_moving],
        )
        envs.append(env)
    return envs


def train_dqn(
    train_seeds=None,
    val_seeds=None,
    episodes=5000,
    max_steps=500,
    # DQN超参
    hidden_dim=128,
    lr=5e-4,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.999,
    batch_size=64,
    buffer_capacity=50000,
    target_update_freq=100,
    tau=0.005,
    # 环境
    map_rows=15,
    map_cols=20,
    obstacle_count=20,
    num_moving=1,
    # 保存
    checkpoint_dir=None,
    save_best_only=True,
    eval_freq=500,
    update_freq=1,
    device=None,
):
    """训练DQN智能体，支持断点续训。"""
    if train_seeds is None:
        train_seeds = list(range(100, 120))  # 20 maps
    if val_seeds is None:
        val_seeds = [7, 11, 21, 42, 66]

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(RESULTS_DIR, "dqn_train")

    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, "dqn_model.pt")
    history_path = os.path.join(checkpoint_dir, "history.json")
    ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pkl")

    obs_dim = get_obs_shape(LOCAL_OBS_WINDOW)
    print(f"观测维度: {obs_dim}")
    print(f"训练地图: {len(train_seeds)}张, 验证地图: {len(val_seeds)}张")
    print(f"训练轮数: {episodes}, 每轮最大步数: {max_steps}")
    print(f"地图大小: {map_rows}x{map_cols}, 移动障碍物: {num_moving}")
    print(f"检查点目录: {checkpoint_dir}")

    # ---- 尝试恢复 ----
    start_episode = 1
    agent = None

    if os.path.exists(ckpt_path):
        print("发现检查点，尝试恢复...")
        try:
            with open(ckpt_path, "rb") as f:
                ckpt = pickle.load(f)
            agent = DQNAgent(
                input_dim=obs_dim, hidden_dim=hidden_dim, lr=lr,
                gamma=gamma, epsilon_start=epsilon_start,
                epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
                batch_size=batch_size, buffer_capacity=buffer_capacity,
                target_update_freq=target_update_freq, tau=tau, device=device,
            )
            agent.load(model_path)
            agent.epsilon = ckpt["epsilon"]
            agent.optimizer.load_state_dict(ckpt["optimizer_state"])
            agent.memory = ckpt["memory"]
            history = ckpt["history"]
            start_episode = ckpt["episode"] + 1
            best_eval_rate = ckpt.get("best_eval_rate", 0.0)
            recent_success = ckpt.get("recent_success", history["success"][-100:])
            print(f"从第 {start_episode} 轮恢复, epsilon={agent.epsilon:.3f}, "
                  f"best_eval={best_eval_rate:.1%}")
        except Exception as e:
            print(f"恢复失败 ({e})，从头开始训练")
            agent = None
            start_episode = 1

    if agent is None:
        agent = DQNAgent(
            input_dim=obs_dim, hidden_dim=hidden_dim, lr=lr,
            gamma=gamma, epsilon_start=epsilon_start,
            epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
            batch_size=batch_size, buffer_capacity=buffer_capacity,
            target_update_freq=target_update_freq, tau=tau, device=device,
        )
        history = {
            "rewards": [], "success": [], "collisions": [],
            "timeouts": [], "steps": [],
            "eval_success_rates": [], "eval_episodes": [], "epsilon": [],
        }
        recent_success = []
        best_eval_rate = 0.0

    # ---- 预缓存环境 ----
    print("预创建训练环境...")
    train_envs = _cache_envs(train_seeds, map_rows, map_cols, obstacle_count, num_moving)
    print(f"完成, {len(train_envs)} 个环境")

    # ---- 训练循环 ----
    pbar = tqdm(range(start_episode, episodes + 1), desc="DQN",
                initial=start_episode - 1, total=episodes)
    for episode in pbar:
        env = train_envs[np.random.randint(0, len(train_envs))]

        state = env.reset()
        obs = get_local_obs(env, state, LOCAL_OBS_WINDOW)
        total_reward = 0.0
        success = 0
        collision = 0
        timeout_flag = 0
        steps = 0

        for step in range(max_steps):
            action = agent.choose_action(obs, training=True)
            next_state, reward, done, info = env.step(action)
            next_obs = get_local_obs(env, next_state, LOCAL_OBS_WINDOW)

            if step == max_steps - 1 and not done:
                reward -= 100.0
                done = True
                timeout_flag = 1

            agent.store_experience(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
            steps += 1

            if update_freq > 0 and len(agent.memory) >= batch_size:
                agent.update()

            if info.get("collision", False):
                collision = 1

            if done:
                if info.get("goal", False) or env.is_goal(next_state[0], next_state[1]):
                    success = 1
                break

        agent.decay_epsilon()

        history["rewards"].append(total_reward)
        history["success"].append(success)
        history["collisions"].append(collision)
        history["timeouts"].append(timeout_flag)
        history["steps"].append(steps)
        history["epsilon"].append(agent.epsilon)

        recent_success.append(success)
        if len(recent_success) > 100:
            recent_success.pop(0)

        # ---- 评估 ----
        if episode % eval_freq == 0 or episode == 1:
            eval_rate, eval_info = evaluate_dqn(
                agent, val_seeds, max_steps, map_rows, map_cols,
                obstacle_count, num_moving)
            history["eval_success_rates"].append(eval_rate)
            history["eval_episodes"].append(episode)

            if eval_rate > best_eval_rate:
                best_eval_rate = eval_rate
                agent.save(model_path)
                tqdm.write(f"  >> 新最佳! 验证成功率={eval_rate:.1%} (ep {episode})")

            # 总是保存检查点（用于恢复）
            with open(ckpt_path, "wb") as f:
                pickle.dump({
                    "episode": episode,
                    "epsilon": agent.epsilon,
                    "optimizer_state": agent.optimizer.state_dict(),
                    "memory": agent.memory,
                    "history": history,
                    "best_eval_rate": best_eval_rate,
                    "recent_success": recent_success,
                }, f)
            # 保存历史
            try:
                with open(history_path, "w") as f:
                    json.dump(history, f)
            except Exception:
                pass

        avg_reward = np.mean(history["rewards"][-100:])
        succ_rate = np.mean(recent_success) if recent_success else 0.0
        pbar.set_postfix({
            "r": f"{avg_reward:.0f}",
            "succ": f"{succ_rate:.1%}",
            "eps": f"{agent.epsilon:.2f}",
        })

    # ---- 最终保存 ----
    agent.save(model_path)
    with open(ckpt_path, "wb") as f:
        pickle.dump({
            "episode": episodes, "epsilon": agent.epsilon,
            "optimizer_state": agent.optimizer.state_dict(),
            "memory": agent.memory, "history": history,
            "best_eval_rate": best_eval_rate,
            "recent_success": list(recent_success),
        }, f)
    with open(history_path, "w") as f:
        json.dump(history, f)

    print(f"\n训练完成: reward={avg_reward:.0f}, succ={succ_rate:.1%}, "
          f"best_eval={best_eval_rate:.1%}")
    return agent, history


def evaluate_dqn(agent, val_seeds, max_steps, map_rows, map_cols, obstacle_count, num_moving):
    """在验证集上评估DQN智能体。"""
    success_count = 0
    total = len(val_seeds)
    details = []

    for seed in val_seeds:
        env = create_dynamic_env(
            seed=seed, rows=map_rows, cols=map_cols,
            obstacle_count=obstacle_count,
            danger_radius=DANGER_RADIUS,
            num_moving=num_moving,
            obstacle_patterns=MOVING_OBSTACLE_PATTERNS[:num_moving],
        )
        state = env.reset()
        obs = get_local_obs(env, state, LOCAL_OBS_WINDOW)

        for step in range(max_steps):
            action = agent.choose_action(obs, training=False)
            next_state, reward, done, info = env.step(action)
            obs = get_local_obs(env, next_state, LOCAL_OBS_WINDOW)
            if done:
                row, col, _ = next_state
                if env.is_goal(row, col):
                    success_count += 1
                    details.append({"seed": seed, "success": True, "steps": step + 1})
                else:
                    details.append({"seed": seed, "success": False, "reason": "collision"})
                break
        else:
            details.append({"seed": seed, "success": False, "reason": "timeout"})

    return success_count / total, details


def extract_dqn_path(agent, env, max_steps=500):
    """使用训练好的DQN提取一条路径。"""
    state = env.reset()
    obs = get_local_obs(env, state, LOCAL_OBS_WINDOW)
    path_states = [state]
    visited = set()
    visited.add((state[0], state[1]))

    for _ in range(max_steps):
        action = agent.choose_action(obs, training=False)
        next_state, reward, done, info = env.step(action)
        path_states.append(next_state)

        if info.get("collision", False):
            return {"success": False, "collision": True, "path_states": path_states}

        row, col, _ = next_state
        if env.is_goal(row, col):
            return {"success": True, "collision": False, "path_states": path_states}

        obs = get_local_obs(env, next_state, LOCAL_OBS_WINDOW)
        pos = (row, col)
        if pos in visited:
            return {"success": False, "collision": False, "path_states": path_states}
        visited.add(pos)

    return {"success": False, "collision": False, "path_states": path_states}


def main():
    """命令行入口"""
    import argparse
    parser = argparse.ArgumentParser(description="DQN 训练脚本")
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--map-rows", type=int, default=12)
    parser.add_argument("--map-cols", type=int, default=16)
    parser.add_argument("--obstacle-count", type=int, default=12)
    parser.add_argument("--num-moving", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    parser.add_argument("--eval-freq", type=int, default=500)
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "dqn_train"))
    args = parser.parse_args()

    train_seeds = list(range(100, 120))  # 20 maps
    val_seeds = [7, 11, 21, 42, 66]

    agent, history = train_dqn(
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        episodes=args.episodes,
        max_steps=args.max_steps,
        map_rows=args.map_rows,
        map_cols=args.map_cols,
        obstacle_count=args.obstacle_count,
        num_moving=args.num_moving,
        lr=args.lr,
        epsilon_decay=args.epsilon_decay,
        eval_freq=args.eval_freq,
        checkpoint_dir=args.output,
        hidden_dim=128,
    )

    # 绘制训练曲线
    from visualize import moving_average
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = args.output
    rewards = history["rewards"]
    smoothed = moving_average(rewards, window=100)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(rewards, alpha=0.3)
    x = np.arange(len(smoothed)) + 100
    ax.plot(x, smoothed, linewidth=2)
    ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
    ax.set_title("DQN Reward Curve"); ax.grid(True)

    ax = axes[0, 1]
    success_rates = [np.mean(history["success"][max(0, i-99):i+1])
                     for i in range(len(history["success"]))]
    ax.plot(success_rates)
    ax.set_xlabel("Episode"); ax.set_ylabel("Success Rate (100-eps)")
    ax.set_title("DQN Success Rate"); ax.set_ylim(0, 1.05); ax.grid(True)

    ax = axes[1, 0]
    eval_eps = history["eval_episodes"]
    eval_rates = history["eval_success_rates"]
    ax.plot(eval_eps, eval_rates, "o-", markersize=6)
    ax.set_xlabel("Episode"); ax.set_ylabel("Eval Success Rate")
    ax.set_title("Evaluation on Validation Set"); ax.set_ylim(0, 1.05); ax.grid(True)

    ax = axes[1, 1]
    ax.plot(history["epsilon"])
    ax.set_xlabel("Episode"); ax.set_ylabel("Epsilon")
    ax.set_title("Epsilon Decay"); ax.grid(True)

    plt.tight_layout()
    for name in ["dqn_reward", "dqn_success_rate", "dqn_eval", "dqn_epsilon"]:
        pass
    fig.savefig(os.path.join(out, "dqn_training_curves.png"), dpi=150)
    print(f"训练曲线已保存到 {os.path.join(out, 'dqn_training_curves.png')}")

    plt.close("all")

    # 单独绘制奖励曲线
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, alpha=0.3, color="blue")
    if len(smoothed) > 0:
        ax.plot(np.arange(len(smoothed)) + 100, smoothed, linewidth=2, color="blue")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
    ax.set_title("DQN Reward Curve"); ax.grid(True)
    fig.savefig(os.path.join(out, "dqn_reward.png"), dpi=150)
    plt.close(fig)

    # 单独绘制成功率
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(success_rates)
    ax.set_xlabel("Episode"); ax.set_ylabel("Success Rate (last 100)")
    ax.set_title("DQN Success Rate"); ax.set_ylim(0, 1.05); ax.grid(True)
    fig.savefig(os.path.join(out, "dqn_success_rate.png"), dpi=150)
    plt.close(fig)

    # 评估曲线
    if eval_eps:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(eval_eps, eval_rates, "o-", markersize=6)
        ax.set_xlabel("Episode"); ax.set_ylabel("Eval Success Rate")
        ax.set_title("Evaluation on Validation Set"); ax.set_ylim(0, 1.05); ax.grid(True)
        fig.savefig(os.path.join(out, "dqn_eval.png"), dpi=150)
        plt.close(fig)

    # Epsilon曲线
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["epsilon"])
    ax.set_xlabel("Episode"); ax.set_ylabel("Epsilon")
    ax.set_title("Epsilon Decay"); ax.grid(True)
    fig.savefig(os.path.join(out, "dqn_epsilon.png"), dpi=150)
    plt.close(fig)

    print("全部完成!")


if __name__ == "__main__":
    main()
