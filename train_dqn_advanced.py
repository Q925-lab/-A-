# train_dqn_advanced.py
"""
终极DQN训练脚本：针对交互式Demo场景优化

核心改进:
  1. 动态障碍物注入: 训练中随机添加/移除障碍，模拟玩家行为
  2. 课程学习: 前30%轮次无障碍，30-60%轻度注入，60-100%重度注入
  3. 大地图: 18×24，长路径
  4. 大量训练图: 50张训练 + 10张验证
  5. 更大网络: 256 hidden units
  6. 更长训练: 10000 episodes
"""

import os, json, pickle, random, time
import numpy as np
import torch
from tqdm import tqdm

from config import (
    FREE, OBSTACLE, DANGER, START, GOAL, DANGER_RADIUS,
    MOVING_OBSTACLE_PATTERNS, LOCAL_OBS_WINDOW, RESULTS_DIR, ACTIONS,
)
from grid_env import create_dynamic_env, get_local_obs, get_obs_shape
from dqn_agent import DQNAgent


def train_dqn_dynamic(
    train_seeds=None,
    val_seeds=None,
    episodes=10000,
    max_steps=500,
    hidden_dim=256,
    lr=3e-4,
    epsilon_decay=0.9995,
    map_rows=18,
    map_cols=24,
    obstacle_count=20,
    num_moving=2,
    checkpoint_dir=None,
    eval_freq=1000,
    device=None,
):
    """
    训练DQN，注入动态障碍物模拟玩家交互。
    """
    if train_seeds is None:
        train_seeds = list(range(100, 150))  # 50 maps
    if val_seeds is None:
        val_seeds = [7, 11, 21, 42, 66, 88, 100, 156, 202, 250]

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(RESULTS_DIR, "dqn_advanced")

    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, "dqn_model.pt")
    history_path = os.path.join(checkpoint_dir, "history.json")
    ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pkl")

    obs_dim = get_obs_shape(LOCAL_OBS_WINDOW)
    print(f"观测维度: {obs_dim}")
    print(f"训练地图: {len(train_seeds)}张, 验证: {len(val_seeds)}张")
    print(f"地图: {map_rows}x{map_cols}, 移动障碍: {num_moving}")
    print(f"轮数: {episodes}, 网络: {hidden_dim} hidden")
    print(f"课程学习: 动态障碍渐进注入")

    # 尝试恢复
    start_episode = 1
    agent = None
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "rb") as f:
                ckpt = pickle.load(f)
            agent = DQNAgent(input_dim=obs_dim, hidden_dim=hidden_dim, lr=lr,
                             epsilon_decay=epsilon_decay)
            agent.load(model_path)
            agent.epsilon = ckpt["epsilon"]
            agent.optimizer.load_state_dict(ckpt["optimizer_state"])
            agent.memory = ckpt["memory"]
            history = ckpt["history"]
            start_episode = ckpt["episode"] + 1
            best_eval = ckpt.get("best_eval_rate", 0.0)
            print(f"从第 {start_episode} 轮恢复, eps={agent.epsilon:.3f}")
        except Exception as e:
            print(f"恢复失败 ({e}), 从头开始")
            agent = None
            start_episode = 1

    if agent is None:
        agent = DQNAgent(
            input_dim=obs_dim, hidden_dim=hidden_dim, lr=lr,
            epsilon_decay=epsilon_decay, buffer_capacity=100000,
        )
        history = {"rewards": [], "success": [], "collisions": [],
                   "timeouts": [], "steps": [], "epsilon": [],
                   "eval_rates": [], "eval_eps": []}
        best_eval = 0.0

    # 预缓存环境
    print("预创建训练环境...")
    train_envs = []
    for seed in train_seeds:
        env = create_dynamic_env(
            seed=seed, rows=map_rows, cols=map_cols,
            obstacle_count=obstacle_count, danger_radius=DANGER_RADIUS,
            num_moving=num_moving,
            obstacle_patterns=MOVING_OBSTACLE_PATTERNS[:num_moving],
        )
        train_envs.append(env)
    print(f"完成, {len(train_envs)} 个环境")

    recent_success = []
    pbar = tqdm(range(start_episode, episodes + 1), desc="DQN-Adv")
    for episode in pbar:
        env = random.choice(train_envs)

        # === 课程学习: 四阶段渐进 ===
        progress = episode / episodes
        if progress < 0.4:
            # Phase 1: 纯导航，零干扰
            inject_prob = 0.0
            inject_interval = 9999
        elif progress < 0.6:
            # Phase 2: 轻度注入 (10%概率，每100步)
            inject_prob = 0.10
            inject_interval = 100
        elif progress < 0.8:
            # Phase 3: 中度注入 (25%概率，每70步)
            inject_prob = 0.25
            inject_interval = 70
        else:
            # Phase 4: 重度注入 (40%概率，每50步)
            inject_prob = 0.40
            inject_interval = 50

        state = env.reset()
        obs = get_local_obs(env, state, LOCAL_OBS_WINDOW)
        total_reward = 0.0
        success = 0
        collision = 0
        timeout_flag = 0
        steps = 0

        # 记录本次注入的障碍物（结束时恢复）
        injected_cells = []

        for step in range(max_steps):
            action = agent.choose_action(obs, training=True)
            ns, reward, done, info = env.step(action)
            next_obs = get_local_obs(env, ns, LOCAL_OBS_WINDOW)

            if step == max_steps - 1 and not done:
                reward -= 100.0
                done = True
                timeout_flag = 1

            agent.store_experience(obs, action, reward, next_obs, done)

            obs = next_obs
            total_reward += reward
            steps += 1

            if len(agent.memory) >= agent.batch_size:
                agent.update()

            if info.get("collision", False):
                collision = 1

            if done:
                if info.get("goal", False) or env.is_goal(ns[0], ns[1]):
                    success = 1
                break

            # === 动态障碍物注入: 模拟玩家放障碍 ===
            if random.random() < inject_prob and step > 0 and step % inject_interval == 0:
                # 在机器人前方随机选一个FREE格子放障碍
                attempts = 0
                while attempts < 20:
                    r_offset = random.randint(-3, 3)
                    c_offset = random.randint(-3, 3)
                    rr = ns[0] + r_offset
                    cc = ns[1] + c_offset
                    if (0 <= rr < map_rows and 0 <= cc < map_cols
                            and env._static_grid[rr, cc] == FREE
                            and (rr, cc) != env.goal
                            and abs(rr - ns[0]) + abs(cc - ns[1]) > 1):
                        env._static_grid[rr, cc] = OBSTACLE
                        injected_cells.append((rr, cc))
                        break
                    attempts += 1

                # 同时随机移除一个之前注入的障碍（保持地图可通行）
                if len(injected_cells) > 8:
                    old = injected_cells.pop(0)
                    if env._static_grid[old[0], old[1]] == OBSTACLE:
                        env._static_grid[old[0], old[1]] = FREE

        agent.decay_epsilon()

        # 恢复注入的障碍物
        for rr, cc in injected_cells:
            if env._static_grid[rr, cc] == OBSTACLE:
                env._static_grid[rr, cc] = FREE

        history["rewards"].append(total_reward)
        history["success"].append(success)
        history["collisions"].append(collision)
        history["timeouts"].append(timeout_flag)
        history["steps"].append(steps)
        history["epsilon"].append(agent.epsilon)

        recent_success.append(success)
        if len(recent_success) > 100:
            recent_success.pop(0)

        # 评估
        if episode % eval_freq == 0 or episode == 1:
            eval_rate = evaluate_agent(agent, val_seeds, max_steps,
                                       map_rows, map_cols, obstacle_count, num_moving)
            history["eval_rates"].append(eval_rate)
            history["eval_eps"].append(episode)

            if eval_rate > best_eval:
                best_eval = eval_rate
                agent.save(model_path)
                tqdm.write(f"  >> 新最佳! 验证={eval_rate:.1%} (ep {episode})")

            # 保存检查点
            try:
                with open(ckpt_path, "wb") as f:
                    pickle.dump({
                        "episode": episode, "epsilon": agent.epsilon,
                        "optimizer_state": agent.optimizer.state_dict(),
                        "memory": agent.memory,
                        "history": history, "best_eval_rate": best_eval,
                    }, f)
                with open(history_path, "w") as f:
                    json.dump(history, f)
            except Exception:
                pass

        avg_r = np.mean(history["rewards"][-100:])
        succ_r = np.mean(recent_success) if recent_success else 0.0
        pbar.set_postfix({"r": f"{avg_r:.0f}", "succ": f"{succ_r:.1%}",
                         "eps": f"{agent.epsilon:.2f}",
                         "eval": f"{history['eval_rates'][-1]:.1%}" if history["eval_rates"] else "-"})

    # 最终保存
    agent.save(model_path)
    print(f"\n训练完成! best_eval={best_eval:.1%}")
    draw_curves(history, checkpoint_dir)
    return agent, history


def evaluate_agent(agent, val_seeds, max_steps, rows, cols, obs_count, num_moving):
    """评估智能体"""
    success_count = 0
    for seed in val_seeds:
        env = create_dynamic_env(
            seed=seed, rows=rows, cols=cols,
            obstacle_count=obs_count, danger_radius=DANGER_RADIUS,
            num_moving=num_moving,
            obstacle_patterns=MOVING_OBSTACLE_PATTERNS[:num_moving],
        )
        state = env.reset()
        obs = get_local_obs(env, state, LOCAL_OBS_WINDOW)
        for _ in range(max_steps):
            action = agent.choose_action(obs, training=False)
            ns, reward, done, info = env.step(action)
            obs = get_local_obs(env, ns, LOCAL_OBS_WINDOW)
            if done:
                if env.is_goal(ns[0], ns[1]):
                    success_count += 1
                break
    return success_count / len(val_seeds)


def draw_curves(history, out_dir):
    """绘制训练曲线"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rewards = history["rewards"]
    success = history["success"]
    eval_eps = history.get("eval_eps", [])
    eval_rates = history.get("eval_rates", [])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Reward
    ax = axes[0, 0]
    ax.plot(rewards, alpha=0.3, color="blue")
    if len(rewards) >= 100:
        smoothed = np.convolve(rewards, np.ones(100)/100, mode="valid")
        ax.plot(range(100, len(rewards)+1), smoothed, color="blue", linewidth=2)
    ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
    ax.set_title("DQN Reward Curve"); ax.grid(True)

    # Success rate
    ax = axes[0, 1]
    rates = [np.mean(success[max(0,i-99):i+1]) for i in range(len(success))]
    ax.plot(rates)
    ax.set_xlabel("Episode"); ax.set_ylabel("Success Rate (100-eps)")
    ax.set_title("Training Success Rate"); ax.set_ylim(0, 1.05); ax.grid(True)

    # Eval
    ax = axes[1, 0]
    if eval_eps:
        ax.plot(eval_eps, eval_rates, "o-", markersize=6, color="green")
    ax.set_xlabel("Episode"); ax.set_ylabel("Eval Success Rate")
    ax.set_title("Validation Performance"); ax.set_ylim(0, 1.05); ax.grid(True)

    # Epsilon
    ax = axes[1, 1]
    ax.plot(history["epsilon"])
    ax.set_xlabel("Episode"); ax.set_ylabel("Epsilon")
    ax.set_title("Exploration Decay"); ax.grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"曲线已保存到 {os.path.join(out_dir, 'training_curves.png')}")


def main():
    agent, history = train_dqn_dynamic(
        episodes=8000,
        max_steps=300,
        hidden_dim=256,
        lr=3e-4,
        epsilon_decay=0.9995,
        map_rows=12,
        map_cols=16,
        obstacle_count=12,
        num_moving=1,
        eval_freq=1000,
    )


if __name__ == "__main__":
    main()
