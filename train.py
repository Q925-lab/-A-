# train.py

import os
import numpy as np
from tqdm import tqdm

from q_learning import QLearningAgent


def train_q_learning(
    env,
    episodes=5000,
    max_steps=800,
    alpha=0.1,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    save_path=None,
):
    """
    训练 Q-learning 智能体。

    返回：
        agent: 训练后的智能体
        history: 训练记录，包括奖励、成功率、碰撞次数、超时次数、步数等
    """
    agent = QLearningAgent(
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )

    rewards_history = []
    success_history = []
    collision_history = []
    timeout_history = []
    steps_history = []

    recent_success = []

    for episode in tqdm(range(1, episodes + 1), desc="Q-learning Training"):
        state = env.reset()

        total_reward = 0.0
        success = 0
        collision = 0
        timeout_flag = 0
        steps = 0

        for step in range(max_steps):
            action = agent.choose_action(state, training=True)

            next_state, reward, done, info = env.step(action)

            # 关键修改：
            # 如果走到最大步数还没有到达终点，也没有碰撞，
            # 就把它视为“超时失败”，并且这个惩罚要参与 Q 表更新。
            timeout = (step == max_steps - 1 and not done)
            if timeout:
                reward -= 100.0
                done = True
                timeout_flag = 1
                info["timeout"] = True

            # Q-learning 更新必须放在 timeout 惩罚之后
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            if info.get("collision", False):
                collision = 1

            if done:
                row, col, _ = next_state
                if env.is_goal(row, col):
                    success = 1
                break

        agent.decay_epsilon()

        rewards_history.append(total_reward)
        collision_history.append(collision)
        timeout_history.append(timeout_flag)
        steps_history.append(steps)

        recent_success.append(success)
        if len(recent_success) > 100:
            recent_success.pop(0)

        success_rate = np.mean(recent_success)
        success_history.append(success_rate)

    history = {
        "rewards": rewards_history,
        "success_rates": success_history,
        "collisions": collision_history,
        "timeouts": timeout_history,
        "steps": steps_history,
    }

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        agent.save(save_path)
        print(f"Q 表已保存到：{save_path}")

    return agent, history


def extract_qlearning_path(env, agent, max_steps=1000):
    """
    使用训练后的 Q 表，从起点出发生成一条贪心路径。
    """
    state = env.reset()
    path_states = [state]

    visited = set()
    visited.add(state)

    collision = False
    success = False

    for _ in range(max_steps):
        action = agent.choose_action(state, training=False)

        next_state, reward, done, info = env.step(action)

        path_states.append(next_state)

        if info.get("collision", False):
            collision = True
            break

        row, col, _ = next_state

        if env.is_goal(row, col):
            success = True
            break

        # 如果进入循环，说明当前策略还不够好
        if next_state in visited:
            break

        visited.add(next_state)
        state = next_state

    return {
        "success": success,
        "collision": collision,
        "path_states": path_states,
    }