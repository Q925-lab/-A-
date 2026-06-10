# ppo_agent.py
"""
PPO (Proximal Policy Optimization) 训练模块

使用 stable-baselines3 的 PPO 实现，封装 Gymnasium 环境接口。

PPO 核心思想 (Schulman et al., 2017):
  - 策略梯度方法，使用 clip 机制限制更新步长
  - L(θ) = E[min(r_t(θ)·A_t,  clip(r_t(θ), 1-ε, 1+ε)·A_t)]
  - 通过限制新旧策略的 KL 散度，稳定训练过程

与 DQN 的对比:
  - DQN: 值函数方法 (Q-learning)，离散动作，off-policy
  - PPO: 策略梯度方法，连续+离散，on-policy，更稳定
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

from config import (
    ACTIONS, DIRECTIONS, FREE, OBSTACLE, DANGER, START, GOAL,
    MOVING_OBSTACLE_PATTERNS, DANGER_RADIUS, LOCAL_OBS_WINDOW,
    RESULTS_DIR,
)
from grid_env import create_dynamic_env, get_local_obs, get_obs_shape

# 尝试导入 SB3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("[WARN] stable-baselines3 未安装，PPO功能不可用")
    print("  pip install stable-baselines3 gymnasium")


class DynamicGridEnv(gym.Env):
    """
    将动态栅格环境包装为 Gymnasium 接口，供 SB3 使用。

    观测: 局部观测向量 (obs_dim,)
    动作: 离散 4 个 (forward, turn_left, turn_right, backward)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, seed=None, map_rows=12, map_cols=16,
                 obstacle_count=12, num_moving=1, max_steps=300):
        super().__init__()
        self.map_rows = map_rows
        self.map_cols = map_cols
        self.obstacle_count = obstacle_count
        self.num_moving = num_moving
        self.max_steps = max_steps
        self.init_seed = seed if seed is not None else np.random.randint(0, 1000)

        obs_dim = get_obs_shape(LOCAL_OBS_WINDOW)
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTIONS))

        self.action_list = ACTIONS
        self.env = None
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        s = seed if seed is not None else np.random.randint(0, 1000)
        self.env = create_dynamic_env(
            seed=s, rows=self.map_rows, cols=self.map_cols,
            obstacle_count=self.obstacle_count,
            danger_radius=DANGER_RADIUS,
            num_moving=self.num_moving,
            obstacle_patterns=MOVING_OBSTACLE_PATTERNS[:self.num_moving],
        )
        state = self.env.reset()
        obs = get_local_obs(self.env, state, LOCAL_OBS_WINDOW)
        self.current_step = 0
        return obs.astype(np.float32), {}

    def step(self, action_idx):
        self.current_step += 1
        action = self.action_list[action_idx]
        ns, reward, done, info = self.env.step(action)
        obs = get_local_obs(self.env, ns, LOCAL_OBS_WINDOW)

        # 超时截断
        if self.current_step >= self.max_steps and not done:
            reward -= 100.0
            done = True

        terminated = done and (info.get("collision", False) or
                               self.env.is_goal(ns[0], ns[1]))
        truncated = done and not terminated

        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def render(self):
        pass  # PPO训练时不需要渲染


def make_env(seed, map_rows=12, map_cols=16, obstacle_count=12, num_moving=1, max_steps=300):
    """工厂函数：创建环境实例"""
    def _init():
        return DynamicGridEnv(
            seed=seed, map_rows=map_rows, map_cols=map_cols,
            obstacle_count=obstacle_count, num_moving=num_moving,
            max_steps=max_steps)
    return _init


class ProgressCallback(BaseCallback):
    """自定义回调：打印训练进度"""
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        if self.n_calls % 10000 == 0:
            if "rollout/ep_rew_mean" in self.model.logger.name_to_value:
                r = self.model.logger.name_to_value["rollout/ep_rew_mean"]
                print(f"  [PPO] step={self.n_calls} reward={r:.1f}")
        return True


def train_ppo(
    train_seeds=None,
    total_timesteps=200000,
    map_rows=12,
    map_cols=16,
    obstacle_count=12,
    num_moving=1,
    max_steps=300,
    save_path=None,
    device="auto",
):
    """
    使用 stable-baselines3 PPO 训练。

    参数:
        train_seeds: 训练地图种子列表
        total_timesteps: 总训练步数（推荐20万+）
        save_path: 模型保存路径
    """
    if not HAS_SB3:
        raise ImportError("请先安装: pip install stable-baselines3 gymnasium")

    if train_seeds is None:
        train_seeds = list(range(100, 120))

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "ppo_train", "ppo_model")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 创建向量化环境（SB3要求）
    print(f"[PPO] Creating {len(train_seeds)} training environments...")
    env_fns = [make_env(s, map_rows, map_cols, obstacle_count, num_moving, max_steps)
               for s in train_seeds]
    env = DummyVecEnv(env_fns)

    # 可选：归一化观测
    # env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # 创建评估环境
    eval_env = DummyVecEnv([
        make_env(7, map_rows, map_cols, obstacle_count, num_moving, max_steps)])

    # PPO 超参（针对离散动作的小型网格环境优化）
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # 熵正则：鼓励探索
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[128, 128],  # 两个隐藏层，每层128
            activation_fn=torch.nn.ReLU,
        ),
        verbose=1,
        device=device,
    )

    print(f"[PPO] Training for {total_timesteps} timesteps...")
    print(f"  Policy: MLP(obs_dim -> 128 -> 128 -> 4)")
    print(f"  Learning rate: 3e-4, Clip range: 0.2, Entropy coef: 0.01")

    # 评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(save_path),
        log_path=os.path.dirname(save_path),
        eval_freq=max(total_timesteps // 10, 5000),
        n_eval_episodes=5,
        deterministic=True,
    )

    # 训练
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, ProgressCallback()],
        progress_bar=True,
    )

    # 保存
    model.save(save_path)
    print(f"[PPO] Model saved to {save_path}.zip")

    env.close()
    eval_env.close()

    return model


def evaluate_ppo(model_path, seeds=None, map_rows=12, map_cols=16,
                 obstacle_count=12, num_moving=1, max_steps=300, render=False):
    """
    评估训练好的PPO模型。

    返回: success_rate, details list
    """
    if not HAS_SB3:
        raise ImportError("请先安装: pip install stable-baselines3 gymnasium")

    if seeds is None:
        seeds = [7, 11, 21, 42, 66]

    model = PPO.load(model_path)
    results = []

    for seed in seeds:
        env = DynamicGridEnv(
            seed=seed, map_rows=map_rows, map_cols=map_cols,
            obstacle_count=obstacle_count, num_moving=num_moving,
            max_steps=max_steps)
        obs, _ = env.reset()
        total_reward = 0.0
        success = False
        steps = 0

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1
            if terminated or truncated:
                if env.env and env.env.is_goal(
                        env.env.current_state[0], env.env.current_state[1]):
                    success = True
                break

        results.append({
            "seed": seed, "success": success, "steps": steps,
            "reward": total_reward,
        })

    success_rate = sum(1 for r in results if r["success"]) / len(results)
    return success_rate, results


def main():
    """命令行入口"""
    import argparse
    parser = argparse.ArgumentParser(description="PPO Training")
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--map-rows", type=int, default=12)
    parser.add_argument("--map-cols", type=int, default=16)
    parser.add_argument("--obstacle-count", type=int, default=12)
    parser.add_argument("--num-moving", type=int, default=1)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--eval-only", type=str, default=None,
                       help="只评估，不训练。传入模型路径")
    args = parser.parse_args()

    if args.eval_only:
        rate, details = evaluate_ppo(
            args.eval_only, map_rows=args.map_rows, map_cols=args.map_cols,
            obstacle_count=args.obstacle_count, num_moving=args.num_moving)
        print(f"PPO Success Rate: {rate:.1%}")
        for d in details:
            print(f"  Seed {d['seed']}: {'OK' if d['success'] else 'FAIL'} "
                  f"steps={d['steps']} reward={d['reward']:.0f}")
    else:
        train_ppo(
            total_timesteps=args.timesteps,
            map_rows=args.map_rows,
            map_cols=args.map_cols,
            obstacle_count=args.obstacle_count,
            num_moving=args.num_moving,
            save_path=args.output,
        )


if __name__ == "__main__":
    main()
