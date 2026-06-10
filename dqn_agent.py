# dqn_agent.py
"""
DQN智能体：Dueling Network + Prioritized Experience Replay
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple

from config import ACTIONS

Experience = namedtuple("Experience", [
    "state", "action", "reward", "next_state", "done", "priority"
])


# ============================================================
# Dueling DQN 网络
# ============================================================

class DuelingDQN(nn.Module):
    """
    Dueling Network: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    """
    def __init__(self, input_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.n_actions = n_actions

        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

    def forward(self, x):
        feat = self.feature(x)
        v = self.value(feat)
        a = self.advantage(feat)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q


# ============================================================
# Prioritized Experience Replay
# ============================================================

class PrioritizedReplayBuffer:
    """
    优先级经验回放。
    使用分段线性插值做重要性采样。
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        max_prio = max(self.priorities) if self.priorities else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.priorities.append(0.0)

        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        scaled = np.array(self.priorities[:len(self.buffer)]) ** self.alpha
        probs = scaled / scaled.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor([ACTIONS.index(a) for a in actions]),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones, dtype=np.float32)),
            indices,
            torch.FloatTensor(weights),
        )

    def update_priorities(self, indices, td_errors):
        for idx, td_err in zip(indices, td_errors):
            self.priorities[idx] = abs(td_err) + 1e-6


# ============================================================
# DQN Agent
# ============================================================

class DQNAgent:
    """
    DQN智能体：Dueling架构 + PER + Target Network
    """

    def __init__(
        self,
        input_dim,
        n_actions=None,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        buffer_capacity=50000,
        batch_size=64,
        target_update_freq=100,
        tau=0.005,
        device=None,
    ):
        self.actions = ACTIONS if n_actions is None else n_actions
        self.n_actions = len(self.actions)
        self.input_dim = input_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DuelingDQN(input_dim, self.n_actions, hidden_dim).to(self.device)
        self.target_net = DuelingDQN(input_dim, self.n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(capacity=buffer_capacity)

        self.train_step = 0

    def choose_action(self, obs, training=True):
        """
        epsilon-greedy 动作选择。
        """
        if training and random.random() < self.epsilon:
            return random.choice(self.actions)

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(obs_tensor).cpu().numpy().flatten()

        best_idx = np.argmax(q_values)
        return self.actions[best_idx]

    def store_experience(self, state_obs, action, reward, next_state_obs, done):
        self.memory.push(state_obs, action, reward, next_state_obs, done)

    def update(self):
        """
        执行一步训练更新。
        """
        sample = self.memory.sample(self.batch_size)
        if sample is None:
            return None

        states, actions, rewards, next_states, dones, indices, weights = sample
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # TD error
        td_error = (target_q - current_q).detach().cpu().numpy()

        # Weighted MSE loss
        loss = (weights * F.mse_loss(current_q, target_q, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities
        self.memory.update_priorities(indices, td_error)

        # Soft update target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self._soft_update()

        return loss.item()

    def _soft_update(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.target_net.eval()
