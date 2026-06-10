# q_learning.py

import random
import numpy as np

from config import ACTIONS


class QLearningAgent:
    """
    表格型 Q-learning 智能体。
    Q 表使用字典保存：
        key: state = (row, col, direction)
        value: 每个动作对应的 Q 值数组
    """

    def __init__(
        self,
        actions=None,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    ):
        self.actions = actions if actions is not None else ACTIONS

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = {}

    def get_q_values(self, state):
        """
        如果状态没有出现过，则初始化该状态的 Q 值。
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions), dtype=float)
        return self.q_table[state]

    def choose_action(self, state, training=True):
        """
        epsilon-greedy 动作选择。
        training=True 时允许随机探索；
        training=False 时只选择当前 Q 值最大的动作。
        """
        q_values = self.get_q_values(state)

        if training and random.random() < self.epsilon:
            action_index = random.randrange(len(self.actions))
            return self.actions[action_index]

        max_q = np.max(q_values)

        # 随机选择一个最大 Q 值动作，避免总是偏向第一个动作
        best_indices = np.where(q_values == max_q)[0]
        action_index = random.choice(best_indices)

        return self.actions[action_index]

    def update(self, state, action, reward, next_state, done):
        """
        Q-learning 更新公式：
        Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') - Q(s,a)]
        """
        action_index = self.actions.index(action)

        q_values = self.get_q_values(state)
        current_q = q_values[action_index]

        if done:
            target_q = reward
        else:
            next_q_values = self.get_q_values(next_state)
            target_q = reward + self.gamma * np.max(next_q_values)

        q_values[action_index] = current_q + self.alpha * (target_q - current_q)

    def decay_epsilon(self):
        """
        逐渐降低探索率。
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """
        保存 Q 表。
        """
        np.save(path, self.q_table, allow_pickle=True)

    def load(self, path):
        """
        加载 Q 表。
        """
        self.q_table = np.load(path, allow_pickle=True).item()