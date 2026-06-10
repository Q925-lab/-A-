# mcts_planner.py
"""
蒙特卡洛树搜索 (MCTS) 路径规划

零训练，开箱即用。每一步从当前状态出发，模拟数百条路径，
选出最优动作。天然适应动态障碍物。

与课程选题三(博弈AI)直接关联，报告可写:
  Minimax → Alpha-Beta 剪枝 → MCTS (AlphaGo核心)

核心步骤 (每次决策):
  1. Selection:   从根节点出发，用UCB1公式选最有潜力的子节点
  2. Expansion:   到达叶节点后，展开一个未尝试过的动作
  3. Simulation:  从新节点开始，随机模拟直到终点/碰撞/超时
  4. Backpropagation: 将模拟结果回传更新所有祖先节点

UCB1公式:  UCB = win_rate + C * sqrt(ln(N_parent) / N_child)

复杂度: O(rollouts × depth), 不依赖训练, 在线搜索
"""

import math
import random
import time
import numpy as np

from config import ACTIONS, DIRECTIONS, FREE, OBSTACLE


class MCTSNode:
    __slots__ = ('state', 'action', 'parent', 'children',
                 'visits', 'wins', 'untried_actions')

    def __init__(self, state, action=None, parent=None):
        self.state = state          # (row, col, direction)
        self.action = action        # 从父节点到此节点的动作
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried_actions = list(ACTIONS)


class MCTSPlanner:
    """
    MCTS 路径规划器。

    参数:
        env: DynamicGridEnv 实例
        rollouts: 每次决策的模拟次数 (越大越强, 越慢)
        max_depth: 每次模拟的最大步数
        C: UCB1 探索常数 (1.4=标准, 2.0=更探索)
        use_heuristic: 是否在随机模拟中偏向目标方向
    """

    def __init__(self, env, rollouts=300, max_depth=80, C=1.4,
                 use_heuristic=True):
        self.env = env
        self.rollouts = rollouts
        self.max_depth = max_depth
        self.C = C
        self.use_heuristic = use_heuristic
        self.goal = env.goal

    def choose_action(self, state, obs=None):
        """
        对给定状态，运行MCTS并返回最优动作。
        返回: action string ("forward", "turn_left", ...)
        """
        root = MCTSNode(state)

        for _ in range(self.rollouts):
            node = root

            # 1. Selection: UCB1 下降到叶节点
            while not node.untried_actions and node.children:
                node = self._select_best_child(node)

            # 2. Expansion: 尝试未探索动作
            if node.untried_actions:
                node = self._expand(node)

            # 3. Simulation: 随机模拟
            reward = self._simulate(node)

            # 4. Backpropagation: 回传结果
            self._backpropagate(node, reward)

        # 返回访问次数最多的子节点的动作
        if not root.children:
            return random.choice(ACTIONS)

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

    def _select_best_child(self, node):
        """UCB1 选择"""
        log_n = math.log(max(node.visits, 1))
        best_val = -float('inf')
        best_child = None

        for child in node.children:
            if child.visits == 0:
                return child  # 未访问的优先
            exploitation = child.wins / child.visits
            exploration = self.C * math.sqrt(log_n / child.visits)
            ucb = exploitation + exploration
            if ucb > best_val:
                best_val = ucb
                best_child = child

        return best_child or random.choice(node.children)

    def _expand(self, node):
        """展开一个未尝试的动作"""
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)

        next_state, valid, _ = self.env.get_next_state(node.state, action)
        if not valid:
            # 碰撞动作 → 视为死节点
            child = MCTSNode(node.state, action, parent=node)
            child.visits = 1
            child.wins = -1.0  # 惩罚碰撞
            node.children.append(child)
            return child

        child = MCTSNode(next_state, action, parent=node)
        node.children.append(child)
        return child

    def _simulate(self, node):
        """从节点开始模拟，返回奖励"""
        state = node.state
        depth = 0
        visited = set()

        while depth < self.max_depth:
            r, c, d = state

            if self.env.is_goal(r, c):
                return 1.0

            if not (0 <= r < self.env.height and 0 <= c < self.env.width):
                return -1.0
            if self.env._static_grid[r, c] == OBSTACLE:
                return -1.0

            # 死循环检测
            state_key = (r, c, d)
            if state_key in visited:
                return -0.5
            visited.add(state_key)

            if self.use_heuristic:
                action = self._heuristic_action(state)
            else:
                action = random.choice(ACTIONS)

            next_state, valid, _ = self.env.get_next_state(state, action)
            if not valid:
                return -1.0

            state = next_state
            depth += 1

        dist = abs(state[0] - self.goal[0]) + abs(state[1] - self.goal[1])
        max_dist = self.env.height + self.env.width
        return 0.3 * (1.0 - dist / max_dist)

    def _heuristic_action(self, state):
        """启发式动作：选指向目标且不会撞墙的动作"""
        r, c, d = state
        gr, gc = self.goal

        actions_scores = []
        for action in ACTIONS:
            ns, valid, info = self.env.get_next_state(state, action)
            if not valid:
                continue  # 撞墙，跳过
            nr, nc, nd = ns
            # 曼哈顿距离 + 转弯代价
            dist = abs(nr - gr) + abs(nc - gc)
            if action in ("turn_left", "turn_right"):
                dist += 0.5  # 转弯有小代价
            elif action == "backward":
                dist += 1.0  # 后退代价大
            # 危险区代价
            if info.get("danger", False):
                dist += 0.3
            actions_scores.append((dist, action))

        if not actions_scores:
            return random.choice(ACTIONS)

        # 随机选择，偏向好动作 (90%概率选最优)
        actions_scores.sort()
        if random.random() < 0.90:
            return actions_scores[0][1]
        elif random.random() < 0.95:
            return actions_scores[min(1, len(actions_scores)-1)][1]
        else:
            return random.choice([a for _, a in actions_scores])

    def _backpropagate(self, node, reward):
        """沿树向上回传奖励"""
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent


def mcts_get_path(env, max_steps=300, rollouts=200, verbose=False):
    """
    使用MCTS从起点走到终点，返回路径。
    零训练，直接可用。
    """
    planner = MCTSPlanner(env, rollouts=rollouts, max_depth=60, C=1.4)

    state = env.reset()
    path = [(state[0], state[1])]
    total_steps = 0
    success = False
    collision = False

    for step in range(max_steps):
        action = planner.choose_action(state)
        next_state, reward, done, info = env.step(action)

        pos = (next_state[0], next_state[1])
        if pos != path[-1]:
            path.append(pos)

        state = next_state
        total_steps += 1

        if done:
            if info.get("collision", False):
                collision = True
            elif env.is_goal(next_state[0], next_state[1]):
                success = True
            break

        if verbose and step % 50 == 0:
            print(f"  Step {step}: pos=({state[0]},{state[1]}) dir={state[2]}")

    return {
        "success": success,
        "collision": collision,
        "path": path,
        "steps": total_steps,
    }
