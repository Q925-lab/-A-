# grid_env.py

import numpy as np
from config import FREE, OBSTACLE, DANGER, START, GOAL, DIRECTIONS


class GridEnv:
    """
    二维栅格移动机器人环境。
    状态 state = (row, col, direction)
    direction: 0上, 1右, 2下, 3左
    """

    def __init__(self, grid, start, goal, start_dir=1):
        self.grid = np.array(grid, dtype=int)
        self.start = start
        self.goal = goal
        self.start_dir = start_dir
        self.start_state = (start[0], start[1], start_dir)

    @property
    def height(self):
        return self.grid.shape[0]

    @property
    def width(self):
        return self.grid.shape[1]

    def in_bounds(self, row, col):
        return 0 <= row < self.height and 0 <= col < self.width

    def is_obstacle(self, row, col):
        return self.grid[row, col] == OBSTACLE

    def is_danger(self, row, col):
        return self.grid[row, col] == DANGER

    def is_goal(self, row, col):
        return (row, col) == self.goal

    def is_valid_position(self, row, col):
        if not self.in_bounds(row, col):
            return False
        if self.is_obstacle(row, col):
            return False
        return True

    def get_next_state(self, state, action):
        """
        给定当前状态和动作，返回下一个状态。
        不直接修改环境，方便 A* 和 Q-learning 共用。

        返回：
        next_state, valid, info
        """
        row, col, direction = state

        if action == "turn_left":
            next_direction = (direction - 1) % 4
            next_state = (row, col, next_direction)
            return next_state, True, {"turn": True, "danger": self.is_danger(row, col)}

        if action == "turn_right":
            next_direction = (direction + 1) % 4
            next_state = (row, col, next_direction)
            return next_state, True, {"turn": True, "danger": self.is_danger(row, col)}

        dr, dc = DIRECTIONS[direction]

        if action == "forward":
            next_row = row + dr
            next_col = col + dc
            next_direction = direction

        elif action == "backward":
            next_row = row - dr
            next_col = col - dc
            next_direction = direction

        else:
            raise ValueError(f"Unknown action: {action}")

        if not self.is_valid_position(next_row, next_col):
            return state, False, {"collision": True}

        next_state = (next_row, next_col, next_direction)
        return next_state, True, {
            "turn": False,
            "danger": self.is_danger(next_row, next_col),
            "goal": self.is_goal(next_row, next_col),
        }
    
    def reset(self):
        """
        重置环境，用于每个 Q-learning episode 的开始。
        """
        self.current_state = self.start_state
        return self.current_state


    def get_distance_to_goal(self, row, col):
        """
        使用曼哈顿距离计算当前位置到目标点的距离。
        """
        return abs(row - self.goal[0]) + abs(col - self.goal[1])


    def step(self, action):
        """
        Q-learning 使用的环境交互接口。

        输入：
            action: 动作名称

        返回：
            next_state: 下一个状态
            reward: 奖励值
            done: 是否结束
            info: 额外信息
        """
        current_state = self.current_state
        row, col, direction = current_state

        old_distance = self.get_distance_to_goal(row, col)

        next_state, valid, info = self.get_next_state(current_state, action)

        reward = -1.0
        done = False

        # 撞墙或撞障碍物
        if not valid:
            reward = -100.0
            done = True
            return current_state, reward, done, {"collision": True}

        next_row, next_col, next_direction = next_state
        new_distance = self.get_distance_to_goal(next_row, next_col)

        # 基础步数惩罚
        reward = -1.0

        # 转弯惩罚
        if action in ["turn_left", "turn_right"]:
            reward -= 2.0

        # 后退惩罚
        if action == "backward":
            reward -= 1.0

        # 危险区域惩罚
        if self.is_danger(next_row, next_col):
            reward -= 5.0

        # 边界软约束：靠边不是禁止，但要扣分，避免贴边绕行
        distance_to_edge = min(
            next_row,
            next_col,
            self.height - 1 - next_row,
            self.width - 1 - next_col,
        )

        if distance_to_edge <= 2:
            reward -= 3.0

        # 靠近目标奖励 / 远离目标惩罚
        if new_distance < old_distance:
            reward += 3.0
        elif new_distance > old_distance:
            reward -= 3.0

        # 到达终点
        if self.is_goal(next_row, next_col):
            reward += 100.0
            done = True

        self.current_state = next_state

        return next_state, reward, done, info

def add_danger_zones(grid, radius=1):
    """
    在障碍物周围自动生成危险区域。
    危险区域不是障碍物，机器人可以经过，但代价更高。
    """
    grid = np.array(grid, dtype=int)
    danger_grid = grid.copy()

    rows, cols = grid.shape

    obstacle_positions = np.argwhere(grid == OBSTACLE)

    for r, c in obstacle_positions:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if danger_grid[nr, nc] == FREE:
                        danger_grid[nr, nc] = DANGER

    return danger_grid


def is_reachable(grid, start, goal):
    """
    用简单 BFS 检查起点到终点是否存在可行路径。
    这里只检查位置连通性，不考虑朝向。
    """
    from collections import deque

    rows, cols = grid.shape
    visited = set()
    queue = deque([start])
    visited.add(start)

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        r, c = queue.popleft()

        if (r, c) == goal:
            return True

        for dr, dc in moves:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            if (nr, nc) in visited:
                continue

            if grid[nr, nc] == OBSTACLE:
                continue

            visited.add((nr, nc))
            queue.append((nr, nc))

    return False


def keep_clear_area(grid, center, radius=2):
    """
    清空起点或终点附近区域，避免随机障碍堵在起点/终点旁边。
    """
    r0, c0 = center
    rows, cols = grid.shape

    for r in range(r0 - radius, r0 + radius + 1):
        for c in range(c0 - radius, c0 + radius + 1):
            if 0 <= r < rows and 0 <= c < cols:
                if grid[r, c] != OBSTACLE or r in [0, rows - 1] or c in [0, cols - 1]:
                    continue
                grid[r, c] = FREE

    return grid


def generate_random_thin_obstacles(
    rows=30,
    cols=40,
    obstacle_count=55,
    min_len=2,
    max_len=5,
    seed=42,
):
    """
    生成随机细条状障碍地图。
    特点：
    1. 每个障碍厚度为 1；
    2. 障碍长度较短；
    3. 障碍数量较多；
    4. 不生成迷宫式贯穿墙；
    5. 主要分布在中心区域，边缘附近少放。
    """
    rng = np.random.default_rng(seed)
    grid = np.zeros((rows, cols), dtype=int)

    # 外边界墙
    grid[0, :] = OBSTACLE
    grid[-1, :] = OBSTACLE
    grid[:, 0] = OBSTACLE
    grid[:, -1] = OBSTACLE

    # 起点和终点
    start = (3, 3)
    goal = (rows - 4, cols - 4)

    placed = 0
    attempts = 0
    max_attempts = obstacle_count * 30

    while placed < obstacle_count and attempts < max_attempts:
        attempts += 1

        # 主要在中心区域生成，避免贴边过多
        r = rng.integers(4, rows - 4)
        c = rng.integers(4, cols - 4)

        length = rng.integers(min_len, max_len + 1)
        horizontal = rng.random() < 0.5

        cells = []

        if horizontal:
            for k in range(length):
                cells.append((r, c + k))
        else:
            for k in range(length):
                cells.append((r + k, c))

        valid = True

        for rr, cc in cells:
            # 越界或太靠边，跳过
            if rr <= 1 or rr >= rows - 2 or cc <= 1 or cc >= cols - 2:
                valid = False
                break

            # 起点终点附近不放障碍
            if abs(rr - start[0]) + abs(cc - start[1]) <= 5:
                valid = False
                break

            if abs(rr - goal[0]) + abs(cc - goal[1]) <= 5:
                valid = False
                break

            # 已经有障碍则跳过
            if grid[rr, cc] == OBSTACLE:
                valid = False
                break

        if not valid:
            continue

        # 临时放置障碍
        backup = grid.copy()
        for rr, cc in cells:
            grid[rr, cc] = OBSTACLE

        # 检查不能把地图彻底堵死
        if is_reachable(grid, start, goal):
            placed += 1
        else:
            grid = backup

    return grid, start, goal


# ============================================================
# 动态障碍物环境
# ============================================================

import random as _random


def _place_moving_obstacles(grid, start, goal, num_obstacles, rng):
    """
    在可通行区域放置移动障碍物，避开起点终点。
    返回 [(row, col), ...]
    """
    rows, cols = grid.shape
    positions = []
    attempts = 0
    while len(positions) < num_obstacles and attempts < 500:
        r = rng.integers(2, rows - 2)
        c = rng.integers(2, cols - 2)
        if grid[r, c] != FREE:
            attempts += 1
            continue
        if abs(r - start[0]) + abs(c - start[1]) < 6:
            attempts += 1
            continue
        if abs(r - goal[0]) + abs(c - goal[1]) < 6:
            attempts += 1
            continue
        if (r, c) in positions:
            attempts += 1
            continue
        positions.append((r, c))
        attempts += 1
    return positions


class DynamicGridEnv:
    """
    带移动障碍物的二维栅格环境。
    状态 state = (row, col, direction)
    """

    def __init__(self, grid, start, goal, start_dir=1,
                 moving_obstacles=None,
                 obstacle_patterns=None,
                 danger_radius=1):
        self._static_grid = np.array(grid, dtype=int)
        self.start = start
        self.goal = goal
        self.start_dir = start_dir
        self.start_state = (start[0], start[1], start_dir)
        self.danger_radius = danger_radius

        # 移动障碍物
        self.moving_obstacles = moving_obstacles if moving_obstacles is not None else []
        self.obstacle_patterns = obstacle_patterns if obstacle_patterns is not None else []
        # 每个移动障碍物的巡逻状态
        self._patrol_state = []
        for i, (r, c) in enumerate(self.moving_obstacles):
            pattern = self.obstacle_patterns[i] if i < len(self.obstacle_patterns) else "random"
            self._patrol_state.append({
                "start": (r, c),
                "direction": 1,   # +1 或 -1
                "step_count": 0,
            })

        self.current_state = self.start_state

    @property
    def height(self):
        return self._static_grid.shape[0]

    @property
    def width(self):
        return self._static_grid.shape[1]

    def _moving_obstacle_set(self):
        return set(self.moving_obstacles)

    def in_bounds(self, row, col):
        return 0 <= row < self.height and 0 <= col < self.width

    def is_obstacle(self, row, col):
        return self._static_grid[row, col] == OBSTACLE

    def is_moving_obstacle(self, row, col):
        return (row, col) in self._moving_obstacle_set()

    def is_danger(self, row, col):
        return self._static_grid[row, col] == DANGER

    def is_goal(self, row, col):
        return (row, col) == self.goal

    def is_valid_position(self, row, col):
        if not self.in_bounds(row, col):
            return False
        if self.is_obstacle(row, col):
            return False
        if self.is_moving_obstacle(row, col):
            return False
        return True

    def get_distance_to_goal(self, row, col):
        return abs(row - self.goal[0]) + abs(col - self.goal[1])

    def get_next_state(self, state, action):
        """
        与静态版本相同的接口，但碰撞检测包含移动障碍物。
        """
        row, col, direction = state

        if action == "turn_left":
            next_direction = (direction - 1) % 4
            next_state = (row, col, next_direction)
            return next_state, True, {"turn": True, "danger": self.is_danger(row, col)}

        if action == "turn_right":
            next_direction = (direction + 1) % 4
            next_state = (row, col, next_direction)
            return next_state, True, {"turn": True, "danger": self.is_danger(row, col)}

        dr, dc = DIRECTIONS[direction]

        if action == "forward":
            next_row = row + dr
            next_col = col + dc
            next_direction = direction
        elif action == "backward":
            next_row = row - dr
            next_col = col - dc
            next_direction = direction
        else:
            raise ValueError(f"Unknown action: {action}")

        if not self.is_valid_position(next_row, next_col):
            return state, False, {"collision": True}

        next_state = (next_row, next_col, next_direction)
        return next_state, True, {
            "turn": False,
            "danger": self.is_danger(next_row, next_col),
            "goal": self.is_goal(next_row, next_col),
        }

    def _update_moving_obstacles(self):
        """每步后更新所有移动障碍物位置"""
        for i, (r, c) in enumerate(self.moving_obstacles):
            pattern = self.obstacle_patterns[i] if i < len(self.obstacle_patterns) else "random"
            ps = self._patrol_state[i]

            if pattern == "horizontal":
                # 水平来回巡逻
                new_c = c + ps["direction"]
                if new_c < 2 or new_c >= self.width - 2 or self._static_grid[r, new_c] == OBSTACLE:
                    ps["direction"] *= -1
                    new_c = c + ps["direction"]
                if 2 <= new_c < self.width - 2 and self._static_grid[r, new_c] != OBSTACLE:
                    self.moving_obstacles[i] = (r, new_c)

            elif pattern == "vertical":
                # 垂直来回巡逻
                new_r = r + ps["direction"]
                if new_r < 2 or new_r >= self.height - 2 or self._static_grid[new_r, c] == OBSTACLE:
                    ps["direction"] *= -1
                    new_r = r + ps["direction"]
                if 2 <= new_r < self.height - 2 and self._static_grid[new_r, c] != OBSTACLE:
                    self.moving_obstacles[i] = (new_r, c)

            elif pattern == "random":
                # 随机游走
                dr = _random.choice([-1, 0, 1])
                dc = _random.choice([-1, 0, 1])
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (2 <= nr < self.height - 2 and 2 <= nc < self.width - 2
                        and self._static_grid[nr, nc] != OBSTACLE
                        and (nr, nc) not in self.moving_obstacles):
                    self.moving_obstacles[i] = (nr, nc)

            elif pattern == "chase":
                # 追踪模式：向机器人方向移动
                if self.current_state:
                    robot_r, robot_c, _ = self.current_state
                    dr = 0 if robot_r == r else (1 if robot_r > r else -1)
                    dc = 0 if robot_c == c else (1 if robot_c > c else -1)
                    nr, nc = r + dr, c + dc
                    if (2 <= nr < self.height - 2 and 2 <= nc < self.width - 2
                            and self._static_grid[nr, nc] != OBSTACLE
                            and (nr, nc) not in self.moving_obstacles):
                        self.moving_obstacles[i] = (nr, nc)

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        current_state = self.current_state
        row, col, direction = current_state

        old_distance = self.get_distance_to_goal(row, col)

        next_state, valid, info = self.get_next_state(current_state, action)

        reward = -1.0
        done = False

        if not valid:
            reward = -100.0
            done = True
            return current_state, reward, done, {"collision": True}

        next_row, next_col, next_direction = next_state
        new_distance = self.get_distance_to_goal(next_row, next_col)

        reward = -1.0

        if action in ["turn_left", "turn_right"]:
            reward -= 2.0

        if action == "backward":
            reward -= 1.0

        if self.is_danger(next_row, next_col):
            reward -= 5.0

        distance_to_edge = min(
            next_row, next_col,
            self.height - 1 - next_row, self.width - 1 - next_col,
        )
        if distance_to_edge <= 2:
            reward -= 3.0

        if new_distance < old_distance:
            reward += 3.0
        elif new_distance > old_distance:
            reward -= 3.0

        if self.is_goal(next_row, next_col):
            reward += 100.0
            done = True

        self.current_state = next_state

        # 更新移动障碍物
        self._update_moving_obstacles()

        # 检查是否与移动障碍物碰撞（更新后）
        if not done:
            cr, cc, _ = self.current_state
            if self.is_moving_obstacle(cr, cc):
                reward = -100.0
                done = True
                info["collision"] = True

        return self.current_state, reward, done, info


def create_dynamic_env(
    seed=7,
    rows=30,
    cols=40,
    obstacle_count=55,
    min_len=2,
    max_len=5,
    danger_radius=1,
    start_dir=1,
    num_moving=3,
    obstacle_patterns=None,
):
    """
    创建带动态障碍物的随机地图。
    """
    rng = np.random.default_rng(seed)

    # 复用现有的静态地图生成
    grid, start, goal = generate_random_thin_obstacles(
        rows=rows, cols=cols,
        obstacle_count=obstacle_count,
        min_len=min_len, max_len=max_len,
        seed=seed,
    )

    # 放置移动障碍物
    moving_positions = _place_moving_obstacles(grid, start, goal, num_moving, rng)

    if obstacle_patterns is None:
        obstacle_patterns = ["horizontal", "vertical", "random"][:num_moving]

    # 生成危险区域
    grid = add_danger_zones(grid, radius=danger_radius)

    # 覆盖起点终点
    grid[start] = START
    grid[goal] = GOAL

    env = DynamicGridEnv(
        grid=grid, start=start, goal=goal, start_dir=start_dir,
        moving_obstacles=moving_positions,
        obstacle_patterns=obstacle_patterns,
        danger_radius=danger_radius,
    )
    return env


def get_local_obs(env, state=None, window_size=7):
    """
    提取以机器人为中心的局部观测窗口。

    参数：
        env: DynamicGridEnv 或 GridEnv
        state: (row, col, direction)，如果为None则使用 env.current_state
        window_size: 窗口大小（奇数）

    返回：
        obs: numpy array, shape (window_size*window_size*3 + 4 + 4,)
        包含：
        - window_size × window_size × 3 (障碍物/危险区/移动障碍物)
        - 4 (朝向one-hot)
        - 4 (目标相对位置: dx, dy, dist, goal_visible)
    """
    if state is None:
        state = env.current_state
    r, c, d = state

    half = window_size // 2
    h, w = env.height, env.width
    gr, gc = env.goal

    obs_channels = []

    # 移动障碍物集合
    moving_set = env._moving_obstacle_set() if hasattr(env, '_moving_obstacle_set') else set()

    for ch_idx in range(3):
        channel = np.zeros((window_size, window_size), dtype=np.float32)
        for i in range(window_size):
            for j in range(window_size):
                rr = r - half + i
                cc = c - half + j
                if 0 <= rr < h and 0 <= cc < w:
                    if ch_idx == 0:
                        # 静态障碍物
                        channel[i, j] = 1.0 if env.is_obstacle(rr, cc) else 0.0
                    elif ch_idx == 1:
                        # 危险区域
                        channel[i, j] = 1.0 if env.is_danger(rr, cc) else 0.0
                    elif ch_idx == 2:
                        # 移动障碍物
                        channel[i, j] = 1.0 if (rr, cc) in moving_set else 0.0
                else:
                    # 越界视为障碍物
                    channel[i, j] = 1.0
        obs_channels.append(channel.flatten())

    # 朝向 one-hot (4维)
    direction_onehot = np.zeros(4, dtype=np.float32)
    direction_onehot[d] = 1.0

    # 目标相对位置 (4维)
    dx = (gc - c) / max(w, 1)      # 归一化到 [-1, 1]
    dy = (gr - r) / max(h, 1)
    dist = np.sqrt(dx**2 + dy**2)  # 归一化距离
    goal_in_window = 1.0 if (abs(gr - r) <= half and abs(gc - c) <= half) else 0.0

    goal_info = np.array([dx, dy, dist, goal_in_window], dtype=np.float32)

    obs = np.concatenate(obs_channels + [direction_onehot, goal_info])
    return obs


def get_obs_shape(window_size=7):
    return window_size * window_size * 3 + 4 + 4  # 3ch*window² + direction(4) + goal_info(4)


def create_demo_env(
    seed=7,
    rows=30,
    cols=40,
    obstacle_count=55,
    min_len=2,
    max_len=5,
    danger_radius=1,
    start_dir=1,
):
    """
    创建开放式随机障碍地图。

    与第一版相比，这里把 seed 和地图复杂度参数显式暴露出来，
    方便批量生成多张地图，并在 results/seed_xx/ 下分类保存实验结果。

    参数：
        seed: 随机种子，用于复现实验地图；
        rows, cols: 地图尺寸；
        obstacle_count: 随机细障碍数量；
        min_len, max_len: 单个细障碍长度范围；
        danger_radius: 障碍物危险区半径；
        start_dir: 起始朝向。
    """
    grid, start, goal = generate_random_thin_obstacles(
        rows=rows,
        cols=cols,
        obstacle_count=obstacle_count,
        min_len=min_len,
        max_len=max_len,
        seed=seed,
    )

    # 生成危险区域
    grid = add_danger_zones(grid, radius=danger_radius)

    # 覆盖起点终点，避免被危险区覆盖
    grid[start] = START
    grid[goal] = GOAL

    env = GridEnv(grid=grid, start=start, goal=goal, start_dir=start_dir)
    return env
