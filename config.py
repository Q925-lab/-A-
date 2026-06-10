# config.py

# ==============================
# 地图元素
# ==============================
FREE = 0            # 可通行区域
OBSTACLE = 1        # 障碍物
DANGER = 2          # 危险区域
START = 3           # 起点
GOAL = 4            # 终点
MOVING_OBSTACLE = 5 # 移动障碍物

# 方向：上、右、下、左
# 坐标采用 row, col，也就是 行、列
DIRECTIONS = {
    0: (-1, 0),   # up
    1: (0, 1),    # right
    2: (1, 0),    # down
    3: (0, -1),   # left
}

DIRECTION_NAMES = {
    0: "Up",
    1: "Right",
    2: "Down",
    3: "Left",
}

# 动作空间：移动机器人不是直接上下左右，而是带朝向约束的动作
ACTIONS = ["forward", "turn_left", "turn_right", "backward"]

# ==============================
# A* 代价参数
# ==============================
MOVE_COST = 1.0
BACKWARD_COST = 1.2
TURN_COST = 0.4
DANGER_COST = 0.8

# 边界软约束：不禁止靠边，但靠边代价更高
EDGE_COST = 1.0
EDGE_MARGIN = 2

# ==============================
# 输出目录
# ==============================
RESULTS_DIR = "results"

# ==============================
# 默认地图参数
# ==============================
MAP_ROWS = 30
MAP_COLS = 40
OBSTACLE_COUNT = 55
OBSTACLE_MIN_LEN = 2
OBSTACLE_MAX_LEN = 5
DANGER_RADIUS = 1
DEFAULT_SEED = 7

# 多地图实验种子。
# 程序会在 results/seed_xx/ 下分别保存每张地图的 A*、Q-learning、曲线与对比表。
EXPERIMENT_SEEDS = [7, 11, 21]

# ==============================
# 动态障碍物参数
# ==============================
NUM_MOVING_OBSTACLES = 3       # 移动障碍物数量
MOVING_OBSTACLE_PATTERNS = [   # 每个移动障碍物的行为模式
    "horizontal",               # 水平巡逻
    "vertical",                 # 垂直巡逻
    "random",                   # 随机游走
]

# ==============================
# 局部观测参数
# ==============================
LOCAL_OBS_WINDOW = 7           # 局部观测窗口大小（奇数）

# ==============================
# Q-learning 默认训练参数
# ==============================
Q_EPISODES = 10000
Q_MAX_STEPS = 1000
Q_ALPHA = 0.1
Q_GAMMA = 0.95
Q_EPSILON_START = 1.0
Q_EPSILON_MIN = 0.05
Q_EPSILON_DECAY = 0.999
