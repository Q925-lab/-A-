# demo.py
"""
Pygame交互式演示：AI逃生挑战 & 对战模式

=== 展示模式（围堵玩法）===
- AI机器人自动从起点走到终点
- 玩家用鼠标点击地图放障碍阻挡AI
- AI利用DQN策略实时避障
- AI逃脱成功→玩家输 | AI撞墙→玩家赢

=== 对战模式（双人对战）===
- 两个AI从各自起点出发
- 玩家1（蓝方）给玩家2的AI放障碍
- 玩家2（红方）给玩家1的AI放障碍
- 能量系统控制障碍物消耗
- 先到终点的AI获胜

控制：
  主菜单: 1=展示模式, 2=对战模式
  展示模式:
    鼠标左键 = 放静态障碍物
    鼠标右键 = 放临时障碍物(5秒消失)
    空格 = 开始/暂停
    R = 重置
    S = 调速
    D = 切换DQN/A*/并排
    Q/ESC = 返回菜单
  对战模式:
    蓝方: Q=静态 W=巡逻 E=追踪 R=临时
    红方: I=静态 O=巡逻 P=追踪 左括号=临时
    Q/ESC = 返回菜单
"""

import os, sys, time, math, random
import numpy as np
import pygame

try:
    from d_star_lite import DStarLite
    HAS_DSTAR = True
except ImportError:
    HAS_DSTAR = False

try:
    from mcts_planner import MCTSPlanner
    HAS_MCTS = True
except ImportError:
    HAS_MCTS = False

# 安全字体函数
# 中文字体文件路径
_ZH_FONT = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
_ARIAL = "C:/Windows/Fonts/arial.ttf"

def _font(size, bold=False, name=None):
    """创建字体，优先支持中文"""
    try:
        # 对于中文显示，强制使用微软雅黑
        return pygame.font.Font(_ZH_FONT, size)
    except Exception:
        try:
            return pygame.font.Font(_ARIAL, size)
        except Exception:
            return pygame.font.Font(None, size)


from config import (
    FREE, OBSTACLE, DANGER, START, GOAL,
    DIRECTIONS, ACTIONS, MOVING_OBSTACLE_PATTERNS,
    LOCAL_OBS_WINDOW, DANGER_RADIUS,
)
from grid_env import create_dynamic_env, get_local_obs, DynamicGridEnv
from astar import astar_search

# ============================================================
# 颜色常量
# ============================================================
CLR_FREE      = (240, 240, 240)
CLR_OBSTACLE  = (50, 50, 50)
CLR_DANGER    = (255, 200, 120)
CLR_START     = (80, 180, 80)
CLR_GOAL      = (220, 60, 60)
CLR_MOVING    = (200, 50, 50)     # 移动障碍物
CLR_ROBOT     = (255, 215, 0)     # 机器人黄色
CLR_ROBOT2    = (100, 200, 255)   # 机器人蓝色(对战)
CLR_PATH      = (100, 150, 255)   # 路径蓝色
CLR_PATH2     = (255, 150, 100)   # 路径红色
CLR_ASTAR     = (0, 200, 100)     # A*路径绿
CLR_OBS_WIN   = (0, 100, 255, 80) # 局部观测框
CLR_GRID      = (220, 220, 220)
CLR_TEXT      = (30, 30, 30)
CLR_PANEL     = (245, 245, 245)
CLR_GOLD      = (255, 200, 0)
CLR_RED       = (240, 50, 50)
CLR_GREEN     = (50, 180, 50)
CLR_BLUE      = (80, 80, 240)
CLR_TEMP_OBS  = (255, 120, 0)     # 临时障碍橙色
CLR_CHASE_OBS = (180, 0, 200)     # 追踪障碍紫色

MAP_COLORS = {FREE: CLR_FREE, OBSTACLE: CLR_OBSTACLE,
              DANGER: CLR_DANGER, START: CLR_START, GOAL: CLR_GOAL}

# ============================================================
# 游戏常量
# ============================================================
MAX_ENERGY = 10
ENERGY_RECHARGE_RATE = 1.0 / 2.0  # 每秒回复1点

OBSTACLE_COSTS = {
    "static": 1,      # 静态障碍 [Q]
    "patrol": 2,      # 巡逻障碍 [W]
    "chase":  4,      # 追踪障碍 [E]
    "temp":   1,      # 临时障碍(5秒) [R]
    "lock":   5,      # 封锁区域(8秒)  [T]
}


# ============================================================
# 辅助: 加载DQN
# ============================================================
def load_dqn(model_path):
    from dqn_agent import DQNAgent
    from grid_env import get_obs_shape
    agent = DQNAgent(input_dim=get_obs_shape(LOCAL_OBS_WINDOW), hidden_dim=128)
    agent.load(model_path)
    agent.epsilon = 0.0
    return agent


# ============================================================
# 地图渲染器（共用）
# ============================================================
class GridRenderer:
    def __init__(self, screen, x, y, w, h, map_rows, map_cols, cell_size):
        self.screen = screen
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.rows, self.cols = map_rows, map_cols
        self.cell = cell_size

    def draw_grid(self, static_grid, moving_set, extra_obstacles=None):
        """绘制栅格地图"""
        for r in range(self.rows):
            for c in range(self.cols):
                px = self.x + c * self.cell
                py = self.y + r * self.cell
                rect = pygame.Rect(px, py, self.cell, self.cell)

                if (r, c) in moving_set:
                    color = CLR_MOVING
                elif extra_obstacles and (r, c) in extra_obstacles:
                    # 检查是否是临时障碍
                    obs_info = extra_obstacles[(r, c)]
                    if isinstance(obs_info, dict) and obs_info.get("type") == "temp":
                        color = CLR_TEMP_OBS
                    elif isinstance(obs_info, dict) and obs_info.get("type") == "chase":
                        color = CLR_CHASE_OBS
                    else:
                        color = CLR_CHASE_OBS
                else:
                    color = MAP_COLORS.get(static_grid[r, c], CLR_FREE)

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, CLR_GRID, rect, 1)

    def draw_robot(self, r, c, d, color=CLR_ROBOT):
        """绘制带朝向的机器人三角箭头"""
        cx = self.x + c * self.cell + self.cell // 2
        cy = self.y + r * self.cell + self.cell // 2
        dir_angles = {0: 90, 1: 0, 2: 270, 3: 180}
        angle = math.radians(dir_angles.get(d, 0))
        sz = self.cell // 2
        tip = (cx + sz * math.cos(angle), cy - sz * math.sin(angle))
        left  = (cx + sz * 0.6 * math.cos(angle + math.radians(140)),
                 cy - sz * 0.6 * math.sin(angle + math.radians(140)))
        right = (cx + sz * 0.6 * math.cos(angle - math.radians(140)),
                 cy - sz * 0.6 * math.sin(angle - math.radians(140)))
        pygame.draw.polygon(self.screen, color, [tip, left, right])
        pygame.draw.polygon(self.screen, (0, 0, 0), [tip, left, right], 1)

    def draw_path(self, positions, color=CLR_PATH, width=2):
        """绘制路径轨迹"""
        if len(positions) < 2:
            return
        pts = [(self.x + c * self.cell + self.cell // 2,
                self.y + r * self.cell + self.cell // 2)
               for r, c in positions]
        pygame.draw.lines(self.screen, color, False, pts, width)

    def draw_marker(self, r, c, text, bg_color, text_color=(255, 255, 255)):
        """绘制起点/终点标记"""
        cx = self.x + c * self.cell + self.cell // 2
        cy = self.y + r * self.cell + self.cell // 2
        pygame.draw.circle(self.screen, bg_color, (cx, cy), self.cell // 3)
        font = _font(max(10, self.cell // 2), bold=True, name="Arial")
        surf = font.render(text, True, text_color)
        self.screen.blit(surf, (cx - surf.get_width() // 2, cy - surf.get_height() // 2))

    def draw_obs_window(self, rr, cc):
        """绘制局部观测框"""
        half = LOCAL_OBS_WINDOW // 2
        ox = self.x + (cc - half) * self.cell
        oy = self.y + (rr - half) * self.cell
        sz = LOCAL_OBS_WINDOW * self.cell
        pygame.draw.rect(self.screen, (0, 100, 255), (ox, oy, sz, sz), 2)


# ============================================================
#  展示模式（围堵玩法）
# ============================================================
class ShowcaseMode:
    def __init__(self, screen, w, h, agent, map_rows=12, map_cols=16,
                 num_moving=1, seed=100):
        self.screen = screen
        self.agent = agent
        self.map_rows = map_rows
        self.map_cols = map_cols
        self.num_moving = num_moving
        self.seed = seed

        # 布局计算
        panel_w = 240
        grid_w = w - panel_w
        grid_h = h - 40
        self.cell = min(grid_w // map_cols, grid_h // map_rows)
        self.cell = max(self.cell, 10)

        self.renderer = GridRenderer(screen, 0, 0, 0, 0, map_rows, map_cols, self.cell)

        self._init_env()
        self._compute_astar()

        # 游戏状态
        self.paused = True
        self.speed = 5         # 5步/秒
        self.mode = "astar"
        self._step_timer = 0.0  # dqn / astar / both
        self.score = 0
        self.player_obstacles = {}   # {(r,c): {"type": "static", "timer": None}}
        self.temp_obstacles = {}     # {(r,c): remaining_seconds}
        self.running = True
        self.result = None  # None/'win'/'lose'

        # 状态已在 _init_env() 中设置，不要覆盖
        self.msg = "Press [Space] to start"

        # 字体
        self.font_s = _font( 14)
        self.font_m = _font( 18)
        self.font_l = _font( 24, bold=True)

    def _init_env(self):
        self.env = create_dynamic_env(
            seed=self.seed, rows=self.map_rows, cols=self.map_cols,
            obstacle_count=15, danger_radius=DANGER_RADIUS,
            num_moving=self.num_moving,
            obstacle_patterns=MOVING_OBSTACLE_PATTERNS[:self.num_moving],
        )
        self.state = self.env.reset()
        self.obs = get_local_obs(self.env, self.state, LOCAL_OBS_WINDOW)
        self.path_trail = [(self.state[0], self.state[1])]
        self.episode_done = False
        self.total_reward = 0.0
        self.step_count = 0
        self.player_obstacles = {}
        self.temp_obstacles = {}
        self.result = None

    def _compute_astar(self):
        """用当前机器人位置作为起点计算A*路径"""
        try:
            from grid_env import GridEnv
            cr, cc, cd = self.state if self.state else self.env.start + (1,)
            se = GridEnv(grid=self.env._static_grid,
                         start=(cr, cc), goal=self.env.goal,
                         start_dir=cd)
            result = astar_search(se)
            if result["success"]:
                self.astar_path = [(s[0], s[1]) for s in result["path_states"]]
                self.astar_state_path = result["path_states"]
                self.astar_step_index = 0
            else:
                self.astar_path = []
                self.astar_state_path = []
                self.astar_step_index = 0
        except Exception:
            self.astar_path = []
            self.astar_state_path = []
            self.astar_step_index = 0

    def _get_astar_action(self):
        """从A*路径提取下一步动作"""
        if not self.astar_state_path:
            return np.random.choice(ACTIONS)
        if self.astar_step_index >= len(self.astar_state_path) - 1:
            return np.random.choice(ACTIONS)

        cr, cc, cd = self.state
        tr, tc, td = self.astar_state_path[self.astar_step_index + 1]

        # 方向不同 → 转弯
        if cd != td:
            if (cd + 1) % 4 == td:
                return "turn_right"
            elif (cd - 1) % 4 == td:
                return "turn_left"
            else:
                return "turn_left"  # 180度

        # 方向相同 → 前进或后退
        dr, dc = DIRECTIONS[cd]
        if (tr, tc) == (cr + dr, cc + dc):
            return "forward"
        elif (tr, tc) == (cr - dr, cc - dc):
            return "backward"
        else:
            # 当前位置已在目标格（转弯后的状态）
            self.astar_step_index += 1
            return self._get_astar_action()

    def _compute_dstar(self):
        """用D* Lite计算路径（支持增量更新）"""
        if not HAS_DSTAR:
            self.dstar_path = []
            self.dstar_planner = None
            return
        try:
            self.dstar_planner = DStarLite(
                self.env._static_grid, self.env.start, self.env.goal)
            path = self.dstar_planner.plan()
            self.dstar_path = path if path else []
        except Exception:
            self.dstar_path = []
            self.dstar_planner = None

    def reset(self, new_seed=None):
        if new_seed is not None:
            self.seed = new_seed
        self._init_env()
        self._compute_astar()
        self.score = 0
        self.msg = f"Reset | Seed={self.seed} | Press [Space]"
        self.paused = True
        print(f"[Reset] Seed={self.seed}")

    def _is_blocked(self, r, c):
        """检查某个位置是否被玩家放置的障碍占据"""
        return ((r, c) in self.player_obstacles or
                (r, c) in self.temp_obstacles)

    def add_obstacle(self, r, c, obs_type="static"):
        """玩家放置障碍物"""
        if r < 0 or r >= self.map_rows or c < 0 or c >= self.map_cols:
            return False
        static = self.env._static_grid
        if static[r, c] != FREE:
            return False
        if (r, c) == self.env.start or (r, c) == self.env.goal:
            return False
        if self._is_blocked(r, c):
            return False
        # 也不能放在机器人当前位置
        if self.state is not None:
            rs, cs, _ = self.state
            if (r, c) == (rs, cs):
                return False

        if obs_type == "static":
            self.player_obstacles[(r, c)] = {"type": "static", "timer": None}
            self.env._static_grid[r, c] = OBSTACLE   # 写入地图让DQN能看见了
            self.score += 30
        elif obs_type == "temp":
            self.temp_obstacles[(r, c)] = 5.0
            self.env._static_grid[r, c] = OBSTACLE   # 同上
            self.score += 10

        self._compute_astar()  # 重规划A*
    # 三种模式含义
    # "astar":  A*控制机器人（绿线），绝对最优，需要全局地图
    # "dqn":    DQN控制机器人（蓝线），局部观测，训练不足可能撞墙
    # "both":   A*控制机器人（绿线），同时显示DQN建议轨迹（红虚线）做对比
    def _draw_both_comparison(self):
        """在both模式下绘制DQN建议轨迹作为对比"""
        if self.mode != "both":
            return
        # 模拟DQN在当前状态下的决策路径（只画不执行）
        temp_state = self.state
        temp_obs = self.obs
        ghost_positions = [(temp_state[0], temp_state[1])]
        seen = set()
        for _ in range(50):
            if not self.agent or temp_obs is None:
                break
            action = self.agent.choose_action(temp_obs, training=False)
            try:
                ns, valid, _ = self.env.get_next_state(temp_state, action)
                if not valid or ns == temp_state:
                    break
                if ns[:2] in [(p[0], p[1]) for p in ghost_positions]:
                    break
                pos = (ns[0], ns[1])
                if pos != ghost_positions[-1]:
                    ghost_positions.append(pos)
                temp_state = ns
            except Exception:
                break
            try:
                temp_obs = get_local_obs(self.env, temp_state, LOCAL_OBS_WINDOW)
            except Exception:
                break
        if len(ghost_positions) > 1:
            self.renderer.draw_path(ghost_positions, (255, 120, 120), 1)

    def remove_obstacle(self, r, c):
        if (r, c) in self.player_obstacles:
            del self.player_obstacles[(r, c)]
            self.env._static_grid[r, c] = FREE  # 恢复
        if (r, c) in self.temp_obstacles:
            del self.temp_obstacles[(r, c)]
            self.env._static_grid[r, c] = FREE  # 恢复

    def _check_collision_with_player_obs(self, r, c):
        """检查机器人是否撞到玩家放的障碍物"""
        return self._is_blocked(r, c)

    def _update_temp_obstacles(self, dt):
        """更新临时障碍物计时器"""
        expired = []
        for pos, remaining in list(self.temp_obstacles.items()):
            remaining -= dt
            if remaining <= 0:
                expired.append(pos)
            else:
                self.temp_obstacles[pos] = remaining
        for pos in expired:
            del self.temp_obstacles[pos]
            r, c = pos
            self.env._static_grid[r, c] = FREE  # 过期恢复

    def step(self, dt=0.0):
        if self.episode_done:
            return
        print(f"[STEP] #{self.step_count+1} pos={self.state[:2] if self.state else 'None'}", end="\r")

        self._update_temp_obstacles(dt)

        # A*/D*/BOTH用搜索控制机器人，DQN用网络
        if self.mode in ("astar", "both"):
            self._compute_astar()
            action = self._get_astar_action()
        elif self.mode == "dstar":
            self._compute_dstar()
            self._compute_astar()
            action = self._get_astar_action()
        elif self.mode == "mcts" and HAS_MCTS:
            if not hasattr(self, 'mcts_planner') or self.mcts_planner is None:
                self.mcts_planner = MCTSPlanner(self.env, rollouts=400, max_depth=60)
            action = self.mcts_planner.choose_action(self.state)
        elif self.agent and self.obs is not None and self.mode == "dqn":
            action = self.agent.choose_action(self.obs, training=False)
        else:
            action = np.random.choice(ACTIONS)

        ns, reward, done, info = self.env.step(action)
        self.obs = get_local_obs(self.env, ns, LOCAL_OBS_WINDOW)

        nr, nc, nd = ns
        old_r, old_c, _ = self.state

        # 检查是否撞到玩家放的障碍物
        if self._check_collision_with_player_obs(nr, nc):
            info["player_collision"] = True
            reward -= 100
            done = True

        self.state = ns
        self.total_reward += reward
        self.step_count += 1
        if (nr, nc) != (old_r, old_c):
            self.path_trail.append((nr, nc))

        if done:
            self.episode_done = True
            if info.get("collision") or info.get("player_collision"):
                self.result = "win"  # 玩家赢了（AI撞了）
                self.msg = f"[WIN] AI撞墙! | 步数={self.step_count}"
                self.score += 200
            elif self.env.is_goal(nr, nc):
                self.result = "lose"
                self.msg = f"[LOSE] AI逃跑了！步数={self.step_count} 奖励={self.total_reward:.0f}"
                self.score += 500
            else:
                self.result = "win"
                self.msg = f"🏆 超时！你守住啦 | 步数={self.step_count}"
                self.score += 100

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                k = event.key
                sc = event.scancode
                print(f"[KEY] key={k} scan={sc} name={pygame.key.name(k)}")
                # 用 scancode 避免中文输入法拦截
                if sc == 44 or k == pygame.K_SPACE:  # SPACE
                    self.paused = not self.paused
                elif sc == 21 or k == pygame.K_r:    # R
                    self.reset(self.seed)
                elif sc == 7  or k == pygame.K_d:    # D (物理扫描码=7)
                    modes = ["dqn", "astar", "dstar", "both"]
                    try:    idx = modes.index(self.mode)
                    except: idx = 0
                    self.mode = modes[(idx + 1) % len(modes)]
                    pygame.display.set_caption(
                        f"Mode: {self.mode.upper()} | Seed={self.seed} | Space=Pause R=Reset")
                    self.msg = f">>> MODE: {self.mode.upper()} <<<"
                    print(f"[KEY] D pressed -> Mode = {self.mode.upper()}")
                    self._compute_astar()
                    self._compute_dstar()
                elif sc == 22 or k == pygame.K_s:    # S
                    speeds = [1, 2, 5, 10, 20]
                    try:    idx = speeds.index(self.speed)
                    except: idx = 0
                    self.speed = speeds[(idx + 1) % len(speeds)]
                    print(f"[KEY] S: speed={self.speed}")
                elif pygame.K_1 <= k <= pygame.K_5:
                    seeds = [100, 105, 110, 115, 120]
                    self.reset(seeds[k - pygame.K_1])
                elif k == pygame.K_ESCAPE:
                    return "menu"

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                r = my // self.cell
                c = mx // self.cell
                if event.button == 1:
                    self.add_obstacle(r, c, "static")
                    print(f"[CLICK] obstacle at ({r},{c})")
                elif event.button == 3:
                    self.add_obstacle(r, c, "temp")
                    print(f"[CLICK] temp at ({r},{c})")
        return "ok"

    def update(self, dt):
        if not self.paused and not self.episode_done:
            self.step(dt)
        elif self.paused and self.episode_done:
            pass  # game over

    def draw(self):
        self.screen.fill(CLR_PANEL)

        # 合并障碍物集合
        all_obs = {}
        all_obs.update(self.player_obstacles)
        all_obs.update({k: {"type": "temp"} for k in self.temp_obstacles})

        # 绘制地图
        self.renderer.draw_grid(
            self.env._static_grid,
            self.env._moving_obstacle_set(),
            extra_obstacles=all_obs,
        )

        # A*路径（绿色）
        if self.mode in ("astar", "both") and self.astar_path:
            self.renderer.draw_path(self.astar_path, CLR_ASTAR, 3)

        # D* Lite路径（橙色）
        if self.mode == "dstar" and self.dstar_path:
            self.renderer.draw_path(self.dstar_path, (220, 140, 40), 3)

        # DQN路径轨迹
        if self.path_trail:
            self.renderer.draw_path(self.path_trail, CLR_PATH, 2)

        # BOTH模式：DQN建议轨迹（红虚线对比）
        if self.mode == "both":
            self._draw_both_comparison()

        # 起点终点
        sr, sc = self.env.start
        gr, gc = self.env.goal
        self.renderer.draw_marker(sr, sc, "S", (255, 255, 255), (0, 0, 0))
        self.renderer.draw_marker(gr, gc, "G", (255, 60, 60))

        # 机器人
        if self.state:
            rr, cc, dd = self.state
            self.renderer.draw_robot(rr, cc, dd, CLR_ROBOT)

        # 右侧面板
        self._draw_panel()

        # 底部状态栏
        self._draw_status()

    def _draw_panel(self):
        px = self.map_cols * self.cell + 10
        py = 10

        def line(text, y, color=CLR_TEXT, font=None):
            f = font or self.font_s
            surf = f.render(text, True, color)
            self.screen.blit(surf, (px, y))
            return y + 24

        y = line("AI Escape Challenge", py, CLR_GOLD, self.font_l)
        y += 8
        y = line(f"地图: {self.seed} ({self.map_rows}×{self.map_cols})", y)
        y = line(f"移动障碍: {self.num_moving}个", y)
        y += 6

        # 游戏状态
        if self.result == "win":
            y = line("状态: 你赢了!", y, CLR_GREEN, self.font_m)
        elif self.result == "lose":
            y = line("状态: AI逃脱!", y, CLR_RED, self.font_m)
        elif self.paused:
            y = line("状态: 暂停中", y, (120, 120, 120), self.font_m)
        else:
            y = line("状态: 运行中", y, CLR_GREEN, self.font_m)

        y += 4
        y = line(f"得分: {self.score}", y, CLR_GOLD, self.font_m)
        y = line(f"步数: {self.step_count}", y)
        y = line(f"奖励: {self.total_reward:.0f}", y)
        y = line(f"模式: {self.mode.upper()}", y)
        y = line(f"速度: {self.speed}×", y)
        if self.player_obstacles or self.temp_obstacles:
            n_obs = len(self.player_obstacles) + len(self.temp_obstacles)
            y = line(f"你的障碍: {n_obs}个", y, CLR_TEMP_OBS)
        y += 8

        y = line("── 操作 ──", y, font=self.font_s)
        y = line("[空格] 开始/暂停", y)
        y = line("[S] 调速(1/2/5/10)", y)
        y = line("[D] DQN/A*/并排", y)
        y = line("[R] 重置 [1-5]换图", y)
        y = line("[左键] 放障碍", y)
        y = line("[右键] 临时障碍(5s)", y)
        y += 5
        y = line("/! 暂停时可放障碍 /!", y, (200, 50, 50), self.font_m)

    def _draw_status(self):
        sy = self.map_rows * self.cell + 5
        txt = (f"Seed={self.seed} | 步={self.step_count} | "
               f"奖励={self.total_reward:.0f} | 得分={self.score} | "
               f"速度={self.speed}× | {self.msg}")
        surf = self.font_s.render(txt, True, CLR_TEXT)
        self.screen.blit(surf, (10, sy))


# ============================================================
#  对战模式
# ============================================================
#  对战模式 V2：双地图并排，各自独立
#  蓝方操作 → 给红方地图放障碍   |   红方操作 → 给蓝方地图放障碍
# ============================================================
class BattleMode:
    def __init__(self, screen, w, h, agent1, agent2=None,
                 map_rows=12, map_cols=16, num_moving=1, seed=42):
        self.screen = screen
        self.agent_blue = agent1
        self.agent_red = agent2 or agent1
        self.map_rows, self.map_cols = map_rows, map_cols
        self.num_moving = num_moving
        self.seed = seed

        # 两张独立地图：并排
        gap = 40
        panel_w = 180
        map_w = self.map_cols * 12   # cell_size=12
        map_h = self.map_rows * 12
        self.cell = 12

        # 地图区域居中
        total_w = panel_w * 2 + map_w * 2 + gap
        total_h = map_h + 40
        offset_x = (w - total_w) // 2
        offset_y = max(10, (h - total_h) // 2)

        self.blue_rect = (offset_x + panel_w, offset_y, map_w, map_h)
        self.red_rect = (offset_x + panel_w + map_w + gap, offset_y, map_w, map_h)

        # 两个独立环境
        from grid_env import generate_random_thin_obstacles, add_danger_zones, DynamicGridEnv
        g1, s1, g1g = generate_random_thin_obstacles(rows=map_rows, cols=map_cols, obstacle_count=12, seed=seed)
        g1 = add_danger_zones(g1, DANGER_RADIUS); g1[s1] = 3; g1[g1g] = 4
        self.env_blue = DynamicGridEnv(g1, s1, g1g, 1)

        g2, s2, g2g = generate_random_thin_obstacles(rows=map_rows, cols=map_cols, obstacle_count=12, seed=seed+100)
        g2 = add_danger_zones(g2, DANGER_RADIUS); g2[s2] = 3; g2[g2g] = 4
        self.env_red = DynamicGridEnv(g2, s2, g2g, 1)

        # 两个渲染器
        self.render_blue = GridRenderer(screen, *self.blue_rect[:2], 0, 0, map_rows, map_cols, self.cell)
        self.render_red = GridRenderer(screen, *self.red_rect[:2], 0, 0, map_rows, map_cols, self.cell)

        # 游戏状态
        self.paused = True
        self.speed = 3
        self.game_over = False
        self.winner = None
        self._step_timer = 0.0

        # 双方状态
        from grid_env import get_local_obs
        self.blue_state = self.env_blue.reset()
        self.red_state = self.env_red.reset()
        self.blue_path = [(self.blue_state[0], self.blue_state[1])]
        self.red_path = [(self.red_state[0], self.red_state[1])]
        self.blue_steps = 0; self.red_steps = 0
        self.blue_done = False; self.red_done = False

        # 能量
        self.blue_energy = 10.0; self.red_energy = 10.0

        # 玩家放置的障碍物 {pos: "blue"|"red"}
        self.placed_obstacles = {}
        self.temp_obstacles = {}  # {pos: timer}

        self.font_s = _font(12)
        self.font_m = _font(16)
        self.font_l = _font(22, bold=True)

    def reset(self, new_seed=None):
        if new_seed: self.seed = new_seed
        self.__init__(self.screen, self.screen.get_width(), self.screen.get_height(),
                      self.agent_blue, self.agent_red,
                      self.map_rows, self.map_cols, self.num_moving, self.seed)

    def _place_obstacle(self, side, r, c):
        """玩家(side)在对方地图上放障碍"""
        player_energy = self.blue_energy if side == "blue" else self.red_energy
        if player_energy < 1: return False
        target_env = self.env_red if side == "blue" else self.env_blue
        if r < 0 or r >= self.map_rows or c < 0 or c >= self.map_cols: return False
        if target_env._static_grid[r, c] != 0: return False
        # 不在对方起/终点上
        if side == "blue":
            if (r, c) == self.env_red.goal or (r, c) == self.env_red.start: return False
        else:
            if (r, c) == self.env_blue.goal or (r, c) == self.env_blue.start: return False
        # 不在对方机器人当前位置
        rr, rc, _ = self.red_state if side == "blue" else self.blue_state
        if (r, c) == (rr, rc): return False

        target_env._static_grid[r, c] = 1
        self.placed_obstacles[(r, c)] = side
        if side == "blue": self.blue_energy -= 1
        else: self.red_energy -= 1
        return True

    def _step_ai(self, env, state, path, agent):
        """让AI走一步，返回(new_state, done, collision)"""
        from config import DIRECTIONS
        obs = get_local_obs(env, state, LOCAL_OBS_WINDOW)
        action = agent.choose_action(obs, training=False)
        r, c, d = state

        if action == "forward":
            dr, dc = DIRECTIONS[d]; nr, nc = r+dr, c+dc; nd = d
        elif action == "backward":
            dr, dc = DIRECTIONS[d]; nr, nc = r-dr, c-dc; nd = d
        elif action == "turn_left":
            nr, nc = r, c; nd = (d-1) % 4
        elif action == "turn_right":
            nr, nc = r, c; nd = (d+1) % 4
        else:
            nr, nc, nd = r, c, d

        if not (0<=nr<self.map_rows and 0<=nc<self.map_cols):
            return state, True, True
        if env._static_grid[nr, nc] == 1:
            return state, True, True

        ns = (nr, nc, nd)
        if (nr, nc) != (r, c):
            path.append((nr, nc))
        return ns, (env.is_goal(nr, nc)), False

    def step(self, dt=0.0):
        if self.game_over: return

        # 更新能量
        self.blue_energy = min(10.0, self.blue_energy + 0.5 * dt)
        self.red_energy = min(10.0, self.red_energy + 0.5 * dt)

        # 过期临时障碍
        expired = [p for p, t in self.temp_obstacles.items() if t <= dt]
        for p in expired:
            del self.temp_obstacles[p]
            # restore cell based on which env
            if (p[0], p[1]) == p: pass  # already deleted
        for p in list(self.temp_obstacles.keys()):
            self.temp_obstacles[p] -= dt

        # 蓝方AI走一步
        if not self.blue_done:
            ns, done, coll = self._step_ai(self.env_blue, self.blue_state, self.blue_path, self.agent_blue)
            self.blue_state = ns; self.blue_steps += 1
            if coll: self.blue_done = True
            elif done: self.blue_done = True; self.winner = "蓝方"

        # 红方AI走一步
        if not self.red_done:
            ns, done, coll = self._step_ai(self.env_red, self.red_state, self.red_path, self.agent_red)
            self.red_state = ns; self.red_steps += 1
            if coll: self.red_done = True
            elif done: self.red_done = True; self.winner = "红方"

        if self.blue_done and self.red_done:
            self.game_over = True
            if not self.winner:
                self.winner = "平局"
        elif self.blue_done:
            self.game_over = True
            self.winner = "红方" if not self.red_done else self.winner
        elif self.red_done:
            self.game_over = True
            self.winner = "蓝方" if not self.blue_done else self.winner

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.QUIT: return "quit"
            if event.type == pygame.KEYDOWN:
                k, sc = event.key, event.scancode
                if k == pygame.K_SPACE:
                    self.paused = not self.paused
                elif k == pygame.K_r or sc == 21:
                    self.reset(self.seed)
                elif k == pygame.K_ESCAPE: return "menu"

                # 蓝方给红方地图放障碍 (Q W E)
                elif sc in (16, 17, 18):  # Q W E
                    mx, my = pygame.mouse.get_pos()
                    # 判断鼠标在哪个地图上
                    rx, ry, rw, rh = self.red_rect
                    if rx <= mx < rx+rw and ry <= my < ry+rh:
                        r = (my - ry) // self.cell
                        c = (mx - rx) // self.cell
                        ok = self._place_obstacle("blue", r, c)
                        print(f"[BATTLE] Blue -> red map ({r},{c}) {'OK' if ok else 'FAIL'}")

                # 红方给蓝方地图放障碍 (I O P)
                elif sc in (31, 32, 33):
                    mx, my = pygame.mouse.get_pos()
                    bx, by, bw, bh = self.blue_rect
                    if bx <= mx < bx+bw and by <= my < by+bh:
                        r = (my - by) // self.cell
                        c = (mx - bx) // self.cell
                        ok = self._place_obstacle("red", r, c)
                        print(f"[BATTLE] Red -> blue map ({r},{c}) {'OK' if ok else 'FAIL'}")

        return "ok"

    def update(self, dt):
        if not self.paused and not self.game_over:
            self.step(dt)

    def draw(self):
        self.screen.fill(CLR_PANEL)

        # 蓝方地图
        self.render_blue.draw_grid(self.env_blue._static_grid,
                                   self.env_blue._moving_obstacle_set())
        if self.blue_path: self.render_blue.draw_path(self.blue_path, CLR_PATH)
        self.render_blue.draw_marker(*self.env_blue.start, "S", (255,255,255), (0,0,0))
        self.render_blue.draw_marker(*self.env_blue.goal, "G", (255,60,60))
        if self.blue_state:
            self.render_blue.draw_robot(*self.blue_state, CLR_ROBOT)

        # 红方地图
        self.render_red.draw_grid(self.env_red._static_grid,
                                  self.env_red._moving_obstacle_set())
        if self.red_path: self.render_red.draw_path(self.red_path, CLR_PATH2)
        self.render_red.draw_marker(*self.env_red.start, "S", (255,255,255), (0,0,0))
        self.render_red.draw_marker(*self.env_red.goal, "G", (255,60,60))
        if self.red_state:
            self.render_red.draw_robot(*self.red_state, CLR_ROBOT2)

        # 分隔线
        mid_x = self.red_rect[0] - 20
        pygame.draw.line(self.screen, (150,150,150), (mid_x, 10),
                         (mid_x, self.screen.get_height()-40), 2)
        l1 = self.font_m.render("蓝方地图", True, CLR_BLUE)
        l2 = self.font_m.render("红方地图", True, CLR_RED)
        self.screen.blit(l1, (self.blue_rect[0]+5, 5))
        self.screen.blit(l2, (self.red_rect[0]+5, 5))

        # 底部状态栏
        t = (f"蓝方: {self.blue_steps}步 能量:{self.blue_energy:.0f}  |  "
             f"红方: {self.red_steps}步 能量:{self.red_energy:.0f}  |  "
             f"Space=开始 R=重置 蓝(QWE)→红图  红(IOP)→蓝图")
        surf = self.font_s.render(t, True, CLR_TEXT)
        self.screen.blit(surf, (10, self.screen.get_height()-25))

        # 游戏结束
        if self.game_over:
            overlay = pygame.Surface((self.screen.get_width(), self.screen.get_height()))
            overlay.set_alpha(180); overlay.fill((0,0,0))
            self.screen.blit(overlay, (0,0))
            txt = f"  {self.winner or '平局'}!  [R]重新开始  [ESC]返回"
            surf = self.font_l.render(txt, True, CLR_GOLD)
            self.screen.blit(surf, ((self.screen.get_width()-surf.get_width())//2,
                                    self.screen.get_height()//2))


# ============================================================
#  主菜单 + 主循环
# ============================================================
class Game:
    def __init__(self):
        pygame.init()
        info = pygame.display.Info()
        self.w = int(info.current_w * 0.85)
        self.h = int(info.current_h * 0.85)
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("AI Escape Challenge")
        self.clock = pygame.time.Clock()
        try:
            self.font_l = _font( 36, bold=True)
            self.font_m = _font( 20)
            self.font_s = _font( 15)
        except Exception:
            self.font_l = pygame.font.Font(None, 36)
            self.font_m = pygame.font.Font(None, 20)
            self.font_s = pygame.font.Font(None, 15)

        # 加载DQN
        model_path = "results/dqn_train_v2/dqn_model.pt"
        if not os.path.exists(model_path):
            model_path = "results/dqn_train/dqn_model.pt"
        self.agent = None
        if os.path.exists(model_path):
            try:
                self.agent = load_dqn(model_path)
                print(f"[OK] DQN模型已加载: {model_path}")
            except Exception as e:
                print(f"[WARN] DQN加载失败: {e}")
        else:
            print(f"[WARN] 模型不存在: {model_path}")
            print("  请先运行 python dqn_train.py 训练模型")

        self.mode = None  # None=menu, "showcase", "battle"

    def draw_menu(self):
        self.screen.fill((20, 20, 40))

        title = self.font_l.render("AI Escape Challenge", True, CLR_GOLD)
        self.screen.blit(title, ((self.w - title.get_width()) // 2, 80))

        subtitle = self.font_m.render("人工智能基础课程大作业 - 动态障碍物环境下的路径规划演示", True, (200, 200, 200))
        self.screen.blit(subtitle, ((self.w - subtitle.get_width()) // 2, 140))

        y = 250
        opt1 = self.font_m.render("[1] 展示模式 — 放置障碍物阻挡AI逃脱", True, (255, 255, 255))
        opt2 = self.font_m.render("[2] 对战模式 — 双人对战，互相给对方AI设置障碍", True, (255, 255, 255))
        opt3 = self.font_m.render("[Q] 退出", True, (180, 180, 180))

        self.screen.blit(opt1, ((self.w - opt1.get_width()) // 2, y))
        self.screen.blit(opt2, ((self.w - opt2.get_width()) // 2, y + 50))
        self.screen.blit(opt3, ((self.w - opt3.get_width()) // 2, y + 100))

        # 说明
        y = y + 180
        hints = [
            "展示模式: 鼠标左键放静态障碍 | 右键放临时障碍(5秒) | 空格开始 | R重置",
            "对战模式: 蓝方[Q/W/E/R] | 红方[I/O/P/[] | 能量条自动回复",
            f"DQN模型: {'已加载 ✅' if self.agent else '未找到 ❌'}",
        ]
        for h in hints:
            surf = self.font_s.render(h, True, (140, 140, 160))
            self.screen.blit(surf, ((self.w - surf.get_width()) // 2, y))
            y += 22

    def run(self):
        running = True
        game_mode = None
        last_dt = time.time()

        while running:
            dt = time.time() - last_dt
            last_dt = time.time()

            events = pygame.event.get()

            try:
                if game_mode is None:
                    self.draw_menu()
                    for event in events:
                        if event.type == pygame.QUIT:
                            running = False
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_1:
                                if self.agent is None:
                                    print("DQN model not found! Run dqn_train.py first.")
                                else:
                                    game_mode = ShowcaseMode(self.screen, self.w, self.h, self.agent)
                            elif event.key == pygame.K_2:
                                if self.agent is None:
                                    print("DQN model not found! Run dqn_train.py first.")
                                else:
                                    game_mode = BattleMode(self.screen, self.w, self.h, self.agent)
                            elif event.key == pygame.K_ESCAPE:
                                running = False
                else:
                    result = game_mode.handle_events(events)
                    if result == "menu":
                        game_mode = None
                    elif result == "quit":
                        running = False
                        break

                    if game_mode and not game_mode.paused:
                        game_mode._step_timer += dt
                        interval = 1.0 / max(game_mode.speed, 1)
                        while game_mode._step_timer >= interval:
                            game_mode.update(interval)
                            game_mode._step_timer -= interval

                    if game_mode:
                        game_mode.draw()

                pygame.display.flip()
                self.clock.tick(60)

            except Exception as e:
                print(f"[ERROR] {e}")
                import traceback
                traceback.print_exc()
                running = False

        pygame.quit()


def main():
    Game().run()


if __name__ == "__main__":
    main()
