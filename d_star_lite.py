# d_star_lite.py
"""
D* Lite 增量启发式搜索算法

论文: Koenig & Likhachev, "D* Lite", AAAI 2002

核心思想:
  1. 从目标反向搜索到起点，维护一致的最短路径树
  2. 当边代价变化时（障碍物出现/消失），仅更新受影响节点
  3. 使用 km 偏移量处理起点移动
  4. 比完整重跑 A* 快一个数量级

数学关键:
  - g(s):  从 s 到目标的最短路径估计
  - rhs(s): one-step lookahead, rhs(s) = min_{s'}(c(s,s') + g(s'))
  - key(s): [min(g,rhs) + h(start,s) + km,  min(g,rhs)]
  - 当 g(s) = rhs(s) 时节点"一致"，否则加入优先队列修复

复杂度: O(n·log n) 最坏，实践中远优于 A* 重规划

用法:
    dstar = DStarLite(grid, start, goal)
    path = dstar.plan()           # 首次规划
    dstar.update_edge(r, c, cost) # 边代价变化
    path = dstar.replan()         # 增量修复
"""

import heapq
import numpy as np

INF = float('inf')


class DStarLite:
    """
    D* Lite 增量路径规划器。

    参数:
        grid: 2D numpy array, 0=FREE, 1=OBSTACLE, ...
        start: (row, col) 起点
        goal:  (row, col) 终点
    """

    def __init__(self, grid, start, goal):
        self.grid = np.array(grid, dtype=int)
        self.rows, self.cols = self.grid.shape
        self.start = start
        self.goal = goal
        self.last_start = start

        # 核心数据结构
        self.g = {}       # g-value
        self.rhs = {}     # rhs-value
        self.U = []       # 优先队列 (key, s)
        self.U_set = set()  # 快速查找
        self.km = 0       # 启发式偏移累积

        self._init()

    def _h(self, s, t):
        """启发式函数：曼哈顿距离 (一致性启发式)"""
        return abs(s[0] - t[0]) + abs(s[1] - t[1])

    def _c(self, s1, s2):
        """两点间转移代价。s1,s2是相邻格。如果s2是障碍则无穷。"""
        r, c = s2
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return INF
        if self.grid[r, c] == 1:  # OBSTACLE
            return INF
        return 1.0

    def _neighbors(self, s):
        """s的四个邻居"""
        r, c = s
        return [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]

    def _pred(self, s):
        """s的前驱 = 邻居 (无向图)"""
        return self._neighbors(s)

    def _succ(self, s):
        """s的后继 = 邻居 (无向图)"""
        return self._neighbors(s)

    def _key(self, s):
        """计算优先级 key = [k1, k2]"""
        g_s = self.g.get(s, INF)
        rhs_s = self.rhs.get(s, INF)
        min_g_rhs = min(g_s, rhs_s)
        k1 = min_g_rhs + self._h(self.start, s) + self.km
        k2 = min_g_rhs
        return (k1, k2)

    def _push(self, s):
        key = self._key(s)
        heapq.heappush(self.U, (key, s))
        self.U_set.add(s)

    def _pop(self):
        while self.U:
            key, s = heapq.heappop(self.U)
            self.U_set.discard(s)
            # 惰性删除：检查 key 是否过期
            if self._key(s) == key:
                return key, s
        return None

    def _top_key(self):
        while self.U:
            key, s = self.U[0]
            if s in self.U_set and self._key(s) == key:
                return key
            heapq.heappop(self.U)
            self.U_set.discard(s)
        return (INF, INF)

    def _update_vertex(self, u):
        """更新节点u的rhs值，维护队列一致性"""
        if u != self.goal:
            # rhs(u) = min_{s'∈Succ(u)} (c(u,s') + g(s'))
            min_cost = INF
            for sp in self._succ(u):
                c = self._c(u, sp)
                g_sp = self.g.get(sp, INF)
                if c + g_sp < min_cost:
                    min_cost = c + g_sp
            self.rhs[u] = min_cost

        if u in self.U_set:
            self.U_set.discard(u)
            # 注意：U中的旧entry会在pop时被惰性删除

        if self.g.get(u, INF) != self.rhs.get(u, INF):
            self._push(u)

    def _init(self):
        """初始化所有值"""
        self.g = {}
        self.rhs = {}
        self.U = []
        self.U_set = set()
        self.km = 0
        self.last_start = self.start

        # 所有节点初始化为无穷
        for r in range(self.rows):
            for c in range(self.cols):
                s = (r, c)
                self.g[s] = INF
                self.rhs[s] = INF

        # 目标节点的rhs=0，加入队列
        self.rhs[self.goal] = 0
        self._push(self.goal)

    def _compute_shortest_path(self):
        """核心：和A*类似的循环，直到起点一致"""
        while (self._top_key() < self._key(self.start) or
               self.g.get(self.start, INF) != self.rhs.get(self.start, INF)):
            popped = self._pop()
            if popped is None:
                break
            key, u = popped
            g_u = self.g.get(u, INF)
            rhs_u = self.rhs.get(u, INF)

            if g_u > rhs_u:
                # 过一致：g太高了，降低到rhs
                self.g[u] = rhs_u
                for pred in self._pred(u):
                    self._update_vertex(pred)
            else:
                # 欠一致：g太低了，设为INF
                self.g[u] = INF
                for pred in self._pred(u):
                    self._update_vertex(pred)
                self._update_vertex(u)

    def plan(self):
        """首次规划：从起点到目标的最优路径"""
        self._init()
        self._compute_shortest_path()
        return self._extract_path()

    def replan(self):
        """增量重规划：边代价变化后修复路径"""
        self._compute_shortest_path()
        return self._extract_path()

    def set_obstacle(self, r, c):
        """放置障碍物"""
        self.grid[r, c] = 1
        s = (r, c)
        # 将障碍物节点的g/rhs设为INF，强制不从该节点经过
        self.g[s] = INF
        self.rhs[s] = INF
        # 更新所有前驱：它们的rhs会因为边代价改变而变化
        for pred in self._pred(s):
            self._update_vertex(pred)
        # 更新节点本身（确保加入队列）
        self._update_vertex(s)

    def remove_obstacle(self, r, c):
        """移除障碍物"""
        self.grid[r, c] = 0
        s = (r, c)
        # 恢复：重新计算rhs
        self.g[s] = INF
        self._update_vertex(s)
        for pred in self._pred(s):
            self._update_vertex(pred)

    def update_start(self, new_start):
        """
        更新起点位置（用于实际步进时）
        返回: True如果到达目标，False继续
        """
        if new_start == self.goal:
            return True

        self.km += self._h(self.last_start, new_start)
        self.last_start = new_start
        self.start = new_start

        # 起点变了，需要重算，但只影响很少的节点
        self._compute_shortest_path()
        return False

    def _extract_path(self):
        """
        从当前g值中提取从起点到目标的最优路径。
        贪心：每步走向 g(s') + c(s,s') 最小的邻居。
        """
        if self.g.get(self.start, INF) == INF:
            return []  # 无路径

        path = [self.start]
        current = self.start
        visited = set([current])
        max_steps = self.rows * self.cols * 2

        for _ in range(max_steps):
            if current == self.goal:
                break

            # 找 g(s') + c(s,s') 最小的邻居
            best_next = None
            best_val = INF
            for nxt in self._succ(current):
                if nxt in visited:
                    continue
                c = self._c(current, nxt)
                g_nxt = self.g.get(nxt, INF)
                val = c + g_nxt
                if val < best_val:
                    best_val = val
                    best_next = nxt

            if best_next is None or best_val >= INF:
                break

            current = best_next
            path.append(current)
            visited.add(current)

        return path

    def get_all_distances(self):
        """返回所有节点的g值（即到目标的最短距离），用于可视化"""
        return {s: self.g.get(s, INF) for s in self.g}


# ============================================================
# 封装函数：兼容现有的展示模式
# ============================================================
def dstar_lite_step(grid, start, goal):
    """
    快速使用D* Lite：给定grid和起终点，返回路径+状态信息。
    返回: {"success": bool, "path": list of (r,c), "expanded": int, "cost": float}
    """
    dstar = DStarLite(grid, start, goal)
    path = dstar.plan()
    if not path:
        return {"success": False, "path": [], "expanded": 0, "cost": INF}
    cost = len(path) - 1 if path else 0
    return {"success": True, "path": path, "expanded": len(path), "cost": float(cost)}
