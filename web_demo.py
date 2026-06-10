# web_demo.py
"""
Gradio Web 演示界面

提供浏览器可访问的交互式路径规划演示。
对比 A* / D* Lite / DQN 三种算法在动态障碍物下的表现。

运行: python web_demo.py
然后浏览器打开 http://127.0.0.1:7860
"""

import os, sys, time, io
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    FREE, OBSTACLE, DANGER, START, GOAL, LOCAL_OBS_WINDOW,
    MOVING_OBSTACLE_PATTERNS, DANGER_RADIUS, ACTIONS,
)
from grid_env import create_dynamic_env, get_local_obs, get_obs_shape
from d_star_lite import DStarLite
from astar import astar_search

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False
    print("[WARN] gradio 未安装: pip install gradio")


# 颜色映射
CELL_COLORS = {
    FREE:     (240, 240, 240),
    OBSTACLE: (50, 50, 50),
    DANGER:   (255, 200, 120),
    START:    (80, 180, 80),
    GOAL:     (220, 60, 60),
    "moving": (200, 50, 50),
    "path_a": (0, 180, 80),
    "path_dqn": (80, 120, 255),
    "path_dstar": (200, 120, 0),
    "robot":   (255, 215, 0),
}

# 全局状态
_dqn_agent = None


def _load_dqn():
    global _dqn_agent
    if _dqn_agent is not None:
        return _dqn_agent
    try:
        from dqn_agent import DQNAgent
        agent = DQNAgent(input_dim=get_obs_shape(LOCAL_OBS_WINDOW), hidden_dim=128)
        agent.load("results/dqn_train/dqn_model.pt")
        agent.epsilon = 0.0
        _dqn_agent = agent
        return agent
    except Exception:
        return None


def render_grid(grid, start, goal, moving_obstacles,
                a_star_path=None, d_star_path=None, dqn_path=None,
                robot_pos=None, cell_size=30):
    """渲染栅格地图为PIL Image"""
    rows, cols = grid.shape
    w, h = cols * cell_size, rows * cell_size
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    moving_set = set(tuple(m) for m in moving_obstacles)

    for r in range(rows):
        for c in range(cols):
            x1, y1 = c * cell_size, r * cell_size
            x2, y2 = x1 + cell_size, y1 + cell_size

            if (r, c) in moving_set:
                color = CELL_COLORS["moving"]
            elif grid[r, c] == START:
                color = CELL_COLORS[START]
            elif grid[r, c] == GOAL:
                color = CELL_COLORS[GOAL]
            else:
                color = CELL_COLORS.get(grid[r, c], CELL_COLORS[FREE])

            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(200, 200, 200))

    # 绘制路径
    half = cell_size // 2
    def draw_path(path, color, width=3):
        if not path or len(path) < 2:
            return
        pts = [(c * cell_size + half, r * cell_size + half) for r, c in path]
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i+1]], fill=color, width=width)

    if a_star_path:
        draw_path(a_star_path, CELL_COLORS["path_a"], 4)
    if d_star_path:
        draw_path(d_star_path, CELL_COLORS["path_dstar"], 3)
    if dqn_path:
        draw_path(dqn_path, CELL_COLORS["path_dqn"], 2)

    # 起点终点标记
    sr, sc = start
    gr_pos, gc = goal
    draw.ellipse([sc*cell_size+4, sr*cell_size+4, (sc+1)*cell_size-4, (sr+1)*cell_size-4],
                 outline=(0,0,0), width=2)
    draw.ellipse([gc*cell_size+4, gr_pos*cell_size+4, (gc+1)*cell_size-4, (gr_pos+1)*cell_size-4],
                 outline=(255,0,0), width=2)
    draw.text((sc*cell_size+half-4, sr*cell_size+half-8), "S", fill=(0,0,0))
    draw.text((gc*cell_size+half-4, gr_pos*cell_size+half-8), "G", fill=(255,0,0))

    # 机器人
    if robot_pos:
        rr, rc = robot_pos
        cx, cy = rc * cell_size + half, rr * cell_size + half
        draw.ellipse([cx-6, cy-6, cx+6, cy+6], fill=CELL_COLORS["robot"], outline=(0,0,0))

    return img


def _get_path(env, mode="astar"):
    """单一算法获取路径"""
    result = {"path": [], "success": False, "steps": 0, "cost": 0}

    if mode == "astar":
        from grid_env import GridEnv
        se = GridEnv(grid=env._static_grid, start=env.start,
                     goal=env.goal, start_dir=1)
        r = astar_search(se)
        if r["success"]:
            result["path"] = [(s[0], s[1]) for s in r["path_states"]]
            result["success"] = True
            result["cost"] = r["total_cost"]

    elif mode == "dstar":
        dstar = DStarLite(env._static_grid, env.start, env.goal)
        path = dstar.plan()
        if path:
            result["path"] = path
            result["success"] = True
            result["cost"] = len(path) - 1

    elif mode == "dqn":
        agent = _load_dqn()
        if agent is None:
            return result
        state = env.reset()
        obs = get_local_obs(env, state, LOCAL_OBS_WINDOW)
        path = [(state[0], state[1])]
        for step in range(300):
            action = agent.choose_action(obs, training=False)
            ns, r, done, info = env.step(action)
            obs = get_local_obs(env, ns, LOCAL_OBS_WINDOW)
            pos = (ns[0], ns[1])
            if pos != path[-1]:
                path.append(pos)
            if done:
                if env.is_goal(ns[0], ns[1]):
                    result["success"] = True
                break
        result["path"] = path
        result["steps"] = len(path) - 1

    return result


def compare_algorithms(seed, obstacle_count, num_moving, show_astar, show_dstar, show_dqn):
    """Gradio回调：运行算法对比并返回图片+统计"""
    env = create_dynamic_env(
        seed=int(seed), rows=12, cols=16,
        obstacle_count=int(obstacle_count),
        danger_radius=DANGER_RADIUS,
        num_moving=int(num_moving),
        obstacle_patterns=MOVING_OBSTACLE_PATTERNS[:int(num_moving)],
    )

    a_path, d_path, q_path = [], [], []
    stats = []

    if show_astar:
        r = _get_path(env, "astar")
        a_path = r["path"]
        stats.append(f"A*: {'OK' if r['success'] else 'FAIL'} | "
                     f"cost={r['cost']:.1f} | len={len(a_path)}")

    if show_dstar:
        r = _get_path(env, "dstar")
        d_path = r["path"]
        stats.append(f"D* Lite: {'OK' if r['success'] else 'FAIL'} | "
                     f"cost={r['cost']:.1f} | len={len(d_path)}")

    if show_dqn:
        r = _get_path(env, "dqn")
        q_path = r["path"]
        stats.append(f"DQN: {'OK' if r['success'] else 'FAIL'} | "
                     f"steps={r['steps']} | len={len(q_path)}")

    img = render_grid(
        env._static_grid, env.start, env.goal,
        env.moving_obstacles,
        a_star_path=a_path if show_astar else None,
        d_star_path=d_path if show_dstar else None,
        dqn_path=q_path if show_dqn else None,
        cell_size=30,
    )

    return img, "\n".join(stats)


def create_web_ui():
    """创建Gradio界面"""
    if not HAS_GRADIO:
        print("请先安装: pip install gradio")
        return

    with gr.Blocks(title="AI Path Planning Demo") as demo:
        gr.Markdown("""
        #  动态环境路径规划：A* vs D* Lite vs DQN
        ### 人工智能基础课程大作业
        ---
        选择地图参数，对比三种算法的路径规划效果。
        - **A-star**: 全局最优路径（需完整地图）
        - **D-star Lite**: 增量重规划（适应动态变化）
        - **DQN**: 深度强化学习（局部观测，学习型）
        """)

        with gr.Row():
            with gr.Column(scale=1):
                seed = gr.Number(label="地图种子 (Seed)", value=7, precision=0)
                obs_count = gr.Slider(5, 30, value=15, step=1, label="障碍物数量")
                num_moving = gr.Slider(0, 4, value=1, step=1, label="移动障碍物数量")

                gr.Markdown("#### 选择对比算法")
                show_a = gr.Checkbox(label="A*", value=True)
                show_d = gr.Checkbox(label="D* Lite", value=True)
                show_q = gr.Checkbox(label="DQN", value=True)

                btn = gr.Button("  运行对比", variant="primary", size="lg")

                stats = gr.Textbox(label="统计信息", lines=5)

            with gr.Column(scale=2):
                output_img = gr.Image(label="路径规划结果", type="pil", height=500)

        btn.click(
            fn=compare_algorithms,
            inputs=[seed, obs_count, num_moving, show_a, show_d, show_q],
            outputs=[output_img, stats],
        )

        gr.Markdown("""
        ---
        ### 算法说明
        | 算法 | 类型 | 优势 | 局限 |
        |------|------|------|------|
        | A-star | 确定性搜索 | 全局最优路径 | 需要完整地图，动态环境需重规划 |
        | D-star Lite | 增量搜索 | 动态环境效率高 | 仍需全局地图 |
        | DQN | 强化学习 | 局部观测即可，实时反应 | 需要大量训练，泛化有限 |
        """)

    return demo


def main():
    if not HAS_GRADIO:
        print("=" * 50)
        print("请先安装 gradio:")
        print("  pip install gradio")
        print("=" * 50)
        return

    demo = create_web_ui()
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False, inbrowser=False)


if __name__ == "__main__":
    main()
