"""Generate report figures from committed experiment data without matplotlib."""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_psh import run_episode
from grid_env import create_dynamic_env
from psh_planner import PredictiveSafeHybridPlanner

OUT = ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)


def font(size, bold=False):
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


NAVY = "#16324F"
BLUE = "#2F6B9A"
TEAL = "#2A9D8F"
RED = "#C94C4C"
INK = "#263238"
GRID = "#D9E1E8"
BG = "#F7F9FB"


def rounded_box(draw, xy, fill, outline=GRID, radius=12, width=2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def architecture():
    image = Image.new("RGB", (1600, 760), "white")
    d = ImageDraw.Draw(image)
    d.text((70, 45), "PSH 预测式安全混合规划架构", fill=INK, font=font(42, True))
    d.text((72, 105), "全局路线负责方向，局部预测负责动态避障，安全盾拥有最终控制权", fill="#607080", font=font(23))
    boxes = [
        ((70, 245, 310, 405), "局部观测", "障碍位置\n机器人状态", BLUE),
        ((385, 190, 675, 350), "运动跟踪与预测", "速度估计\nH 步概率占据", TEAL),
        ((385, 430, 675, 590), "D* Lite 全局走廊", "静态地图\n增量更新", NAVY),
        ((760, 245, 1070, 475), "滚动时域局部规划", "Beam Search\n进展 距离 风险\n转向 访问惩罚", BLUE),
        ((1155, 245, 1415, 475), "安全盾", "风险阈值\n动作否决与回退", RED),
    ]
    for (x1, y1, x2, y2), title, body, color in boxes:
        rounded_box(d, (x1, y1, x2, y2), BG, color, 12, 3)
        d.text((x1 + 25, y1 + 24), title, fill=color, font=font(27, True))
        for i, line in enumerate(body.splitlines()):
            d.text((x1 + 25, y1 + 75 + 34 * i), line, fill=INK, font=font(22))
    arrows = [((310, 325), (385, 270)), ((675, 270), (760, 320)), ((675, 510), (760, 405)), ((1070, 360), (1155, 360)), ((1415, 360), (1530, 360))]
    for start, end in arrows:
        d.line((start, end), fill=INK, width=5)
        ex, ey = end
        d.polygon([(ex, ey), (ex - 18, ey - 10), (ex - 18, ey + 10)], fill=INK)
    d.text((1450, 315), "动作", fill=INK, font=font(23, True))
    d.text((75, 675), "可选学习接口  DQN 仅提供动作先验，不绕过局部优化和安全约束", fill="#607080", font=font(22))
    image.save(OUT / "architecture.png")


def benchmark_chart():
    summary = json.loads((ROOT / "results" / "psh_benchmark" / "summary.json").read_text(encoding="utf-8"))
    image = Image.new("RGB", (1500, 860), "white")
    d = ImageDraw.Draw(image)
    d.text((70, 35), "30 个未见种子上的成功率与碰撞率", fill=INK, font=font(40, True))
    d.text((72, 92), "相同局部观测 半径 5  相同动态轨迹  每回合最多 350 步", fill="#607080", font=font(22))
    left, top, width, height = 260, 170, 1120, 550
    for tick in range(0, 101, 20):
        y = top + height - tick / 100 * height
        d.line((left, y, left + width, y), fill=GRID, width=2)
        d.text((180, y - 13), f"{tick}%", fill="#607080", font=font(20))
    labels = [s["algorithm"].replace("PSH - prediction", "PSH 无预测").replace("PSH - shield", "PSH 无安全盾").replace("PSH (full)", "PSH 完整") for s in summary]
    group = width / len(summary)
    for i, item in enumerate(summary):
        cx = left + group * (i + 0.5)
        success = item["success_rate"]
        collision = item["collision_rate"]
        sy = top + height - success * height
        cy = top + height - collision * height
        d.rectangle((cx - 64, sy, cx - 9, top + height), fill=TEAL)
        d.rectangle((cx + 9, cy, cx + 64, top + height), fill=RED)
        d.text((cx - 66, sy - 32), f"{success*100:.1f}", fill=TEAL, font=font(19, True))
        d.text((cx + 7, cy - 32), f"{collision*100:.1f}", fill=RED, font=font(19, True))
        bbox = d.textbbox((0, 0), labels[i], font=font(18))
        d.text((cx - (bbox[2]-bbox[0])/2, top + height + 26), labels[i], fill=INK, font=font(18))
    d.rectangle((1020, 100, 1045, 125), fill=TEAL)
    d.text((1055, 98), "成功率", fill=INK, font=font(20))
    d.rectangle((1165, 100, 1190, 125), fill=RED)
    d.text((1200, 98), "碰撞率", fill=INK, font=font(20))
    image.save(OUT / "benchmark.png")


def scenario_trace():
    env = create_dynamic_env(seed=24, rows=24, cols=32, obstacle_count=34, num_moving=4, obstacle_patterns=["horizontal", "vertical", "random", "horizontal"])
    result = run_episode(env, PredictiveSafeHybridPlanner(), trace=True)
    trace = result["trace"]
    scale, margin = 24, 45
    image = Image.new("RGB", (32 * scale + margin * 2, 24 * scale + margin * 2), "white")
    d = ImageDraw.Draw(image)
    for r in range(24):
        for c in range(32):
            x1, y1 = margin + c * scale, margin + r * scale
            fill = "#263238" if env._static_grid[r, c] == 1 else ("#EAF2F8" if env._static_grid[r, c] == 2 else "#FFFFFF")
            d.rectangle((x1, y1, x1 + scale, y1 + scale), fill=fill, outline="#E2E7EB")
    points = [(margin + c * scale + scale/2, margin + r * scale + scale/2) for r, c in trace["robot"]]
    if len(points) > 1:
        d.line(points, fill=TEAL, width=6, joint="curve")
    for r, c in trace["obstacles"][0]:
        x, y = margin + c * scale + scale/2, margin + r * scale + scale/2
        d.ellipse((x-7, y-7, x+7, y+7), fill=RED)
    sr, sc = env.start
    gr, gc = env.goal
    d.ellipse((margin+sc*scale+4, margin+sr*scale+4, margin+(sc+1)*scale-4, margin+(sr+1)*scale-4), fill=BLUE)
    d.rectangle((margin+gc*scale+5, margin+gr*scale+5, margin+(gc+1)*scale-5, margin+(gr+1)*scale-5), fill="#F4B942")
    d.text((margin, 8), f"Seed 24  PSH 成功  {result['steps']} 步  最小间距 {result['min_clearance']:.1f} 格", fill=INK, font=font(21, True))
    image.save(OUT / "scenario_trace.png")


if __name__ == "__main__":
    architecture()
    benchmark_chart()
    scenario_trace()
