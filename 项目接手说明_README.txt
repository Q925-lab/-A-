人工智能基础大作业项目接手说明

项目名称：
基于 Q-learning 与 A* 的移动机器人多约束路径规划仿真系统


一、项目用途说明

本项目用于《人工智能基础》课程大作业，主要实现二维栅格地图中的移动机器人路径规划仿真。

项目中实现了两类路径规划方法：

1. A* 算法
   作为传统启发式路径规划方法，用于在已知地图中寻找路径。

2. Q-learning 算法
   作为强化学习路径规划方法，通过智能体与环境交互学习路径策略。

最终通过多组随机地图实验，对比两种方法在路径长度、路径代价、成功率、危险区域规避效果等方面的表现。


二、项目运行环境

建议使用以下 Python 环境：

Python 3.8 及以上版本

项目主要依赖：

numpy
matplotlib
pandas

如果运行 GUI 版本，还需要：

tkinter

安装依赖命令：

pip install numpy matplotlib pandas

如果本机已经安装 Anaconda，也可以直接在 Anaconda Prompt 或 VS Code 终端中运行。


三、项目文件说明

项目主要文件如下：

main.py
主程序入口，用于运行一次路径规划演示。

train.py
训练脚本，用于批量运行多个 seed 实验，是正式实验中最重要的运行文件。

config.py
项目主要参数配置文件，用于修改地图大小、障碍物比例、危险区域代价、Q-learning 参数等。

grid_env.py
栅格地图环境，包括障碍物、危险区、起点终点等。

astar.py
A* 路径规划算法实现。

q_learning.py
Q-learning 算法实现。

visualize.py
可视化模块，用于绘制地图、路径和训练结果。

metrics_utils.py
指标统计与结果保存工具。

gui_app.py
GUI 图形界面版本，如不使用 GUI 可忽略。

comparison.csv
实验结果对比表。

experiment_config.json
本次实验的参数记录。

q_table.npy
Q-learning 训练得到的 Q 表。

README.md / 项目接手说明.txt
项目说明文件。


四、如何运行项目

1. 运行单次演示

如果只是想快速查看当前地图和算法效果，可以运行：

python main.py

该文件通常用于展示当前参数下的 A* 和 Q-learning 路径规划结果。


2. 运行多 seed 实验

如果要进行正式实验，建议运行：

python train.py

train.py 会按照设定的随机种子列表，批量生成实验结果，并输出：

comparison.csv
experiment_config.json
q_table.npy

正式写报告或做 PPT 时，建议主要使用 train.py 生成的数据。


五、参数在哪里修改

项目的主要参数通常在以下两个文件中修改：

config.py
train.py

其中：

config.py：
主要修改地图、奖励、代价、算法参数。

train.py：
主要修改实验次数、随机种子列表、是否批量训练。


六、常用参数说明

1. 地图相关参数

以下参数一般在 config.py 中修改。

GRID_SIZE
表示地图大小。
例如 GRID_SIZE = 30 表示 30 × 30 的栅格地图。

START_POS
表示机器人起点坐标。

GOAL_POS
表示机器人终点坐标。

例如：

START_POS = (1, 1)
GOAL_POS = (28, 28)

OBSTACLE_RATIO
表示障碍物比例。

该值越大，地图越复杂。
如果设置过大，地图可能会接近迷宫，甚至导致起点和终点不可达。

DANGER_COST
表示危险区域的额外代价。

当前项目中曾使用：

DANGER_COST = 1.0

该值越大，算法越倾向于避开危险区域。


2. Q-learning 相关参数

以下参数通常在 config.py 或 q_learning.py 中修改。

ALPHA
学习率，控制 Q 表更新速度。

GAMMA
折扣因子，控制智能体对未来奖励的重视程度。

EPSILON
探索率，控制智能体随机探索的概率。

EPISODES
训练轮数。

训练轮数越大，Q-learning 越有机会学习到稳定路径，但运行时间也会增加。


七、seed 的使用说明

1. seed 是什么

seed 是随机种子，用于控制随机过程。

本项目中 seed 主要影响：

1）随机地图的生成；
2）随机障碍物分布；
3）Q-learning 训练中的随机探索过程。

使用相同 seed 时，实验结果具有较强的可复现性。

也就是说：

相同参数 + 相同 seed ≈ 可以复现相同实验结果


2. 当前使用的 seed

当前多组实验中使用过的 seed 为：

EXPERIMENT_SEEDS = [7, 11, 21]

这行代码通常位于 train.py 中。


3. 如何修改 seed

如果只想测试一个地图，可以写成：

EXPERIMENT_SEEDS = [7]

如果想一次性跑多组实验，可以增加 seed 数量，例如：

EXPERIMENT_SEEDS = [7, 11, 21, 42, 66, 88]

这样程序会依次使用这些 seed 生成地图并训练，最后将结果汇总到 comparison.csv 中。


4. 为什么要用多个 seed

单个 seed 只能代表一张随机地图，结果可能具有偶然性。

例如某个 seed 下：

地图刚好很简单；
障碍物分布刚好很难；
起点到终点路径刚好被迫绕远；
Q-learning 刚好训练不稳定。

因此，正式实验中不能只看一个 seed 的结果。

更合理的做法是：

使用多个 seed 跑多组实验，再观察平均表现和整体趋势。

这样可以减少偶然性，使结论更可靠。


八、输出文件说明

运行 train.py 后，通常会生成以下文件。


1. comparison.csv

该文件保存实验对比结果。

一般包括：

seed
算法名称
路径长度
路径总代价
是否成功到达目标点
危险区域经过情况
训练奖励

具体列名以当前代码输出为准。

这个文件是后续写报告和做图表最重要的数据来源。


2. experiment_config.json

该文件保存本次实验使用的参数。

作用是方便之后复现实验。

如果之后结果和之前不同，可以先检查：

1）seed 是否相同；
2）地图参数是否相同；
3）Q-learning 参数是否相同；
4）危险区域代价是否相同；
5）障碍物生成方式是否相同。


3. q_table.npy

该文件保存 Q-learning 训练得到的 Q 表。

注意：

如果当前程序只保存一个 q_table.npy，那么多 seed 实验时，该文件可能会被最后一次训练结果覆盖。

如果需要保存每个 seed 的 Q 表，建议改成：

q_table_seed7.npy
q_table_seed11.npy
q_table_seed21.npy

这样更方便之后检查每一组实验的训练结果。


九、推荐实验流程

建议组员按照以下顺序接手项目。


第一步：确认环境

先运行：

python --version

确认 Python 版本。

再安装依赖：

pip install numpy matplotlib pandas


第二步：运行单次演示

运行：

python main.py

确认程序能正常生成地图和路径图。


第三步：运行多 seed 实验

运行：

python train.py

查看是否生成：

comparison.csv
experiment_config.json
q_table.npy


第四步：检查结果

重点查看：

comparison.csv

观察 A* 和 Q-learning 的差异。


第五步：修改 seed 或参数继续实验

如果需要扩大实验规模，可以修改：

EXPERIMENT_SEEDS = [7, 11, 21]

例如改为：

EXPERIMENT_SEEDS = [7, 11, 21, 42, 66, 88, 100]

然后重新运行：

python train.py


十、注意事项

1. 不建议随意改动地图生成逻辑

当前地图生成方式已经经过多次调整。

如果随意修改障碍物生成方式，可能导致：

地图变成纯迷宫；
障碍物过于集中；
起点和终点不连通；
A* 和 Q-learning 结果难以比较。

如果只是想增加实验数量，优先修改 seed，而不是修改地图生成逻辑。


2. 修改参数后要重新记录结果

每次修改重要参数后，建议保留对应的：

comparison.csv
experiment_config.json

否则之后可能不知道某张图或某个结果是由哪组参数生成的。

建议保存为类似形式：

results_seed_7_11_21/
results_danger_cost_1.0/
results_more_seeds/


3. Q-learning 结果可能不如 A*

这是正常现象。

A* 是在已知地图上的启发式搜索算法，目标明确，稳定性较强。

Q-learning 是通过试错学习策略，训练结果受以下因素影响较大：

训练轮数；
奖励函数；
探索率；
地图复杂度；
随机 seed。

因此，如果某些 seed 下 Q-learning 走得比较绕，或者奖励曲线不稳定，不一定说明代码错误，可能是强化学习训练本身不稳定。


十一、组员接手建议

如果只是想复现实验：

不要改代码逻辑，只改 EXPERIMENT_SEEDS，然后运行 train.py。

如果想调整地图难度：

优先修改 OBSTACLE_RATIO、DANGER_COST 等参数。

如果想调整 Q-learning 训练效果：

优先修改 EPISODES、ALPHA、GAMMA、EPSILON。

如果想保留不同实验结果：

每次实验后单独保存 comparison.csv 和 experiment_config.json。


十二、当前项目状态说明

当前项目已经完成：

栅格地图环境构建；
随机障碍物生成；
危险区域代价设置；
A* 路径规划；
Q-learning 训练；
多 seed 实验；
实验结果保存；
路径与训练过程可视化。

后续主要工作是：

1. 继续跑更多 seed；
2. 整理 comparison.csv 中的数据；
3. 选择有代表性的实验图；
4. 在报告和 PPT 中解释 A* 与 Q-learning 的差异；
5. 说明 Q-learning 在本项目中的局限性。
