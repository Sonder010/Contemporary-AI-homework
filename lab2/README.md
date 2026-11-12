# 实验二：搜索算法

## 文件

- `experiment2.py`：A*、加权A*、Minimax、Alpha-Beta以及宝藏博弈的实现。
- `test_experiment2.py`：题目给定样例和Minimax/Alpha-Beta一致性测试。
- `实验二搜索算法实验报告.docx`：实验报告。

## 运行环境

使用 Python 3.10 或更高版本。

## 运行

```powershell
python test_experiment2.py
```

测试通过时会输出：

```text
All supplied and cross-check tests passed.
```

## 说明

- A* 默认邻居顺序为“上、右、下、左”，相同长度路径的展开顺序是可复现的。
- `astar` 的 `neighbor_order` 参数支持“上右下左”“下左上右”“右左下上”等四方向排列，对应常量在 `NEIGHBOR_ORDERS` 中。
- 题目一（单位代价迷宫）每次移动代价为1。
- 题目二（加权地形）调用A*时传入`zero_goal_cost=True`，将终点E的进入代价设为0，其余按格子上的数字计价。
- 宝藏博弈的距离矩阵是提前算好的，搜索时直接查表，不用重复跑A*。

如需在其他脚本中调用，主要接口都在`experiment2.py`里，函数名和参数见源码注释即可。
