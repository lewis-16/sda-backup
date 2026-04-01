下面提供一个详细的 Python 绘图脚本，用于生成闭环 kirigami 图案的二维图纸。该脚本基于文章中的参数化模型，允许你自由调整几何参数，并直接输出图案。

---

### 参数说明

闭环 kirigami 图案由以下 8 个独立参数定义：

| 参数        | 含义                                     | 示例值 (设计 B)            |
| ----------- | ---------------------------------------- | -------------------------- |
| `r_o`     | 圆盘外半径                               | 70 mm                      |
| `r_i`     | 中心孔半径                               | 3 mm                       |
| `t`       | 材料厚度（绘图不需）                     | 69 μm                     |
| `Δr1`    | 第一个切缝距中心孔的距离                 | 3 mm                       |
| `Δr2`    | 第二个切缝与第一个切缝的间距             | 2 mm                       |
| `n`       | 径向分布指数（≥1）                      | 1.0                        |
| `Nθ`     | 角向扇区数量                             | 5                          |
| `θ`      | 角度比 = θ_a / θ_i                     | 0.3                        |
| `Δr_min` | 最小材料宽度（制造约束）                 | 1 mm（可根据切割精度调整） |
| `offset`  | 相邻圈切缝的角度偏移（可选，0 表示对齐） | 0                          |

切缝的径向位置计算采用递归公式：

\[
\Delta r_{j+1} = r_o\left[\left(1 + \frac{\Delta r_j}{r_o}\right)^n - 1\right], \quad j \ge 2
\]

角向切缝角度宽度：
\[
\theta_i = \frac{2\pi}{N_\theta (1 + \theta)}, \quad \theta_a = \theta \cdot \theta_i
\]

---

### Python 绘图代码

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

def generate_radial_positions(r_i, r_o, dr1, dr2, n, dr_min):
    """
    计算所有径向切缝的位置（从内到外）
    返回列表 positions，每个元素为半径 r_j
    """
    positions = []
    r = r_i + dr1                     # 第一个切缝
    if r >= r_o:
        return positions
    positions.append(r)
  
    r_next = r + dr2                   # 第二个切缝
    if r_next >= r_o:
        return positions
    positions.append(r_next)
  
    dr_prev = dr2
    r_curr = r_next
    while True:
        dr_next = r_o * ((1 + dr_prev / r_o) ** n - 1)
        if dr_next < dr_min:
            dr_next = dr_min           # 强制满足最小间距
        r_next = r_curr + dr_next
        if r_next >= r_o - dr_min:     # 剩余材料小于最小宽度则停止
            break
        positions.append(r_next)
        dr_prev = dr_next
        r_curr = r_next
    return positions

def draw_kirigami(ax, r_i, r_o, positions, N_theta, theta, offset=0, color='black', lw=0.5):
    """
    在 ax 上绘制 kirigami 图案
    positions: 径向切缝位置列表
    theta: 角度比 θ
    offset: 每圈的角度偏移（弧度），默认为0
    """
    theta_i = 2 * np.pi / (N_theta * (1 + theta))   # 单个切缝角度
    theta_a = theta * theta_i                       # 间隔角度
  
    for j, r in enumerate(positions):
        # 可选：每圈增加偏移量，使相邻圈错开
        phase = offset * j
        for k in range(N_theta):
            start_angle = k * (theta_i + theta_a) + phase
            end_angle = start_angle + theta_i
            # 绘制圆弧（角度从 start 到 end，以度为单位）
            arc = Arc((0, 0), 2*r, 2*r, angle=0, 
                      theta1=np.degrees(start_angle), 
                      theta2=np.degrees(end_angle),
                      color=color, lw=lw)
            ax.add_patch(arc)

# 设置参数（以设计 B 为例）
r_o = 70.0          # mm
r_i = 3.0
dr1 = 3.0
dr2 = 2.0
n = 1.0
N_theta = 5
theta_ratio = 0.3
dr_min = 1.0        # 最小材料宽度
offset = 0.0        # 角度偏移（弧度）

# 生成径向位置
positions = generate_radial_positions(r_i, r_o, dr1, dr2, n, dr_min)
print("径向切缝位置 (mm):", [round(p,2) for p in positions])

# 绘图
fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal')
ax.set_xlim(-r_o-5, r_o+5)
ax.set_ylim(-r_o-5, r_o+5)
ax.axis('off')

# 绘制外圆和内圆
outer_circle = plt.Circle((0,0), r_o, fill=False, edgecolor='black', lw=1)
inner_circle = plt.Circle((0,0), r_i, fill=False, edgecolor='black', lw=1)
ax.add_patch(outer_circle)
ax.add_patch(inner_circle)

# 绘制所有切缝
draw_kirigami(ax, r_i, r_o, positions, N_theta, theta_ratio, offset, color='red', lw=1)

plt.title(f"Kirigami Pattern (Nθ={N_theta}, θ={theta_ratio}, n={n})")
plt.show()
```

---

### 运行结果示例

使用上述设计 B 的参数，你将得到一个类似下图的图案（红色圆弧表示切缝）：

![Kirigami Pattern](https://i.imgur.com/placeholder.png)
（由于无法直接显示图片，请运行代码查看实际输出）

图案包含：

- 黑色外圆和内圆
- 若干圈红色圆弧，每圈均匀分布 5 个切缝，切缝角度由 `θ` 控制
- 径向位置由递归公式生成，确保靠近边缘的切缝间距逐渐增大（如果 `n>1`）

---

### 自定义设计

你可以通过修改参数快速尝试不同设计：

| 参数             | 效果                                                      |
| ---------------- | --------------------------------------------------------- |
| `N_theta`      | 增加则切缝更密集，形成更多“叶片”                        |
| `theta_ratio`  | 增大则切缝变宽，间隔变小                                  |
| `n`            | 控制切缝径向分布：`n=1` 等距，`n>1` 外圈间距逐渐增大  |
| `dr1`, `dr2` | 调整初始切缝位置                                          |
| `offset`       | 设置相邻圈的角度错位（例如 `np.pi/N_theta` 使切缝交错） |

例如，设计 A 的参数为 `N_theta=8, dr1=3, dr2=2, n=1, theta=0.3`，可以修改后运行。

---

### 注意事项

1. **切缝宽度**：实际切割中切缝有一定宽度，代码中用细线表示位置，如需模拟宽度可绘制填充扇形。
2. **制造约束**：代码在径向生成时加入了 `dr_min` 检查，避免材料过窄。
3. **角度偏移**：文章未明确相邻圈是否对齐，你可以通过 `offset` 实现交错设计，观察对稳定性的影响。

---

这个脚本为你提供了一个完全参数化的绘图工具，可以快速生成闭环 kirigami 图案，用于设计分析或激光切割文件输出。如果需要保存为 SVG 或 DXF 矢量图，可用 `plt.savefig()` 导出。
