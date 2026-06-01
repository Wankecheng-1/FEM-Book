import numpy as np
import matplotlib.pyplot as plt

# 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------------------------------
# 1. 正 n 边形逼近 π
# ------------------------------------------------------
def compute_pi(n):
    return n * np.sin(np.pi / n)

# ------------------------------------------------------
# 2. 计算数据
# ------------------------------------------------------
n_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
pi_true = np.pi
pi_approx = []
error = []

for n in n_list:
    pi_a = compute_pi(n)
    pi_approx.append(pi_a)
    err = abs(pi_true - pi_a)
    error.append(err)

# ------------------------------------------------------
# 3. 画图：只画逼近误差 + 斜率2 + 斜率3
# ------------------------------------------------------
plt.figure(figsize=(10, 6))

# 只取 n≥4 画图
h = 1.0 / np.array(n_list[2:])
err_plot = error[2:]

# ---------------------
# 1. 实际误差曲线
# ---------------------
plt.loglog(h, err_plot, 'b-o', linewidth=2, markersize=8, label='实际逼近误差')

# ---------------------
# 2. 斜率 = 2 参考线（理论收敛阶）
# ---------------------
C2 = err_plot[0] / (h[0] ** 2)
ref2 = C2 * h ** 2
plt.loglog(h, ref2, 'r--', linewidth=2, label='斜率 = 2（二阶收敛）')

# ---------------------
# 3. 斜率 = 3 参考线（老师要求加的）
# ---------------------
C3 = err_plot[0] / (h[0] ** 3)
ref3 = C3 * h ** 3
plt.loglog(h, ref3, 'g-.', linewidth=2, label='斜率 = 3（三阶收敛）')

# 图表样式
plt.xlabel('步长 h = 1/n', fontsize=12)
plt.ylabel('误差 |π - πₙ|', fontsize=12)
plt.title('正多边形逼近圆周率收敛图', fontsize=14)
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.show()