import numpy as np
import matplotlib.pyplot as plt

# 定义正方形子网格的大小
n = 5

# 创建一个正方形子网格
fig, ax = plt.subplots()
ax.set_xticks(np.arange(0, n+1, 1))
ax.set_yticks(np.arange(0, n+1, 1))
ax.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
ax.tick_params(which='major', length=0)

# 在随机位置绘制蓝色圆圈
for i in range(5):
    x = np.random.randint(0, n)
    y = np.random.randint(0, n)
    for k in range(20):
        xx = x + np.random.rand()
        yy = y + np.random.rand()
        ax.add_patch(plt.Circle((xx, yy), 0.01, color='red'))
plt.show()