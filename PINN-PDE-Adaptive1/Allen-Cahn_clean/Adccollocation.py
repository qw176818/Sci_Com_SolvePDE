# import numpy as np
# import matplotlib.pyplot as plt
#
# # 设置正方形区域的边长为2
# square_size = 5
#
# # 生成随机采样点的坐标
# x = np.random.uniform(low=0, high=5, size=500)
# y = np.random.uniform(low=0, high=5, size=500)
#
# # 将采样点限制在正方形区域内
# x = np.clip(x, 0, 5)
# y = np.clip(y, 0, 5)
#
# # 创建绘图对象
# fig, ax = plt.subplots()
#
# # 绘制正方形区域
# square = plt.Rectangle((0, 0), square_size, square_size, fill=False)
# ax.add_patch(square)
#
# # 绘制随机采样点
# ax.scatter(x, y, s=1)
# ax.tick_params(which='major', length=0)
# # 显示图形
# plt.show()
import numpy as np
import matplotlib.pyplot as plt

# 定义正方形子网格的大小
n = 5

# 创建一个正方形子网格
fig = plt.figure(figsize = (12, 4))
ax1, ax2, ax3 = fig.subplots(1, 3, sharex = 'col')


ax1.set_xticks(np.arange(0, n+1, 1))
ax1.set_yticks(np.arange(0, n+1, 1))
# ax.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
ax1.tick_params(which='major', length=0)
ax2.set_xticks(np.arange(0, n+1, 1))
ax2.set_yticks(np.arange(0, n+1, 1))
ax2.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
ax2.tick_params(which='major', length=0)

ax3.set_xticks(np.arange(0, n+1, 1))
ax3.set_yticks(np.arange(0, n+1, 1))
# ax3.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
ax3.tick_params(which='major', length=0)
# 在随机位置绘制蓝色圆圈
sum = []
for i in range(50):
    x = np.random.randint(0, n)
    y = np.random.randint(0, n)
    for k in range(1):
        xx = x + np.random.rand()
        yy = y + np.random.rand()
        sum.append([xx, yy])
        ax1.add_patch(plt.Circle((xx, yy), 0.01, color='red'))
        ax3.add_patch(plt.Circle((xx, yy), 0.01, color='red'))

# 在随机位置绘制蓝色圆圈
for i in range(5):
    x = np.random.randint(0, n)
    y = np.random.randint(0, n)
    for k in range(20):
        xx = x + np.random.rand()
        yy = y + np.random.rand()
        ax2.add_patch(plt.Circle((xx, yy), 0.01, color='blue'))
        ax3.add_patch(plt.Circle((xx, yy), 0.01, color='blue'))
plt.savefig("adcpinn.pdf")
plt.savefig("adcpinn.eps")