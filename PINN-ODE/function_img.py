import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 100)
y = np.ones(x.shape)
for i, d in enumerate(x):
    if d < 0:
        y[i] = 0
    else:
        y[i] = d
# y = 1 / (1 + np.exp(-x))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False   # 这两行需要手动设置
# plt.gcf().set_facecolor(np.ones(3)* 240 / 255)   # 生成画布的大小
plt.grid()  # 生成网格
plt.plot(x, y, color='red', linewidth = 0.8)

plt.xlim(-10, 10)
# plt.ylim(0, 3000)

plt.title("Relu函数",fontsize=10,loc='center',color='black')
ax = plt.gca()

plt.savefig("./img/relu.eps")
