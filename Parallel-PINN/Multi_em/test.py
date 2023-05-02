import matplotlib.pyplot as plt

# 创建一个示例图像
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])

# 添加标签
ax.set_xlabel('X轴标签')
ax.set_ylabel('Y轴标签')

# 调整图像周围的空白区域，使标签脱离图像
plt.subplots_adjust(left=0.3, right=0.7, top=0.7, bottom=0.3)

# 显示图像
plt.show()