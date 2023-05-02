import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False # 解决负号无法正常显示的问题
# 定义心形函数
t = np.linspace(0, 2*np.pi, 1000)
x = 16 * np.sin(t)**3
y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
text = "晚安，热爱生活、早睡早起的娃最有魅力!"
plt.text(-10.0, -1.5, text, fontsize=12, color='')
# 绘制心形
plt.plot(x, y, color='red', linewidth=2)

# 显示图形
plt.show()

