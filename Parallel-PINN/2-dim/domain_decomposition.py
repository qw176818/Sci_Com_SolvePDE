import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
fig = plt.figure(figsize=(9.5, 5))
colors = ['tab:blue', 'tab:orange', 'tab:olive', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']*100
# a1 = [[-6, -2.5], [-3, 3], [2.5, 6]]
a2 = [[-6, -3], [-3, -2.5], [-2.5, 2.5], [2.5, 3], [3, 6]]
# for i, d in enumerate(a1):
#     if i % 2 == 0:
#         plt.hlines(0.55, d[0], d[1], linewidth=6, colors=colors[i])
#     else:
#         plt.hlines(0.45, d[0], d[1], linewidth=6, colors=colors[i])
# for i, d in enumerate(a2):
#     if i % 2 == 0:
#         plt.hlines(-0.5, d[0], d[1], linewidth=6, colors='tab:gray')
#     else:
#         plt.hlines(-0.5, d[0], d[1], linewidth=6, colors='tab:pink')
# plt.xlabel("$x$")
# plt.yticks([-1, -0.5, 0, 0.5, 1], [-1, "Overlapping\nDomain", 0, "Domain\nDecomposition", 1])
# plt.title("DDPINN Domain Decomposition")
# # plt.show()
# plt.savefig("2-sin-plot.pdf")
# plt.savefig("2-sin-plot.eps")

w = 12
x = np.linspace(-6, 6, 9000)
y_ext = -1 / w / w * np.sin(w * x)
y_xx_ext = np.sin(w * x)
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
fig = plt.figure(figsize = (10.0, 5.0))
plt.plot(x, y_ext, color='green', label='Exact')
# plt.plot(x, y_xx_ext, color='green', label='Exact')
for i, d in enumerate(a2):
    if i % 2 == 0:
        x1 = np.linspace(d[0], d[1], 3000)
        y1 = -1 / w / w * np.sin(w * x1)
        plt.plot(x1, y1, colors[int(i / 2)], label='Sub-d'+str(int(i / 2) + 1))
    else:
        x1 = np.linspace(d[0], d[1], 1000)
        y1 = -1 / w / w * np.sin(w * x1)
        plt.plot(x1, y1, "pink", label='Cross-R'+str(int(i / 2) + 1))
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-0.2, 0.2)
plt.xlim(-6.0, 10.0)
plt.title(r"DDPINN w = 12")
# plt.title(r"DDPINN w = 12, $\frac{d^2u}{dx^2}$")
plt.legend(fontsize = 13)
# plt.show()
plt.savefig("ddpinn-2-sin-plot.pdf")
plt.savefig("ddpinn-2-sin-plot.eps")