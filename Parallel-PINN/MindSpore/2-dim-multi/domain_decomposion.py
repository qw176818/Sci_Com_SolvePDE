import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
fig = plt.figure(figsize=(9.5, 5))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']*100
a1 = [[-8, -4], [-4.5, 0.5], [-0.5, 4.5], [4, 8]]
a2 = [[-8, -4.5], [-4.5, -4], [-4, -0.5], [-0.5, 0.5], [0.5, 4], [4, 4.5], [4.5, 8]]
for i, d in enumerate(a1):
    if i % 2 == 0:
        plt.hlines(0.55, d[0], d[1], linewidth=6, colors=colors[i])
    else:
        plt.hlines(0.45, d[0], d[1], linewidth=6, colors=colors[i])
for i, d in enumerate(a2):
    if i % 2 == 0:
        plt.hlines(-0.5, d[0], d[1], linewidth=6, colors='tab:gray')
    else:
        plt.hlines(-0.5, d[0], d[1], linewidth=6, colors='tab:pink')
plt.xlabel("$x$")
plt.yticks([-1, -0.5, 0, 0.5, 1], [-1, "Overlapping\nDomain", 0, "Domain\nDecomposition", 1])
plt.title("DDPINN Domain Decomposition")

plt.savefig("2-multy-plot.pdf")
plt.savefig("2-multy-plot.eps")

