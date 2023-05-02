import math
import sys
sys.path.insert(0, '../../Utilities/')
import numpy as np
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from plotting import newfig, savefig

from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__ == '__main__':
    data = scipy.io.loadmat('AC.mat')
    t = data['tt'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    # t = t.T[0]
    # x = x.T[0]
    # t = np.outer(t, np.ones(512))
    # x = np.outer(x, np.ones(201)).T
    X, T = np.meshgrid(x ,t)
    Exact = np.real(data['uu']).T
    Exact1 = Exact.copy()
    for i in range(50, 201):
        for j in range(0, 512):
            if Exact1[i][j] > 0:
                Exact1[i][j] = Exact[i][j] * 2
    fig, ax = newfig(2.0, 0.9)
    ax.axis('off')
    gs1 = gridspec.GridSpec(1, 2)
    ax = plt.subplot(gs1[0, 0], projection = '3d')
    # ax = fig.add_subplot(2, 3, 1, projection='3d')
    ax.plot_surface(X, T, Exact, cmap='viridis', edgecolor='none')
    ax.set_title('Allen-Cahn-AS')
    ax = plt.subplot(gs1[0, 1], projection = '3d')
    # ax1 = fig.add_subplot(2, 3, 2, projection='3d')
    ax.plot_surface(X, T, Exact1, cmap='viridis', edgecolor='none')
    ax.set_title('Allen-Cahn-PS')
    savefig("./figures/img_tmp")