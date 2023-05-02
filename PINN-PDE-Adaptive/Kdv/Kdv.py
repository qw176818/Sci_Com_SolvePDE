import numpy as np
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math
def f(x, t):
    c = 3
    x0 = 0
    u = (math.sqrt(c) / 2) * (x - x0 - c * t)
    u1 = 2.0 / (np.exp(u) + np.exp(-u))
    u2 = (c / 2) * u1 * u1
    return u2
if __name__ == '__main__':

    t = np.linspace(0, 5, 500)
    x = np.linspace(-20, 20, 1000)
    t = np.outer(t, np.ones(1000))
    x = np.outer(x, np.ones(500)).T
    Exact = f(x, t)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x = x.T
    t = t.T
    Exact = Exact.T
    ax.plot_surface(x, t, Exact, cmap='rainbow', edgecolor= 'none')
    ax.set_title('Kdv-AS')
    plt.show()