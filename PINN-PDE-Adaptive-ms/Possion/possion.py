import numpy as np
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


if __name__ == '__main__':
    y = np.linspace(0, 1, 100)
    x = np.linspace(0, 1, 100)
    y = np.outer(y, np.ones(100))
    x = np.outer(x, np.ones(100)).T
    Exact = x * (x - 1) * np.sin(2 * np.pi * y) + 1
    # xx * (xx - 1) * np.sin(2 * np.pi * yy) + 1
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, Exact, cmap='viridis', edgecolor='none')
    ax.set_title('Possion-AS')
    plt.show()