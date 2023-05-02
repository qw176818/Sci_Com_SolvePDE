import numpy as np
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
if __name__ == '__main__':
    data = scipy.io.loadmat('burgers_shock.mat')
    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]

    t = t.T[0]
    x = x.T[0]
    t = np.outer(t, np.ones(256))
    x = np.outer(x, np.ones(100)).T

    Exact = np.real(data['usol']).T
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, t, Exact, cmap='viridis', edgecolor='none')
    ax.set_title('Burgers-AS')
    plt.show()