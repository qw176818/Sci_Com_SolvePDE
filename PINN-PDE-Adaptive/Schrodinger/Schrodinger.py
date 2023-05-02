import numpy as np
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
if __name__ == '__main__':
    data = scipy.io.loadmat('NLS.mat')
    t = data['tt'].flatten()[:, None]
    x = data['x'].flatten()[:, None]

    t = t.T[0]
    x = x.T[0]
    t = np.outer(t, np.ones(256))
    x = np.outer(x, np.ones(201)).T

    Exact = np.real(data['uu']).T
    Exact1 = np.imag(data['uu']).T
    Exact2 = np.sqrt(Exact ** 2 + Exact1 ** 2)
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.plot_surface(x, t, Exact, cmap='viridis', edgecolor='none')
    ax.set_title('Schrodinger-AS-u')
    bx = fig.add_subplot(1, 3, 2, projection='3d')
    bx.plot_surface(x, t, Exact1, cmap='viridis', edgecolor='none')
    bx.set_title('Schrodinger-AS-v')
    cx = fig.add_subplot(1, 3, 3, projection='3d')
    cx.plot_surface(x, t, Exact2, cmap='viridis', edgecolor='none')
    cx.set_title('Schrodinger-AS-h')
    plt.show()