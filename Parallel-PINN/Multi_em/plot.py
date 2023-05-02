import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties

x = np.linspace(-7, 7, 200)
w1 = 1
w2 = 7
y = np.sin(w1 * x) + np.sin(w2 * x)

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

fig = plt.figure(figsize = (10.0, 5.0))
plt.plot(x, y, color='orange', label='Exact')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(-2.2, 2.2)
plt.xlim(-7.0, 7.0)
plt.legend(fontsize = 13)
plt.show()