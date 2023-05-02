import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch import autograd
import matplotlib as mpl
x_in = torch.from_numpy(np.linspace(-4, 4, 201))
x_in = torch.reshape(x_in, (201, 1))
x_in = torch.tensor(x_in, dtype=torch.float32)
y_out = torch.zeros((201, 1))
for i, d in enumerate(x_in):
    if d < 0:
        y_out[i] = 1
            # 0.3 * torch.sin(d) + 0.1 * torch.sin(3 * d)
    else:
        y_out[i] = 1 + 0.1 * d * torch.cos(10 * d)
print(y_out)