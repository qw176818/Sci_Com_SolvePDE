import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch import autograd
import matplotlib as mpl
import csv
import time
class Net(nn.Module):
    def __init__(self, NN):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(1, NN)
        self.h1_layer = nn.Linear(NN, NN)
        self.h2_layer = nn.Linear(NN, NN)
        self.h3_layer = nn.Linear(NN, NN)
        self.h4_layer = nn.Linear(NN, NN)
        self.output_layer = nn.Linear(NN, 1)
    def forward(self, x):
        out1 = torch.tanh(self.input_layer(x))
        out2 = torch.tanh(self.h1_layer(out1))
        out3 = torch.tanh(self.h2_layer(out2))
        out4 = torch.tanh(self.h3_layer(out3))
        out5 = torch.tanh(self.h4_layer(out4))
        out_final = self.output_layer(out5)
        return out_final
# 定义的神经网络
net = Net(90)
# 定义的损失函数
mse_loss_function = torch.nn.MSELoss(reduction='mean')
# 定义的优化器
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4)
def PDE(x, net):
    y = net(x)
    y_x = autograd.grad(y, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    # y_xx = autograd.grad(y_x, x, torch.22ones_like(x), create_graph = True, retain_graph = True)[0]
    tmp = 2 * torch.exp(-2 * x)
    y1_x = y_x + tmp
    return y1_x
iteration = 30000
start_time = time.time()
for epoch in range(iteration + 1):
    optimizer.zero_grad()
    x0 = torch.zeros(100, 1) - 2.0
    y0 = net(x0)
    y0_r = 4.0 * torch.ones(100, 1)
    mse_b_1 = mse_loss_function(y0, torch.exp(y0_r))
    x1 = torch.zeros(100, 1) + 2.0
    y1 = net(x1)
    y1_r = -4.0 * torch.ones(100, 1)
    mse_b_2 = mse_loss_function(y1, torch.exp(y1_r))
    x_in = np.random.uniform(low = -2.0, high = 2.0, size = (3000, 1))
    ca_in = autograd.Variable(torch.from_numpy(x_in).float(), requires_grad = True)
    ca_out = PDE(ca_in, net)
    ca_all_zeros = autograd.Variable(torch.from_numpy(np.zeros((3000, 1))).float(), requires_grad = False)
    mse_f = mse_loss_function(ca_out, ca_all_zeros)
    loss = (mse_b_1 + mse_b_2 + mse_f)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(epoch, "Traning Loss:", loss.data)
        print(f'times {epoch}  -  loss: {loss.item()}')
eclapse = time.time() - start_time
x_star = torch.linspace(-2, 2, 3000)
x_star = torch.reshape(x_star, (3000, 1))
x_star = torch.tensor(x_star, dtype=torch.float32)
u_pred = net(x_star)
u_star = np.exp(-2 * x_star)
error_u = np.linalg.norm(u_star - u_pred.detach().numpy(), 2) / np.linalg.norm(u_star, 2)
print(error_u, eclapse / 60, eclapse % 60)