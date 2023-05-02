import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch import autograd

class Net(nn.Module):
    def __init__(self, NN):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(1, NN)
        self.h1_layer = nn.Linear(NN, NN)
        self.h2_layer = nn.Linear(NN, NN)
        self.h3_layer = nn.Linear(NN, NN)
        self.h4_layer = nn.Linear(NN, NN)
        # self.h5_layer = nn.Linear(NN, NN)
        # self.h6_layer = nn.Linear(NN, NN)
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
net = Net(60)
# 定义的损失函数
mse_loss_function = torch.nn.MSELoss(reduction='mean')
# 定义的优化器
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4)
w1 = 1
w2 = 7
def PDE(x, net):
    y = torch.tanh(w2 * (x + 2 * torch.pi)) * net(x)
    y_x = autograd.grad(y, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    y_xx = autograd.grad(y_x, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    tmp = w1 * torch.cos(w1 * x) + w2 * torch.cos(w2 * x)
    res = y_x - tmp
    return res
plt.ion()
iteration = 50000

for epoch in range(iteration + 1):
    optimizer.zero_grad()
    x_in = np.random.uniform(low = -8.0, high = -4.0, size = (600, 1))
    ca_in = autograd.Variable(torch.from_numpy(x_in).float(), requires_grad = True)
    ca_out = PDE(ca_in, net)
    ca_all_zeros = autograd.Variable(torch.from_numpy(np.zeros((600, 1))).float(), requires_grad = False)
    mse_f = mse_loss_function(ca_out, ca_all_zeros)
    loss = mse_f
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(epoch, "Traning Loss:", loss.data)
        print(f'times {epoch}  -  loss: {loss.item()}')
    if epoch % 10000 == 0:
        plt.cla()
        pre_in = torch.linspace(-8.0, -4.0, 9000)
        pre_in = torch.reshape(pre_in, (9000, 1))
        y = torch.sin(w1 * pre_in) + torch.sin(w2 * pre_in)
        y_train0 = torch.tanh(w2 * (pre_in + 2 * torch.pi)) * net(pre_in)

        plt.xlabel(f'x-{epoch}')
        plt.ylabel('sin(x)')
        plt.plot(pre_in.detach().numpy(), y.detach().numpy(), linewidth=0.5)
        plt.plot(pre_in.detach().numpy(), y_train0.detach().numpy(), c='red', linewidth=0.5)
        plt.pause(0.1)
torch.save(net.state_dict(), "ddpinn_sub_d")

