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
        out1 = torch.sin(self.input_layer(x))
        out2 = torch.sin(self.h1_layer(out1))
        out3 = torch.sin(self.h2_layer(out2))
        out4 = torch.sin(self.h3_layer(out3))
        out5 = torch.sin(self.h4_layer(out4))
        # out6 = torch.sin(self.h5_layer(out5))
        # out7 = torch.sin(self.h6_layer(out6))
        out_final = self.output_layer(out5)
        return out_final
# 定义的神经网络
net = Net(40)
# 定义的损失函数
mse_loss_function = torch.nn.MSELoss(reduction='mean')
# 定义的优化器
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4)
def PDE(x, w, net):
    y = -1 / w / w * torch.tanh(w * (x - torch.pi)) + torch.tanh(w * (x - torch.pi)) * torch.tanh(w * (x - torch.pi)) * net(x)
    y_x = autograd.grad(y, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    y_xx = autograd.grad(y_x, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    tmp = torch.sin(w * x)
    re = y_xx - tmp
    return re
plt.ion()
iteration = 100000
w = 12
loss_sum = []
for epoch in range(iteration + 1):
    optimizer.zero_grad()
    x_in = np.random.uniform(low = 2, high = 6, size = (800, 1))
    ca_in = autograd.Variable(torch.from_numpy(x_in).float(), requires_grad = True)
    ca_out = PDE(ca_in, w, net)
    ca_all_zeros = autograd.Variable(torch.from_numpy(np.zeros((800, 1))).float(), requires_grad = False)
    mse_f = mse_loss_function(ca_out, ca_all_zeros)

    loss = mse_f
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        # y = torch.exp(-2 * ca_in)  # y 真实值
        # y_train0 = net(ca_in)  # y 预测值
        print(epoch, "Traning Loss:", loss.data)
        print(f'times {epoch}  -  loss: {loss.item()}')
    loss_sum.append([epoch + 1, loss.item()])

    if epoch % 10000 == 0:
        plt.cla()
        pre_in = torch.linspace(2, 6, 9000)
        pre_in = torch.reshape(pre_in, (9000, 1))
        y = -1 / w / w * torch.sin(w * pre_in)
        y_train0 = -1 / w / w * torch.tanh(w * (pre_in - torch.pi)) + torch.tanh(w * (pre_in - torch.pi)) * torch.tanh(w * (pre_in - torch.pi)) * net(pre_in)

        plt.xlabel(f'x-{epoch}')
        plt.ylabel('sin(x)')
        plt.plot(pre_in.detach().numpy(), y.detach().numpy(), linewidth=0.5)
        plt.plot(pre_in.detach().numpy(), y_train0.detach().numpy(), c='red', linewidth=0.5)
        plt.pause(0.1)
torch.save(net.state_dict(), "./ddpinn_sub_d3")