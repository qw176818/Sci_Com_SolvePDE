import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch import autograd
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
class Net(nn.Module):
    def __init__(self, NN):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(1, NN)
        self.h1_layer = nn.Linear(NN, NN)
        # self.h2_layer = nn.Linear(NN, NN)
        self.output_layer = nn.Linear(NN, 1)
        self.a = nn.Parameter(torch.ones((1, NN),dtype=torch.float32, requires_grad = True))
    def forward(self, x):
        out1 = torch.tanh(self.input_layer(x))
        out2 = torch.tanh(self.h1_layer(out1))
        # out3 = torch.tanh(self.h2_layer(out2))
        out_final = self.output_layer(out2)
        return out_final
class Net1(nn.Module):
    def __init__(self, NN):
        super(Net1, self).__init__()
        self.input_layer = nn.Linear(1, NN)
        self.h1_layer = nn.Linear(NN, NN)
        # self.h2_layer = nn.Linear(NN, NN)
        self.output_layer = nn.Linear(NN, 1)
        self.a = nn.Parameter(torch.ones((1, NN),dtype=torch.float32, requires_grad = True))
    def forward(self, x):
        out1 = torch.tanh(self.a * self.input_layer(x))
        out2 = torch.tanh(self.a * self.h1_layer(out1))
        # out3 = torch.tanh(self.h2_layer(out2))
        out_final = self.output_layer(out2)
        return out_final
net = Net(100)
net1 = Net1(100)
def init_weights1(model):
    if isinstance(model, nn.Linear):
        torch.nn.init.xavier_uniform(model.weight)
        model.bias.data.fill_(0)
net.apply(init_weights1)
net1.apply(init_weights1)
optimizer = torch.optim.Adam(net.parameters(), lr = 2e-4)
optimizer1 = torch.optim.Adam(net1.parameters(), lr =2e-4)
iteration = 50000
mse_loss_function = torch.nn.MSELoss(reduction='mean')
x_in = torch.from_numpy(np.linspace(-4, 4, 201))
x_in = torch.reshape(x_in, (201, 1))
x_in = torch.tensor(x_in, dtype=torch.float32)
y_exact = torch.zeros((201, 1))
for i, d in enumerate(x_in):
    if d < 0:
        y_exact[i] = 0.1 * torch.sin(4 * d)
    else:
        y_exact[i] = 1 + 0.1 * d * torch.cos(16 * d)
fig = plt.figure(figsize=(9.5, 4))
plt.subplot(1, 2, 1)
plt.plot(x_in.detach().numpy(), y_exact.detach().numpy(), color='r', label='Exact')
plt.xlabel('x', fontsize=14)
plt.ylabel('u', fontsize=14)

# plt.subplot(1, 2, 2)
# plt.plot(x_in.detach().numpy(), y_exact.detach().numpy(), color='r', label='Exact')
# plt.xlabel('x', fontsize=14)
# plt.ylabel('u', fontsize=14)
# plt.legend(fontsize=13)
# plt.savefig('./sacmp.eps', format='eps')
for epoch in range(iteration + 1):
    optimizer.zero_grad()
    y_out = net(x_in)
    loss = mse_loss_function(y_out, y_exact)
    loss.backward()
    if epoch % 100 == 0:
        print(epoch, loss.item())
    optimizer.step()
    if epoch == 10000:
        y_pre = net(x_in)
        plt.plot(x_in.detach().numpy(), y_pre.detach().numpy(), color='b', label='Iter = 10000', linewidth=1, linestyle = '--')
    if epoch == 40000:
        y_pre = net(x_in)
        plt.plot(x_in.detach().numpy(), y_pre.detach().numpy(), color='g', label='Iter = 40000', linewidth=1, linestyle = '--')
    if epoch == 50000:
        y_pre = net(x_in)
        plt.plot(x_in.detach().numpy(), y_pre.detach().numpy(), color='y', label='Iter = 50000', linewidth=1, linestyle = '--')
plt.legend(fontsize=10)
plt.subplot(1, 2, 2)
plt.plot(x_in.detach().numpy(), y_exact.detach().numpy(), color='r', label='Exact')
plt.xlabel('x', fontsize=14)
plt.ylabel('u', fontsize=14)

for epoch in range(iteration + 1):
    optimizer1.zero_grad()
    y_out = net1(x_in)
    loss1 = mse_loss_function(y_out, y_exact)
    loss1.backward()
    if epoch % 100 == 0:
        print(epoch, loss1.item())
    optimizer1.step()
    if epoch == 10000:
        y_pre = net1(x_in)
        plt.plot(x_in.detach().numpy(), y_pre.detach().numpy(), color='b', label='Iter = 10000', linestyle = '--')
    if epoch == 40000:
        y_pre = net1(x_in)
        plt.plot(x_in.detach().numpy(), y_pre.detach().numpy(), color='g', label='Iter = 40000', linestyle = '--')
    if epoch == 50000:
        y_pre = net1(x_in)
        plt.plot(x_in.detach().numpy(), y_pre.detach().numpy(), color='y', label='Iter = 50000', linewidth=1, linestyle = '--')
plt.legend(fontsize=10)
plt.savefig('./sacmp.eps', format='eps')
plt.savefig('./sacmp.pdf', format='pdf')