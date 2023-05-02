import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch import autograd
import matplotlib as mpl
import csv
# import wandb
# wandb.init(project = "test-project", entity = "diag_lixl")
def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        y.append(eval(row[1]))
        x.append(eval(row[0]))
    return x, y
class Net(nn.Module):
    def __init__(self, NN):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(1, NN)
        self.h1_layer = nn.Linear(NN, NN)
        self.h2_layer = nn.Linear(NN, NN)
        self.output_layer = nn.Linear(NN, 1)

    def forward(self, x):
        out1 = torch.tanh(self.input_layer(x))
        out2 = torch.tanh(self.h1_layer(out1))
        out3 = torch.tanh(self.h2_layer(out2))
        out_final = self.output_layer(out3)
        return out_final
# 定义的神经网络
net = Net(50)
# 定义的损失函数
mse_loss_function = torch.nn.MSELoss(reduction='mean')
# 定义的优化器
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4)
def PDE(x, net):
    y = net(x)
    y_x = autograd.grad(y, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    # y_xx = autograd.grad(y_x, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    tmp = 2 * torch.exp(-2 * x)
    y1_x = y_x + tmp
    return y1_x
iteration = 30000
x_result = []
y_result = []
loss_sum = []
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
    # wandb.log({"Loss": loss.item()})
    loss_sum.append([epoch, loss.item()/1000])
    if epoch % 100 == 0:
        # y = torch.exp(-2 * ca_in)  # y 真实值
        # y_train0 = net(ca_in)  # y 预测值
        print(epoch, "Traning Loss:", loss.data)
        print(f'times {epoch}  -  loss: {loss.item()}')
    if epoch % 10000 == 0:
        # plt.cla()
        pre_in = torch.linspace(-2, 2, 3000)
        pre_in = torch.reshape(pre_in, (3000, 1))
        y = torch.exp(-2 * pre_in)
        y_train0 = net(pre_in)
        # plt.subplot(2, 2, int(epoch / 10000) + 1)
        plt.xlabel(f'x-{epoch}')
        plt.ylabel('exp(-2x)')
        x_result.append(pre_in.detach().numpy())
        y_result.append(y_train0.detach().numpy())
        plt.plot(pre_in.detach().numpy(), y.detach().numpy(), linewidth=0.5)
        plt.plot(pre_in.detach().numpy(), y_train0.detach().numpy(), c='red', linewidth=0.5)
np.savetxt("PINN_loss.csv", loss_sum, delimiter=',')
figure = plt.figure(figsize=(8, 8))
plt.subplot(2, 2, 1)
y = np.exp(-2 * x_result[0])
plt.xlabel(f'x-{0}')
plt.ylabel('exp(-2x)')
plt.plot(x_result[0], y, linewidth = 0.5)
plt.plot(x_result[0], y_result[0], c='red', linewidth=0.5)
ax2 = plt.subplot(2, 2, 2)
plt.xlabel(f'x-{10000}')
plt.ylabel('exp(-2x)')
plt.plot(x_result[0], y, linewidth = 0.5)
plt.plot(x_result[0], y_result[1], c='red', linewidth=0.5)

ax3 = plt.subplot(2, 2, 3)
plt.xlabel(f'x-{20000}')
plt.ylabel('exp(-2x)')
plt.plot(x_result[0], y, linewidth = 0.5)
plt.plot(x_result[0], y_result[2], c='red', linewidth=0.5)

ax4 = plt.subplot(2, 2, 4)
plt.xlabel(f'x-{30000}')
plt.ylabel('exp(-2x)')
plt.plot(x_result[0], y, linewidth = 0.5)
plt.plot(x_result[0], y_result[3], c='red', linewidth=0.5)
plt.savefig('./ODE.pdf', format='pdf')
plt.show()

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

fig = plt.figure()
x2, y2 = readcsv("PINN_loss.csv")
plt.plot(x2, y2, color='red', label='PINN',    linewidth=0.5)

plt.savefig('./ODE1.pdf', format = 'pdf')
