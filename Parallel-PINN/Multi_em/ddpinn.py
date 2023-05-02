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
        self.output_layer = nn.Linear(NN, 1)
    def forward(self, x):
        out1 = torch.tanh(self.input_layer(x))
        out2 = torch.tanh(self.h1_layer(out1))
        out3 = torch.tanh(self.h2_layer(out2))
        out4 = torch.tanh(self.h3_layer(out3))
        out5 = torch.tanh(self.h4_layer(out4))
        out_final = self.output_layer(out5)
        return out_final
w1 = 1
w2 = 12
def PDE1(x, net):
    y = torch.tanh(w2 * (x + 2 * torch.pi)) * net(x)
    y_x = autograd.grad(y, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    # y_xx = autograd.grad(y_x, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    tmp = w1 * torch.cos(w1 * x) + w2 * torch.cos(w2 * x)
    y1_x = y_x - tmp
    return y1_x
def PDE2(x, net):
    y = torch.tanh(w2 * (x + torch.pi)) * net(x)
    y_x = autograd.grad(y, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    # y_xx = autograd.grad(y_x, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    tmp = w1 * torch.cos(w1 * x) + w2 * torch.cos(w2 * x)
    y1_x = y_x - tmp
    return y1_x
def PDE3(x, net):
    y = torch.tanh(w2 * x) * net(x)
    y_x = autograd.grad(y, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    # y_xx = autograd.grad(y_x, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    tmp = w1 * torch.cos(w1 * x) + w2 * torch.cos(w2 * x)
    y1_x = y_x - tmp
    return y1_x
def PDE4(x, net):
    y = torch.tanh(w2 * (x - torch.pi)) * net(x)
    y_x = autograd.grad(y, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    # y_xx = autograd.grad(y_x, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    tmp = w1 * torch.cos(w1 * x) + w2 * torch.cos(w2 * x)
    y1_x = y_x - tmp
    return y1_x
def PDE5(x, net):
    y = torch.tanh(w2 * (x - 2 * torch.pi)) * net(x)
    y_x = autograd.grad(y, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    # y_xx = autograd.grad(y_x, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    tmp = w1 * torch.cos(w1 * x) + w2 * torch.cos(w2 * x)
    y1_x = y_x - tmp
    return y1_x
# 定义的神经网络
net1 = Net(40)
net2 = Net(40)
net3 = Net(40)
net4 = Net(40)
net5 = Net(40)
# 定义的损失函数
mse_loss_function = torch.nn.MSELoss(reduction='mean')
# 定义的优化器
optimizer1 = torch.optim.Adam(net1.parameters(), lr = 1e-4)
optimizer2 = torch.optim.Adam(net2.parameters(), lr = 1e-4)
optimizer3 = torch.optim.Adam(net3.parameters(), lr = 1e-4)
optimizer4 = torch.optim.Adam(net4.parameters(), lr = 1e-4)
optimizer5 = torch.optim.Adam(net5.parameters(), lr = 1e-4)
plt.ion()
iteration = 50000
# for epoch in range(iteration + 1):
#     optimizer1.zero_grad()
#     optimizer2.zero_grad()
#     optimizer3.zero_grad()
#     optimizer4.zero_grad()
#     optimizer5.zero_grad()
#
#     x1_in = np.random.uniform(low = -7, high = -4.5, size = (700, 1))
#     ca1_in = autograd.Variable(torch.from_numpy(x1_in).float(), requires_grad = True)
#     ca1_out = PDE1(ca1_in, net1)
#     ca1_all_zeros = autograd.Variable(torch.from_numpy(np.zeros((700, 1))).float(), requires_grad = False)
#     mse_f1 = mse_loss_function(ca1_out, ca1_all_zeros)
#
#     x2_in = np.random.uniform(low = -5, high = -1.5, size = (600, 1))
#     ca2_in = autograd.Variable(torch.from_numpy(x2_in).float(), requires_grad = True)
#     ca2_out = PDE2(ca2_in, net2)
#     ca2_all_zeros = autograd.Variable(torch.from_numpy(np.zeros((600, 1))).float(), requires_grad = False)
#     mse_f2 = mse_loss_function(ca2_out, ca2_all_zeros)
#
#     x3_in = np.random.uniform(low = -2, high = 2, size = (400, 1))
#     ca3_in = autograd.Variable(torch.from_numpy(x3_in).float(), requires_grad = True)
#     ca3_out = PDE3(ca3_in, net3)
#     ca3_all_zeros = autograd.Variable(torch.from_numpy(np.zeros((400, 1))).float(), requires_grad = False)
#     mse_f3 = mse_loss_function(ca3_out, ca3_all_zeros)
#
#     x4_in = np.random.uniform(low = 1.5, high = 5, size = (600, 1))
#     ca4_in = autograd.Variable(torch.from_numpy(x4_in).float(), requires_grad = True)
#     ca4_out = PDE4(ca4_in, net4)
#     ca4_all_zeros = autograd.Variable(torch.from_numpy(np.zeros((600, 1))).float(), requires_grad = False)
#     mse_f4 = mse_loss_function(ca4_out, ca4_all_zeros)
#
#     x5_in = np.random.uniform(low = 4.5, high = 7, size = (700, 1))
#     ca5_in = autograd.Variable(torch.from_numpy(x5_in).float(), requires_grad = True)
#     ca5_out = PDE5(ca5_in, net5)
#     ca5_all_zeros = autograd.Variable(torch.from_numpy(np.zeros((700, 1))).float(), requires_grad = False)
#     mse_f5 = mse_loss_function(ca5_out, ca5_all_zeros)
#
#     # d1 = np.random.uniform(low = -6, high = -2, size = (400, 1))
#     #
#     # d2 = np.random.uniform(low = -6, high = -2, size = (400, 1))
#     loss1 = mse_f1
#     loss2 = mse_f2
#     loss3 = mse_f3
#     loss4 = mse_f4
#     loss5 = mse_f5
#     loss1.backward()
#     loss2.backward()
#     loss3.backward()
#     loss4.backward()
#     loss5.backward()
#     optimizer1.step()
#     optimizer2.step()
#     optimizer3.step()
#     optimizer4.step()
#     optimizer5.step()
#     if epoch % 100 == 0:
#         print(epoch, "Traning Loss:", loss1.data)
#         print(f'times {epoch}  -  loss1: {loss1.item()}, loss2: {loss2.item()}, loss3: {loss3.item()}, loss4: {loss4.item()}, loss5: {loss5.item()}')
#     if epoch % 10000 == 0:
#         plt.cla()
#         pre_in1 = torch.linspace(-7, -4.5, 9000)
#         pre_in1 = torch.reshape(pre_in1, (9000, 1))
#         y = np.sin(w1 * pre_in1) + np.sin(w2 * pre_in1)
#         y_train0 = torch.tanh(w2 * (pre_in1 + 2 * torch.pi)) * net1(pre_in1)
#         plt.xlabel(f'x-{epoch}')
#         plt.ylabel('sin(x)')
#         plt.subplot(1, 5, 1)
#         plt.plot(pre_in1.detach().numpy(), y.detach().numpy(), linewidth=0.5)
#         plt.plot(pre_in1.detach().numpy(), y_train0.detach().numpy(), c='red', linewidth=0.5)
#
#         pre_in2 = torch.linspace(-5, -1.5, 9000)
#         pre_in2 = torch.reshape(pre_in2, (9000, 1))
#         y = np.sin(w1 * pre_in2) + np.sin(w2 * pre_in2)
#         y_train0 = torch.tanh(w2 * (pre_in2 + torch.pi)) * net2(pre_in2)
#         plt.xlabel(f'x-{epoch}')
#         plt.ylabel('sin(x)')
#         plt.subplot(1, 5, 2)
#         plt.plot(pre_in2.detach().numpy(), y.detach().numpy(), linewidth=0.5)
#         plt.plot(pre_in2.detach().numpy(), y_train0.detach().numpy(), c='red', linewidth=0.5)
#
#         pre_in3 = torch.linspace(-2, 2, 9000)
#         pre_in3 = torch.reshape(pre_in3, (9000, 1))
#         y = np.sin(w1 * pre_in3) + np.sin(w2 * pre_in3)
#         y_train0 = torch.tanh(w2 * (pre_in3)) * net3(pre_in3)
#
#         plt.xlabel(f'x-{epoch}')
#         plt.ylabel('sin(x)')
#         plt.subplot(1, 5, 3)
#         plt.plot(pre_in3.detach().numpy(), y.detach().numpy(), linewidth=0.5)
#         plt.plot(pre_in3.detach().numpy(), y_train0.detach().numpy(), c='red', linewidth=0.5)
#
#
#         pre_in4 = torch.linspace(1.5, 5, 9000)
#         pre_in4 = torch.reshape(pre_in4, (9000, 1))
#         y = np.sin(w1 * pre_in4) + np.sin(w2 * pre_in4)
#         y_train0 = torch.tanh(w2 * (pre_in4 - torch.pi)) * net4(pre_in4)
#         plt.xlabel(f'x-{epoch}')
#         plt.ylabel('sin(x)')
#         plt.subplot(1, 5, 4)
#         plt.plot(pre_in4.detach().numpy(), y.detach().numpy(), linewidth=0.5)
#         plt.plot(pre_in4.detach().numpy(), y_train0.detach().numpy(), c='red', linewidth=0.5)
#
#
#         pre_in5 = torch.linspace(4.5, 7, 9000)
#         pre_in5 = torch.reshape(pre_in5, (9000, 1))
#         y = np.sin(w1 * pre_in5) + np.sin(w2 * pre_in5)
#         y_train0 = torch.tanh(w2 * (pre_in5 - 2 * torch.pi)) * net5(pre_in5)
#         plt.xlabel(f'x-{epoch}')
#         plt.ylabel('sin(x)')
#         plt.subplot(1, 5, 5)
#         plt.plot(pre_in5.detach().numpy(), y.detach().numpy(), linewidth=0.5)
#         plt.plot(pre_in5.detach().numpy(), y_train0.detach().numpy(), c='red', linewidth=0.5)
#         plt.pause(0.1)
# torch.save(net1.state_dict(), "./ddpinn_sub_d1")
# torch.save(net2.state_dict(), "./ddpinn_sub_d2")
# torch.save(net3.state_dict(), "./ddpinn_sub_d3")
# torch.save(net2.state_dict(), "./ddpinn_sub_d4")
# torch.save(net3.state_dict(), "./ddpinn_sub_d5")

net1.load_state_dict(torch.load("./ddpinn_sub_d1"))
net2.load_state_dict(torch.load("./ddpinn_sub_d2"))
net3.load_state_dict(torch.load("./ddpinn_sub_d3"))
net4.load_state_dict(torch.load("./ddpinn_sub_d4"))
net5.load_state_dict(torch.load("./ddpinn_sub_d5"))

a = [[-7, -4.5], [-5, -1.5], [-2, 2], [1.5, 5], [4.5, 7]]
bias = [2, 1, 0, -1, -2]
net = [net1, net2, net3, net4, net5]
num_p = 500
for i in range(5):
    xpre = torch.linspace(a[i][0], a[i][1], num_p)
    xpre = torch.reshape(xpre, (num_p, 1))
    ypre = torch.tanh(w2 * (xpre + torch.pi * bias[i])) * net[i](xpre)
    y_star = torch.sin(w1 * xpre) + torch.sin(w2 * xpre)
    sub_l2_norm = np.linalg.norm(y_star.detach().numpy() - ypre.detach().numpy(), 2) \
                   / np.linalg.norm(y_star.detach().numpy(), 2)
    print("sub" + str(i + 1) + "_l2_norm: ")
    print(sub_l2_norm)
a2 = [[-7, -5], [-5, -4.5], [-4.5, -2], [-2, -1.5], [-1.5, 1.5], [1.5, 2], [2, 4.5], [4.5, 5], [5, 7]]
# net = [net1, net, net2, net, net3, net, net4, net, net5]
colors = ['tab:blue', 'tab:orange', 'tab:olive', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']*100
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
fig = plt.figure(figsize = (10.0, 5.0))
x = torch.linspace(-7, 7, num_p)
x = torch.reshape(x, (num_p, 1))
y_star = torch.sin(w1 * x) + torch.sin(w2 * x)
plt.plot(x.detach().numpy(), y_star.detach().numpy(), color='green', label='Exact')
num_p = 1000
for i, d in enumerate(a2):
    x = torch.linspace(a2[i][0], a2[i][1], num_p)
    x = torch.reshape(x, (num_p, 1))

    if i % 2 == 0:
        y = torch.tanh(w2 * (x + bias[int(i / 2)] * torch.pi)) * (net[int(i / 2)](x))
        plt.plot(x.detach().numpy(), y.detach().numpy(),  colors[int(i / 2)], label='Sub-d'+str(int((i / 2) + 1)))
    else:
        y1 = torch.tanh(w2 * (x + bias[int(i / 2)] * torch.pi)) * net[int(i / 2)](x)
        y2 = torch.tanh(w2 * (x + bias[int((i + 1) / 2)] * torch.pi)) * net[int((i + 1) / 2)](x)
        y = (y1 + y2) / 2
        plt.plot(x.detach().numpy(), y.detach().numpy(),  "pink", label='Cross-R'+str(int((i + 1) / 2)))

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-2.2, 2.2)
plt.xlim(-7.0, 11.0)
plt.title("DDPINN $w_{1} = 1, w_{2} = 12$")
plt.legend(fontsize = 13)
# plt.show()
plt.savefig("DDPINN-1-multy-12.pdf")
plt.savefig("DDPINN-1-multy-12.eps")
num1 = 180
num2 = 25
x_sum = np.array([[0]])
y_sum = np.array([[0]])
y_sumr = np.array([[0]])
for i, d in enumerate(a2):
    if i % 2 == 0:
        x = torch.linspace(a2[i][0], a2[i][1], num1)
        x = torch.reshape(x, (num1, 1))
        y_r = np.sin(w1 * x) + np.sin(w2 * x)
        y = (torch.tanh(w2 * (x + bias[int(i / 2)] * torch.pi)) * (net[int(i / 2)](x))).detach().numpy()
        x = x.detach().numpy()
        y_sum = np.concatenate((y_sum, y))
        x_sum = np.concatenate((x_sum, x))
        y_sumr = np.concatenate((y_sumr, y_r))
    else:
        x = torch.linspace(a2[i][0], a2[i][1], num2)
        x = torch.reshape(x, (num2, 1))
        y_r = torch.sin(w1 * x) + torch.sin(w2 * x)
        y1 = torch.tanh(w2 * (x + bias[int(i / 2)] * torch.pi)) * net[int(i / 2)](x)
        y2 = torch.tanh(w2 * (x + bias[int((i + 1) / 2)] * torch.pi)) * net[int((i + 1) / 2)](x)
        y = (y1 + y2) / 2
        y = y.detach().numpy()
        x = x.detach().numpy()
        y_sum = np.concatenate((y_sum, y))
        x_sum = np.concatenate((x_sum, x))
        y_sumr = np.concatenate((y_sumr, y_r))
x_sum = x_sum[1:]
y_sum = y_sum[1:]
y_sumr = y_sumr[1:]
error = np.linalg.norm(y_sum - y_sumr, 2) / np.linalg.norm(y_sumr, 2)
print(error)

# x1 = torch.linspace(-6, -2, 150)
# x1 = torch.reshape(x1, (150, 1))
# y1 = torch.tanh(w * (x1 + torch.pi)) * net1(x1).detach().numpy()
#
# x2 = torch.linspace(-2.5, 2.5, 200)
# x2 = torch.reshape(x2, (200, 1))
# y2 = torch.tanh(w * (x2)) * net2(x2).detach().numpy()
#
# x3 = torch.linspace(2, 6, 150)
# x3 = torch.reshape(x3, (150, 1))
# y3 = torch.tanh(w * (x3 - torch.pi)) * net3(x3).detach().numpy()
#
# x1 = x1.detach().numpy()
# x2 = x2.detach().numpy()
# x3 = x3.detach().numpy()
# x = np.concatenate((x1, x2, x3))
# y_r = np.sin(w * x)
# y = np.concatenate((y1, y2, y3))
# error = np.linalg.norm(y_r - y, 2) / np.linalg.norm(y_r, 2)
# print(error)
