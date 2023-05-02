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
class Net1(nn.Module):
    def __init__(self, NN):
        super(Net1, self).__init__()
        self.input_layer = nn.Linear(1, NN)
        self.h1_layer = nn.Linear(NN, NN)
        self.h2_layer = nn.Linear(NN, NN)
        self.h3_layer = nn.Linear(NN, NN)
        self.output_layer = nn.Linear(NN, 1)
    def forward(self, x):
        out1 = torch.sin(self.input_layer(x))
        out2 = torch.sin(self.h1_layer(out1))
        out3 = torch.sin(self.h2_layer(out2))
        out4 = torch.sin(self.h3_layer(out3))
        out_final = self.output_layer(out4)
        return out_final
# 定义的神经网络
net1 = Net(30)
net2 = Net1(30)
net3 = Net(30)
# 定义的损失函数
mse_loss_function = torch.nn.MSELoss(reduction='mean')
# 定义的优化器
optimizer1 = torch.optim.Adam(net1.parameters(), lr = 1e-4)
optimizer2 = torch.optim.Adam(net2.parameters(), lr = 1e-4)
optimizer3 = torch.optim.Adam(net3.parameters(), lr = 1e-4)
def PDE1(x, w, net):
    y = torch.tanh(w * (x + torch.pi)) * net1(x)
    y_x = autograd.grad(y, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    # y_xx = autograd.grad(y_x, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    tmp = w * torch.cos(w * x)
    y1_x = y_x - tmp
    return y1_x
def PDE2(x, w, net):
    y = torch.tanh(w * x) * net2(x)
    y_x = autograd.grad(y, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    # y_xx = autograd.grad(y_x, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    tmp = w * torch.cos(w * x)
    y1_x = y_x - tmp
    return y1_x
def PDE3(x, w, net):
    y = torch.tanh(w * (x - torch.pi)) * net3(x)
    y_x = autograd.grad(y, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    # y_xx = autograd.grad(y_x, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    tmp = w * torch.cos(w * x)
    y1_x = y_x - tmp
    return y1_x
plt.ion()
iteration = 50000
w = 12
# for epoch in range(iteration + 1):
#     optimizer1.zero_grad()
#     optimizer2.zero_grad()
#     optimizer3.zero_grad()
#
#     x1_in = np.random.uniform(low = -6, high = -2, size = (400, 1))
#     ca1_in = autograd.Variable(torch.from_numpy(x1_in).float(), requires_grad = True)
#     ca1_out = PDE1(ca1_in, w, net1)
#     ca1_all_zeros = autograd.Variable(torch.from_numpy(np.zeros((400, 1))).float(), requires_grad = False)
#     mse_f1 = mse_loss_function(ca1_out, ca1_all_zeros)
#
#     x2_in = np.random.uniform(low = -2.5, high = 2.5, size = (200, 1))
#     ca2_in = autograd.Variable(torch.from_numpy(x2_in).float(), requires_grad = True)
#     ca2_out = PDE2(ca2_in, w, net2)
#     ca2_all_zeros = autograd.Variable(torch.from_numpy(np.zeros((200, 1))).float(), requires_grad = False)
#     mse_f2 = mse_loss_function(ca2_out, ca2_all_zeros)
#
#     x3_in = np.random.uniform(low = 2, high = 6, size = (400, 1))
#     ca3_in = autograd.Variable(torch.from_numpy(x3_in).float(), requires_grad = True)
#     ca3_out = PDE3(ca3_in, w, net3)
#     ca3_all_zeros = autograd.Variable(torch.from_numpy(np.zeros((400, 1))).float(), requires_grad = False)
#     mse_f3 = mse_loss_function(ca3_out, ca3_all_zeros)
#
#     b1 = np.random.uniform(low = -6, high = -2, size = (400, 1))
#
#     d2 = np.random.uniform(low = -6, high = -2, size = (400, 1))
#
#
#     loss1 = mse_f1
#     loss2 = mse_f2
#     loss3 = mse_f3
#     loss1.backward()
#     loss2.backward()
#     loss3.backward()
#     optimizer1.step()
#     optimizer2.step()
#     optimizer3.step()
#     if epoch % 100 == 0:
#         print(epoch, "Traning Loss:", loss1.data)
#         print(f'times {epoch}  -  loss1: {loss1.item()}, loss2: {loss2.item()}, loss3: {loss3.item()}')
#     if epoch % 10000 == 0:
#         plt.cla()
#         pre_in1 = torch.linspace(-6, -2, 9000)
#         pre_in1 = torch.reshape(pre_in1, (9000, 1))
#         y = torch.sin(w * pre_in1)
#         y_train0 = torch.tanh(w * (pre_in1 + torch.pi)) * net1(pre_in1)
#
#         plt.xlabel(f'x-{epoch}')
#         plt.ylabel('sin(x)')
#         plt.subplot(1, 3, 1)
#         plt.plot(pre_in1.detach().numpy(), y.detach().numpy(), linewidth=0.5)
#         plt.plot(pre_in1.detach().numpy(), y_train0.detach().numpy(), c='red', linewidth=0.5)
#
#         pre_in2 = torch.linspace(-2.5, 2.5, 9000)
#         pre_in2 = torch.reshape(pre_in2, (9000, 1))
#         y = torch.sin(w * pre_in2)
#         y_train0 = torch.tanh(w * pre_in2) * net2(pre_in2)
#
#         plt.xlabel(f'x-{epoch}')
#         plt.ylabel('sin(x)')
#         plt.subplot(1, 3, 2)
#         plt.plot(pre_in2.detach().numpy(), y.detach().numpy(), linewidth=0.5)
#         plt.plot(pre_in2.detach().numpy(), y_train0.detach().numpy(), c='red', linewidth=0.5)
#
#         pre_in3 = torch.linspace(2, 6, 9000)
#         pre_in3 = torch.reshape(pre_in3, (9000, 1))
#         y = torch.sin(w * pre_in3)
#         y_train0 = torch.tanh(w * (pre_in3 - torch.pi)) * net3(pre_in3)
#
#         plt.xlabel(f'x-{epoch}')
#         plt.ylabel('sin(x)')
#         plt.subplot(1, 3, 3)
#         plt.plot(pre_in3.detach().numpy(), y.detach().numpy(), linewidth=0.5)
#         plt.plot(pre_in3.detach().numpy(), y_train0.detach().numpy(), c='red', linewidth=0.5)
#         plt.pause(0.1)
# torch.save(net1.state_dict(), "./ddpinn_sub_d1")
# torch.save(net2.state_dict(), "./ddpinn_sub_d2")
# torch.save(net3.state_dict(), "./ddpinn_sub_d3")
net1.load_state_dict(torch.load("./ddpinn_sub_d1"))
net2.load_state_dict(torch.load("./ddpinn_sub_d2"))
net3.load_state_dict(torch.load("./ddpinn_sub_d3"))
xpre_1 = torch.linspace(-6, -2, 500)
xpre_1 = torch.reshape(xpre_1, (500, 1))
ypre_1 = torch.tanh(w * (xpre_1 + torch.pi)) * net1(xpre_1)
y_star1 = torch.sin(w * xpre_1)

xpre_2 = torch.linspace(-2.5, 2.5, 500)
xpre_2 = torch.reshape(xpre_2, (500, 1))
ypre_2 = torch.tanh(w * xpre_2) * net2(xpre_2)
y_star2 = torch.sin(w * xpre_2)

xpre_3 = torch.linspace(2, 6, 500)
xpre_3 = torch.reshape(xpre_3, (500, 1))
ypre_3 = torch.tanh(w * (xpre_3 - torch.pi)) * net3(xpre_3)
y_star3 = torch.sin(w * xpre_3)
sub1_l2_norm = np.linalg.norm(y_star1.detach().numpy() - ypre_1.detach().numpy(), 2) / np.linalg.norm(y_star1.detach().numpy(), 2)
sub2_l2_norm = np.linalg.norm(y_star2.detach().numpy() - ypre_2.detach().numpy(), 2) / np.linalg.norm(y_star2.detach().numpy(), 2)
sub3_l2_norm = np.linalg.norm(y_star3.detach().numpy() - ypre_3.detach().numpy(), 2) / np.linalg.norm(y_star3.detach().numpy(), 2)
print(sub1_l2_norm, sub2_l2_norm, sub3_l2_norm)
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
fig = plt.figure(figsize = (11.0, 5.0))
plt.ylim(-1.2, 1.2)
plt.xlim(-6.0, 9.0)
plt.title("DDPINN w = 12")

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('x')
plt.ylabel('y')
x1 = torch.linspace(-6, -2.5, 500)
x1 = torch.reshape(x1, (500, 1))
y1 = torch.tanh(w * (x1 + torch.pi)) * net1(x1)
y1_r = torch.sin(w * x1)
# plt.plot(x1.detach().numpy(), y1_r.detach().numpy(), color= '', label = 'Sub-d1')
plt.plot(x1.detach().numpy(), y1.detach().numpy(), color= 'blue', label = 'Sub-d1')

x2 = torch.linspace(-2.5, -2, 1000)
x2 = torch.reshape(x2, (1000, 1))
y2_r = torch.sin(w * x2)
y2_1 = torch.tanh(w * (x2 + torch.pi)) * net1(x2)
y2_2 = torch.tanh(w * x2) * net2(x2)
y2 = (y2_1 + y2_2) / 2
plt.plot(x2.detach().numpy(), y2.detach().numpy(), color= 'pink', label = 'Cross-R1')

x1 = torch.linspace(-2, 2, 500)
x1 = torch.reshape(x1, (500, 1))
y1 = torch.tanh(w * x1) * net2(x1)
y1_r = torch.sin(w * x1)
plt.plot(x1.detach().numpy(), y1.detach().numpy(), color= 'orange', label = 'Sub-d2')

x2 = torch.linspace(2, 2.5, 1000)
x2 = torch.reshape(x2, (1000, 1))
y2_r = torch.sin(w * x2)
y2_3 = torch.tanh(w * (x2 - torch.pi)) * net3(x2)
y2_2 = torch.tanh(w * x2) * net2(x2)
y2 = (y2_3 + y2_2) / 2
plt.plot(x2.detach().numpy(), y2.detach().numpy(), color= 'pink', label = 'Cross-R2')

x1 = torch.linspace(2.5, 6, 500)
x1 = torch.reshape(x1, (500, 1))
y1 = torch.tanh(w * (x1 - torch.pi)) * net3(x1)
y1_r = torch.sin(w * x1)
plt.plot(x1.detach().numpy(), y1.detach().numpy(), color= 'red', label = 'Sub-d3')
plt.legend(fontsize = 13)
plt.savefig("DDPINN-1-sin-12-12.pdf")
plt.savefig("DDPINN-1-sin-12-12.eps")


x1 = torch.linspace(-6, -2, 150)
x1 = torch.reshape(x1, (150, 1))
y1 = torch.tanh(w * (x1 + torch.pi)) * net1(x1).detach().numpy()

x2 = torch.linspace(-2.5, 2.5, 200)
x2 = torch.reshape(x2, (200, 1))
y2 = torch.tanh(w * (x2)) * net2(x2).detach().numpy()

x3 = torch.linspace(2, 6, 150)
x3 = torch.reshape(x3, (150, 1))
y3 = torch.tanh(w * (x3 - torch.pi)) * net3(x3).detach().numpy()

x1 = x1.detach().numpy()
x2 = x2.detach().numpy()
x3 = x3.detach().numpy()
x = np.concatenate((x1, x2, x3))
y_r = np.sin(w * x)
y = np.concatenate((y1, y2, y3))
error = np.linalg.norm(y_r - y, 2) / np.linalg.norm(y_r, 2)
print(error)
