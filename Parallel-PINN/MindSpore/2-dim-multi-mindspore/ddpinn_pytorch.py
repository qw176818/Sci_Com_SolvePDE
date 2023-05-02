import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch import autograd
import time
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


net1 = Net(60)
net2 = Net(60)
net3 = Net(60)
net4 = Net(60)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']*100
a = [[-8, -4], [-4.5, 0.5], [-0.5, 4.5], [4.0, 8.0]]
bias = [2, 1, -1, -2]
# net = [net1, net2, net3, net4]
num_p = 500
w1 = 1
w2 = 7
plt.ion()
iteration = 50000
# 定义的损失函数
mse_loss_function = torch.nn.MSELoss(reduction='mean')
# 定义的优化器
optimizer1 = torch.optim.Adam(net1.parameters(), lr = 1e-4)
optimizer2 = torch.optim.Adam(net2.parameters(), lr = 1e-4)
optimizer3 = torch.optim.Adam(net3.parameters(), lr = 1e-4)
optimizer4 = torch.optim.Adam(net4.parameters(), lr = 1e-4)

def PDE1(x, net):
    y = torch.tanh(w2 * (x + 2 * torch.pi)) * net(x)
    y_x = autograd.grad(y, x, torch.ones_like(x), create_graph=True, retain_graph=True)[0]
    tmp = w1 * torch.cos(w1 * x) + w2 * torch.cos(w2 * x)
    res = y_x - tmp
    return res

def PDE2(x, net):
    y = torch.tanh(w2 * (x + torch.pi)) * net(x)
    y_x = autograd.grad(y, x, torch.ones_like(x), create_graph=True, retain_graph=True)[0]
    tmp = w1 * torch.cos(w1 * x) + w2 * torch.cos(w2 * x)
    res = y_x - tmp
    return res

def PDE3(x, net):
    y = torch.tanh(w2 * (x - torch.pi)) * net(x)
    y_x = autograd.grad(y, x, torch.ones_like(x), create_graph=True, retain_graph=True)[0]
    tmp = w1 * torch.cos(w1 * x) + w2 * torch.cos(w2 * x)
    res = y_x - tmp
    return res

def PDE4(x, net):
    y = torch.tanh(w2 * (x - 2 * torch.pi)) * net(x)
    y_x = autograd.grad(y, x, torch.ones_like(x), create_graph=True, retain_graph=True)[0]
    tmp = w1 * torch.cos(w1 * x) + w2 * torch.cos(w2 * x)
    res = y_x - tmp
    return res

start_time = time.time()
for epoch in range(iteration + 1):
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    optimizer3.zero_grad()
    optimizer4.zero_grad()
    x_in1 = np.random.uniform(low = -8.0, high = -4.0, size = (600, 1))
    ca_in1 = autograd.Variable(torch.from_numpy(x_in1).float(), requires_grad = True)
    ca_out1 = PDE1(ca_in1, net1)
    ca_all_zeros1 = autograd.Variable(torch.from_numpy(np.zeros((600, 1))).float(), requires_grad = False)
    mse_f1 = mse_loss_function(ca_out1, ca_all_zeros1)


    x_in2 = np.random.uniform(low = -4.5, high = 0.5, size = (400, 1))
    ca_in2 = autograd.Variable(torch.from_numpy(x_in2).float(), requires_grad = True)
    ca_out2 = PDE2(ca_in2, net2)
    ca_all_zeros2 = autograd.Variable(torch.from_numpy(np.zeros((400, 1))).float(), requires_grad = False)
    mse_f2 = mse_loss_function(ca_out2, ca_all_zeros2)


    x_in3 = np.random.uniform(low = -0.5, high = 4.5, size = (400, 1))
    ca_in3 = autograd.Variable(torch.from_numpy(x_in3).float(), requires_grad = True)
    ca_out3 = PDE3(ca_in3, net3)
    ca_all_zeros3 = autograd.Variable(torch.from_numpy(np.zeros((400, 1))).float(), requires_grad = False)
    mse_f3 = mse_loss_function(ca_out3, ca_all_zeros3)


    x_in4 = np.random.uniform(low = 4.0, high = 8.0, size = (600, 1))
    ca_in4 = autograd.Variable(torch.from_numpy(x_in4).float(), requires_grad = True)
    ca_out4 = PDE4(ca_in4, net4)
    ca_all_zeros4 = autograd.Variable(torch.from_numpy(np.zeros((600, 1))).float(), requires_grad = False)
    mse_f4 = mse_loss_function(ca_out4, ca_all_zeros4)

    mse_f1.backward()
    mse_f2.backward()
    mse_f3.backward()
    mse_f4.backward()

    optimizer1.step()
    optimizer2.step()
    optimizer3.step()
    optimizer4.step()
    if epoch % 100 == 0:
        print(epoch, "Traning Loss:", mse_f1.data, mse_f2.data, mse_f3.data, mse_f4.data)
        print(f'times {epoch}  -  loss: {mse_f1.item(), mse_f2.item(), mse_f3.item(), mse_f4.item()}')
    # if epoch % 10000 == 0:
    #     plt.cla()
    #     pre_in = torch.linspace(-8.0, -4.0, 9000)
    #     pre_in = torch.reshape(pre_in, (9000, 1))
    #     y = torch.sin(w1 * pre_in) + torch.sin(w2 * pre_in)
    #     y_train0 = torch.tanh(w2 * (pre_in + 2 * torch.pi)) * net(pre_in)
    #
    #     plt.xlabel(f'x-{epoch}')
    #     plt.ylabel('sin(x)')
    #     plt.plot(pre_in.detach().numpy(), y.detach().numpy(), linewidth=0.5)
    #     plt.plot(pre_in.detach().numpy(), y_train0.detach().numpy(), c='red', linewidth=0.5)
    #     plt.pause(0.1)
eclapse = time.time() - start_time
print(eclapse)
torch.save(net1.state_dict(), "ddpinn_sub_d1")
torch.save(net2.state_dict(), "ddpinn_sub_d2")
torch.save(net3.state_dict(), "ddpinn_sub_d3")
torch.save(net4.state_dict(), "ddpinn_sub_d4")

net1.load_state_dict(torch.load("./ddpinn_sub_d1"))
net2.load_state_dict(torch.load("./ddpinn_sub_d2"))
net3.load_state_dict(torch.load("./ddpinn_sub_d3"))
net4.load_state_dict(torch.load("./ddpinn_sub_d4"))
net = [net1, net2, net3, net4]
for i in range(4):
    xpre = torch.linspace(a[i][0], a[i][1], num_p)
    xpre = torch.reshape(xpre, (num_p, 1))
    ypre = torch.tanh(w2 * (xpre + torch.pi * bias[i])) * net[i](xpre)
    y_star = torch.sin(w1 * xpre) + torch.sin(w2 * xpre)
    sub_l2_norm = np.linalg.norm(y_star.detach().numpy() - ypre.detach().numpy(), 2) \
                   / np.linalg.norm(y_star.detach().numpy(), 2)
    print("sub" + str(i + 1) + "_l2_norm: ")
    print(sub_l2_norm)
a2 = [[-8, -4.5], [-4.5, -4], [-4, -0.5], [-0.5, 0.5], [0.5, 4], [4.0, 4.5], [4.5, 8.0]]
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
plt.ylabel('u')
plt.ylim(-2.2, 2.2)
plt.xlim(-8.0, 12.0)
plt.title("DDPINN $w_{1} = 1, w_{2} = 7$")
plt.legend(fontsize = 13)
# plt.show()
plt.savefig("DDPINN-2-multy-12.pdf")
plt.savefig("DDPINN-2-multy-12.eps")
num1 = 225
num2 = 33
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