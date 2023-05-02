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
        self.output_layer = nn.Linear(NN, 1)
    def forward(self, x):
        out1 = torch.tanh(self.input_layer(x))
        out2 = torch.tanh(self.h1_layer(out1))
        out3 = torch.tanh(self.h1_layer(out2))
        out4 = torch.tanh(self.h1_layer(out3))
        out_final = self.output_layer(out4)
        return out_final
# 定义的神经网络
net = Net(40)
# 定义的损失函数
mse_loss_function = torch.nn.MSELoss(reduction='mean')
# 定义的优化器
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4)
def PDE(x, w, net):
    y = -1 / w / w * torch.tanh(w * x) + torch.tanh(w * x) * torch.tanh(w * x) * net(x)
    y_x = autograd.grad(y, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    y_xx = autograd.grad(y_x, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    tmp = torch.sin(w * x)
    re = y_xx - tmp
    return re
plt.ion()
iteration = 50000
w = 12
# loss_sum = []
# for epoch in range(iteration + 1):
#     optimizer.zero_grad()
#     # x0 = torch.zeros(100, 1)
#     # y0 = net(x0)
#     # y0_r = torch.zeros(100, 1)
#     # mse_b_1 = mse_loss_function(y0, y0_r)
#     #
#     # x1 = torch.zeros(100, 1) + 6.0
#     # y1 = net(x1)
#     # y1_r = np.sin(6.0 * w) * torch.ones(100, 1)
#     # mse_b_2 = mse_loss_function(y1, y1_r)
#
#     x_in = np.random.uniform(low = -6.0, high = 6.0, size = (9000, 1))
#     ca_in = autograd.Variable(torch.from_numpy(x_in).float(), requires_grad = True)
#     ca_out = PDE(ca_in, w, net)
#     ca_all_zeros = autograd.Variable(torch.from_numpy(np.zeros((9000, 1))).float(), requires_grad = False)
#     mse_f = mse_loss_function(ca_out, ca_all_zeros)
#
#     loss = mse_f
#     loss.backward()
#     optimizer.step()
#     if epoch % 100 == 0:
#         # y = torch.exp(-2 * ca_in)  # y 真实值
#         # y_train0 = net(ca_in)  # y 预测值
#         print(epoch, "Traning Loss:", loss.data)
#         print(f'times {epoch}  -  loss: {loss.item()}')
#     loss_sum.append([epoch + 1, loss.item()])
#
#     if epoch % 10000 == 0:
#         plt.cla()
#         pre_in = torch.linspace(-6, 6, 9000)
#         pre_in = torch.reshape(pre_in, (9000, 1))
#         y = 1 / w / w * torch.sin(w * pre_in)
#         y_train0 = -1 / w / w * torch.tanh(w * pre_in) + torch.tanh(w * pre_in) * torch.tanh(w * pre_in) * net(pre_in)
#
#         plt.xlabel(f'x-{epoch}')
#         plt.ylabel('sin(x)')
#         plt.plot(pre_in.detach().numpy(), y.detach().numpy(), linewidth=0.5)
#         plt.plot(pre_in.detach().numpy(), y_train0.detach().numpy(), c='red', linewidth=0.5)
#         plt.pause(0.1)
# torch.save(net.state_dict(), "PINN-12-4-40-model")
net.load_state_dict(torch.load('PINN-12-4-40-model'))
# np.savetxt("./log/PINN_loss_w12_4_40.csv", loss_sum, delimiter=',')
x = np.linspace(-6, 6, 9000)
y_ext = -1 / w / w * np.sin(w * x)
y_xx_ext = np.sin(w * x)
x0 = torch.reshape(torch.tensor(x, dtype=torch.float32), (9000, 1))
x0.requires_grad = True
y0 = -1 / w / w * torch.tanh(w * x0) + torch.tanh(w * x0) * torch.tanh(w * x0) * net(x0)
y_x = autograd.grad(y0, x0, torch.ones_like(x0), create_graph = True, retain_graph = True)[0]
y_xx = autograd.grad(y_x, x0, torch.ones_like(x0), create_graph = True, retain_graph = True)[0]
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

fig = plt.figure(figsize = (10.0, 5.0))
# plt.plot(x, y_ext, color='green', label='Exact')
# plt.plot(x0.detach().numpy(), y0.detach().numpy(), color = 'red', label = 'PINN')
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.ylim(-0.2, 0.2)
# plt.xlim(-7.0, 7.0)
# plt.legend(fontsize = 13)
# plt.title("PINN w = 12, 4 layer 40 neural per layer")
# plt.savefig("12-4-40.pdf")
# plt.savefig("12-4-40.eps")

# plt.plot(x, y_xx_ext, color='green', label='Exact')
# plt.plot(x0.detach().numpy(), y_xx.detach().numpy(), color = 'red', label = 'PINN')
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.xlabel('x')
# plt.ylabel(r"$\frac{d^2u}{dx^2}$")
# plt.ylim(-2.2, 2.2)
# plt.xlim(-7.0, 7.0)
# plt.legend(fontsize = 13)
# plt.title("PINN w = 12, 4 layer 40 neural per layer")
# plt.savefig("12-4-40-dxx.pdf")
# plt.savefig("12-4-40-dxx.eps")

x0 = np.linspace(-6, 6, 500)
x0 = np.reshape(x0, (500, 1))
y_ext =  -1 / w / w * np.sin(w * x0)
y_xx_ext = np.sin(w * x0)
x0 = torch.reshape(torch.tensor(x0, dtype=torch.float32), (500, 1))
x0.requires_grad = True
y0 = -1 / w / w * torch.tanh(w * x0) + torch.tanh(w * x0) * torch.tanh(w * x0) * net(x0)
y_x = autograd.grad(y0, x0, torch.ones_like(x0), create_graph = True, retain_graph = True)[0]
y_xx = autograd.grad(y_x, x0, torch.ones_like(x0), create_graph = True, retain_graph = True)[0]
error = np.linalg.norm(y_ext - y0.detach().numpy(), 2) / np.linalg.norm(y_ext, 2)
error_u_xx = np.linalg.norm(y_xx_ext - y_xx.detach().numpy(), 2) / np.linalg.norm(y_xx_ext, 2)
print(error, error_u_xx)