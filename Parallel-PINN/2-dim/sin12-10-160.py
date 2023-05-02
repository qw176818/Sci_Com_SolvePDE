import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch import autograd
import os
class Net(nn.Module):
    def __init__(self, NN):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(1, NN)
        self.h1_layer = nn.Linear(NN, NN)
        self.h2_layer = nn.Linear(NN, NN)
        self.h3_layer = nn.Linear(NN, NN)
        self.h4_layer = nn.Linear(NN, NN)
        self.h5_layer = nn.Linear(NN, NN)
        self.h6_layer = nn.Linear(NN, NN)
        self.h7_layer = nn.Linear(NN, NN)
        self.h8_layer = nn.Linear(NN, NN)
        self.h9_layer = nn.Linear(NN, NN)
        self.output_layer = nn.Linear(NN, 1)
    def forward(self, x):
        out1 = torch.tanh(self.input_layer(x))
        out2 = torch.tanh(self.h1_layer(out1))
        out3 = torch.tanh(self.h2_layer(out2))
        out4 = torch.tanh(self.h3_layer(out3))
        out5 = torch.tanh(self.h4_layer(out4))
        out6 = torch.tanh(self.h5_layer(out5))
        out7 = torch.tanh(self.h6_layer(out6))
        out8 = torch.tanh(self.h7_layer(out7))
        out9 = torch.tanh(self.h8_layer(out8))
        out10 = torch.tanh(self.h9_layer(out9))
        out_final = self.output_layer(out10)
        return out_final
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义的神经网络
net = Net(160)
net.to(device)
# 定义的损失函数
mse_loss_function = torch.nn.MSELoss(reduction='mean')
mse_loss_function = mse_loss_function.to(device)
# 定义的优化器
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4)

def PDE(x, w, net):
    Tan = torch.tanh(w * x).to(device)
    y = -1 / w / w * Tan + Tan * Tan * net(x)
    y_x = autograd.grad(y, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    y_xx = autograd.grad(y_x, x, torch.ones_like(x), create_graph = True, retain_graph = True)[0]
    tmp = torch.sin(w * x).to(device)
    re = y_xx - tmp
    return re
plt.ion()
iteration = 50000
w = 12
# loss_sum = []
# for epoch in range(iteration + 1):
#     optimizer.zero_grad()
#     x_in = np.random.uniform(low = -6.0, high = 6.0, size = (9000, 1))
#     ca_in = autograd.Variable(torch.from_numpy(x_in).float(), requires_grad = True)
#     ca_in = ca_in.to(device)
#     ca_out = PDE(ca_in, w, net)
#     ca_all_zeros = autograd.Variable(torch.from_numpy(np.zeros((9000, 1))).float(), requires_grad = False)
#     ca_all_zeros = ca_all_zeros.to(device)
#     mse_f = mse_loss_function(ca_out, ca_all_zeros)
#
#     loss = mse_f
#     loss.backward()
#     optimizer.step()
#     if epoch % 100 == 0:
#         print(epoch, "Traning Loss:", loss.data)
#         print(f'times {epoch}  -  loss: {loss.item()}')
#     loss_sum.append([epoch + 1, loss.item()])
#
#     if epoch % 10000 == 0:
#         plt.cla()
#         pre_in = torch.linspace(-6, 6, 9000).to(device)
#         pre_in = torch.reshape(pre_in, (9000, 1)).to(device)
#         y =1 / w / w * torch.sin(w * pre_in).to(device)
#         y_train0 = -1 / w / w * torch.tanh(w * pre_in) + torch.tanh(w * pre_in) * torch.tanh(w * pre_in) * net(pre_in)
#
#         plt.xlabel(f'x-{epoch}')
#         plt.ylabel('sin(x)')
#         plt.plot(pre_in.cpu().detach().numpy(), y.cpu().detach().numpy(), linewidth=0.5)
#         plt.plot(pre_in.cpu().detach().numpy(), y_train0.cpu().detach().numpy(), c='red', linewidth=0.5)
#         plt.pause(0.1)
# torch.save(net.state_dict(), "PINN-12-10-160-model")
net.load_state_dict(torch.load('PINN-12-10-160-model'))
# np.savetxt("./log/PINN_loss_w12_10_160.csv", loss_sum, delimiter=',')
# net.load_state_dict(torch.load('PINN-12-8-80-model'))
x = np.linspace(-6, 6, 9000)
y_ext = -1 / w / w * np.sin(w * x)
y_xx_ext = np.sin(w * x)
x0 = torch.reshape(torch.tensor(x, dtype=torch.float32), (9000, 1)).to(device)
x0.requires_grad = True
Tan = torch.tanh(w * x0).to(device)
y0 = -1 / w / w * Tan + Tan * Tan * net(x0)
y_x = autograd.grad(y0, x0, torch.ones_like(x0), create_graph = True, retain_graph = True)[0]
y_xx = autograd.grad(y_x, x0, torch.ones_like(x0), create_graph = True, retain_graph = True)[0]
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

fig = plt.figure(figsize = (10.0, 5.0))
# plt.plot(x, y_ext, color='green', label='Exact')
# plt.plot(x0.cpu().detach().numpy(), y0.cpu().detach().numpy(), color = 'red', label = 'PINN')
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.ylim(-0.2, 0.2)
# plt.xlim(-7.0, 7.0)
# plt.legend(fontsize = 13)
# plt.title("PINN w = 12, 10 layer 160 neural per layer")
# plt.savefig("12-10-160.pdf")
# plt.savefig("12-10-160.eps")

plt.plot(x, y_xx_ext, color='green', label='Exact')
plt.plot(x0.cpu().detach().numpy(), y_xx.cpu().detach().numpy(), color = 'red', label = 'PINN')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('x')
plt.ylabel(r"$\frac{d^2u}{dx^2}$")
plt.ylim(-2.2, 2.2)
plt.xlim(-7.0, 7.0)
plt.legend(fontsize = 13)
plt.title("PINN w = 12, 10 layer 160 neural per layer")
plt.savefig("12-10-160-dxx.pdf")
plt.savefig("12-10-160-dxx.eps")

x0 = np.linspace(-6, 6, 500)
x0 = np.reshape(x0, (500, 1))
y_ext =  -1 / w / w * np.sin(w * x0)
y_xx_ext = np.sin(w * x0)
x0 = torch.reshape(torch.tensor(x0, dtype=torch.float32).to(device), (500, 1)).to(device)
x0.requires_grad = True
t = torch.tanh(w * x0).to(device)

y0 = -1 / w / w * t + t * t * net(x0)
y_x = autograd.grad(y0, x0, torch.ones_like(x0), create_graph = True, retain_graph = True)[0]
y_xx = autograd.grad(y_x, x0, torch.ones_like(x0), create_graph = True, retain_graph = True)[0]
error = np.linalg.norm(y_ext - y0.cpu().detach().numpy(), 2) / np.linalg.norm(y_ext, 2)
error_u_xx = np.linalg.norm(y_xx_ext - y_xx.cpu().detach().numpy(), 2) / np.linalg.norm(y_xx_ext, 2)
print(error, error_u_xx)