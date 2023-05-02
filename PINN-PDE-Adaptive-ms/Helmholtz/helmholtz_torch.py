import numpy as np
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
sys.path.insert(0, '../../Utilities/')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torch.nn as nn
from torch import autograd
import os
from torch.utils.tensorboard import SummaryWriter
np.random.seed(1234)
mpl.rcParams['text.usetex'] = False
class Net(nn.Module):
    def __init__(self, NN, lb, ub):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(2, NN)
        self.h1_layer = nn.Linear(NN, NN)
        self.h2_layer = nn.Linear(NN, NN)
        self.h3_layer = nn.Linear(NN, NN)
        self.h4_layer = nn.Linear(NN, NN)
        self.output_layer = nn.Linear(NN, 1)
        self.lb = lb
        self.lb = self.lb.astype(np.float32)
        self.lb = torch.tensor(self.lb)
        self.ub = ub
        self.ub = self.ub.astype(np.float32)
        self.ub = torch.tensor(self.ub)
        # self.n = 1
    def forward(self, x):
        x1 = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        out1 = torch.tanh(self.input_layer(x1))
        out2 = torch.tanh(self.h1_layer(out1))
        out3 = torch.tanh(self.h2_layer(out2))
        out4 = torch.tanh(self.h3_layer(out3))
        out5 = torch.tanh(self.h4_layer(out4))
        out_final = self.output_layer(out5)
        return out_final
def init_weights1(model):
    if isinstance(model, nn.Linear):
        torch.nn.init.xavier_uniform(model.weight)
        model.bias.data.fill_(0)

def PDE(x, t, net):
    x = torch.from_numpy(x)
    x = torch.tensor(x, dtype = torch.float32)
    x = torch.reshape(x, (10000, 1))
    t = torch.from_numpy(t)
    t = torch.tensor(t, dtype = torch.float32)
    t = torch.reshape(t, (10000, 1))
    x.requires_grad_(True)
    t.requires_grad_(True)
    X = torch.concat((x, t), 1)
    net.lb = torch.tensor(net.lb, requires_grad = True)
    net.ub = torch.tensor(net.ub, requires_grad = True)
    X = X.to(device)
    u = net(X)
    u_t = autograd.grad(outputs = u,
                        inputs = t,
                        grad_outputs = torch.ones_like(u),
                        retain_graph = True,
                        create_graph = True,
                        allow_unused = True)
    u_tt = autograd.grad(outputs = u_t,
                        inputs = t,
                        grad_outputs = torch.ones_like(u_t[0]),
                        retain_graph = True,
                        create_graph = True,
                        allow_unused = True)
    u_x = autograd.grad(outputs = u,
                        inputs = x,
                        grad_outputs = torch.ones_like(u),
                        retain_graph = True,
                        create_graph = True,
                        allow_unused = True)
    u_xx = autograd.grad(outputs = u_x,
                        inputs = x,
                        grad_outputs = torch.ones_like(u_x[0]),
                        retain_graph = True,
                        create_graph = True,
                        allow_unused = True)
    pi = torch.pi
    tmp1 = ((6 - torch.pi * torch.pi - 36 * torch.pi * torch.pi) * \
           torch.sin(pi * x)).to(device)
    tmp2 = torch.sin((torch.pi * 6 * t).to(device))


    return (u_tt[0].to(device) + u_xx[0].to(device) + 6 * u) - tmp1 * tmp2

if __name__ == "__main__":
    # 1. Data Process
    noise = 0.0
    N_u = 100
    N_f = 10000
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    yy = np.outer(y, np.ones(100))
    xx = np.outer(x, np.ones(100)).T
    Exact = np.sin(np.pi * xx)* np.sin(6 * np.pi * yy)
    X_star = np.hstack((xx.flatten()[: ,None], yy.flatten()[: ,None]))
    u_star = Exact.flatten()[: ,None]

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)
    xx1 = np.hstack((xx[0:1 ,:].T, yy[0:1 ,:].T))
    uu1 = Exact[0:1 ,:].T
    xx2 = np.hstack((xx[: ,0:1], yy[: ,0:1]))
    uu2 = Exact[: ,0:1]
    xx3 = np.hstack((xx[: ,-1:], yy[: ,-1:]))
    uu3 = Exact[: ,-1:]
    xx4 = np.hstack((xx[-1:, :].T, yy[-1:, :].T))
    uu4 = Exact[-1:, :].T

    X_u_train = np.vstack([xx1, xx2, xx3, xx4])
    u_train = np.vstack([uu1, uu2, uu3, uu4])
    X_f_train = lb + (ub -lb) * lhs(2, N_f)
    # X_f_train = np.vstack((X_f_train, X_u_train))
    # idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    # X_u_train = X_u_train[idx, :]
    # u_train = u_train[idx ,:]
    xxx = X_f_train[:, 0]
    ttt = X_f_train[:, 1]

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cpu")
    # 定义的神经网络
    net = Net(50, lb, ub)
    net.lb = net.lb.to(device)
    net.ub = net.ub.to(device)
    net = net.to(device)
    # net.apply(init_weights1)
    # 定义的损失函数
    mse_loss_function = torch.nn.MSELoss(reduction='mean')
    mse_loss_function = mse_loss_function.to(device)
    # 定义的优化器
    # optimizer_l_bfgs = torch.optim.LBFGS(net.parameters(),
    #                                      max_iter = 50000,
    #                                      max_eval = 50000,
    #                                      tolerance_grad = 1e-7,
    #                                      tolerance_change = 1.0 * np.finfo(float).eps)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    iteration = 30000
    x0 = torch.tensor(X_u_train, dtype=torch.float32)
    y0_r = torch.tensor(u_train, dtype=torch.float32)
    ca_all_zeros = autograd.Variable(torch.from_numpy(np.zeros((10000, 1))).float(), requires_grad=False)

    x0 = x0.to(device)
    y0_r = y0_r.to(device)
    ca_all_zeros = ca_all_zeros.to(device)
    loss_sum = []
    start_time = time.time()

    for epoch in range(iteration + 1):
        optimizer.zero_grad()
        y0 = net(x0)
        mse_b = mse_loss_function(y0, y0_r)

        ca_out = PDE(xxx, ttt, net)
        mse_f = mse_loss_function(ca_out, ca_all_zeros)

        loss = 10000 * mse_b + mse_f
        # torch.log(net.eb * net.ec * net.ei)
        # print(net.eb, net.ec, net.ei)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(epoch, "Traning Loss:", loss.data)
            print(f'times {epoch}  -  loss: {loss.item()}')
        loss_sum.append([epoch + 1, loss.item()])
    elapsed = time.time() - start_time
    # writer = SummaryWriter("./log/test")
    # for i in range(len(loss_sum)):
    #     writer.add_scalar("loss", loss_sum[i], i + 1)
    # writer.flush()
    # writer.close()
    loss_sum = np.array(loss_sum)
    np.savetxt("./log/PINN_loss.csv", loss_sum, delimiter=',')
    # torch.save(net.state_dict(), "./Model_pinn")
    # net.load_state_dict(torch.load("Model_pinn"))
    X_star = torch.tensor(X_star, dtype = torch.float32)
    X_star_GPU = X_star.to(device)
    u_pred = net(X_star_GPU)
    u_pred = u_pred.cpu()
    error_u = np.linalg.norm(u_star - u_pred.detach().numpy(), 2) / np.linalg.norm(u_star, 2)
    print('Error u: %e' % (error_u))
    print(elapsed / 60, elapsed % 60)
    #
    # fig = plt.figure()
    # u_pred = u_pred.detach().numpy()
    # u_pred = np.reshape(u_pred, (100, 100))
    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    # ax.plot_surface(xx, yy, Exact, cmap='viridis', edgecolor='none')
    # ax.set_title('Helmholtz-AS')
    # bx = fig.add_subplot(1, 2, 2, projection='3d')
    # bx.plot_surface(xx, yy, u_pred, cmap='viridis', edgecolor='none')
    # bx.set_title('Helmholtz-PS')
    # plt.show()
