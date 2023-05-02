import sys
sys.path.insert(0, '../../Utilities/')

import numpy as np
import matplotlib.pyplot as plt
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

class Net(nn.Module):
    def __init__(self, NN, lb, ub):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(2, NN)
        self.h1_layer = nn.Linear(NN, NN)
        self.h2_layer = nn.Linear(NN, NN)
        # self.h3_layer = nn.Linear(NN, NN)
        # self.h4_layer = nn.Linear(NN, NN)
        self.output_layer = nn.Linear(NN, 1)
        self.a1 = nn.Parameter(torch.tensor([1.0], requires_grad = True))
        self.a2 = nn.Parameter(torch.tensor([1.0], requires_grad = True))
        self.a3 = nn.Parameter(torch.tensor([1.0], requires_grad = True))
        # self.a4 = nn.Parameter(torch.tensor([1.0], requires_grad = True))
        # self.a5 = nn.Parameter(torch.tensor([1.0], requires_grad = True))
        self.lb = lb
        self.lb = self.lb.astype(np.float32)
        self.lb = torch.tensor(self.lb)

        self.ub = ub
        self.ub = self.ub.astype(np.float32)
        self.ub = torch.tensor(self.ub)
        # self.n = 1
        self.eb = nn.Parameter(torch.tensor([1.0], requires_grad = True))
        self.ei = nn.Parameter(torch.tensor([1.0], requires_grad = True))
        self.ec = nn.Parameter(torch.tensor([1.0], requires_grad = True))
    def forward(self, x):
        x1 = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        out1 = torch.tanh(self.a1 * self.input_layer(x1))
        out2 = torch.tanh(self.a2 * self.h1_layer(out1))
        out3 = torch.tanh(self.a3 * self.h2_layer(out2))
        # out4 = torch.tanh(self.a4 * self.h3_layer(out3))
        # out5 = torch.tanh(self.a5 * self.h4_layer(out4))
        out_final = self.output_layer(out3)
        return out_final

def init_weights1(model):
    if isinstance(model, nn.Linear):
        torch.nn.init.xavier_uniform(model.weight)
        model.bias.data.fill_(0)

def PDE(x, t, net):
    x = np.reshape(x, (102912, 1))
    t = np.reshape(t, (102912, 1))
    x = torch.from_numpy(x)
    x = torch.tensor(x, dtype = torch.float32)
    t = torch.from_numpy(t)
    t = torch.tensor(t, dtype = torch.float32)
    x.requires_grad_(True)
    t.requires_grad_(True)
    X = torch.concat((x, t), dim=1)
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
    u_t2 = u_t[0].to(device)
    u_x = autograd.grad(outputs = u,
                        inputs = x,
                        grad_outputs = torch.ones_like(u),
                        retain_graph = True,
                        create_graph = True,
                        allow_unused = True)
    u_x2 = u_x[0].to(device)
    u_xx = autograd.grad(outputs = u_x,
                        inputs = x,
                        grad_outputs = torch.ones_like(u_x[0]),
                        retain_graph = True,
                        create_graph = True,
                        allow_unused = True)
    u_xx2 = u_xx[0].to(device)
    return u_t2 - (0.0001) * u_xx2 + 5 * u * u * u - 5 * u

def PDE1(x, t, net):
    x.requires_grad_(True)
    t.requires_grad_(True)
    X = torch.concat((x, t), dim=1)
    net.lb = torch.tensor(net.lb, requires_grad = True)
    net.ub = torch.tensor(net.ub, requires_grad = True)
    X = X.to(device)
    u = net(X)
    u_x = autograd.grad(outputs = u,
                        inputs = x,
                        grad_outputs = torch.ones_like(u),
                        retain_graph = True,
                        create_graph = True,
                        allow_unused = True)
    u_x2 = u_x[0].to(device)
    return u_x2

if __name__ == "__main__":
    # 1. Data Process
    noise = 0.0

    N_u = 100
    N_f = 10000

    data = scipy.io.loadmat('AC.mat')

    t = data['tt'].flatten()[: ,None]
    x = data['x'].flatten()[: ,None]
    Exact = np.real(data['uu']).T

    X, T = np.meshgrid(x ,t)

    X_star = np.hstack((X.flatten()[: ,None], T.flatten()[: ,None]))
    u_star = Exact.flatten()[: ,None]

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)
    lb1 = np.array([-0.5, 0])
    ub1 = np.array([0.5, 1])
    # Initial Points
    X_i_train = np.hstack((X[-1:0 ,:].T, T[-1:0 ,:].T))
    u_i_train = Exact[0:1 ,:].T

    # Boundary Points
    X_b_train = np.hstack((X[: ,0:1], T[: ,0:1]))
    u_b_train = Exact[: ,0:1]

    idx = np.random.choice(X_i_train.shape[0], N_u, replace=False)
    X_i_train = X_i_train[idx, :]
    u_i_train = u_i_train[idx, :]

    idx = np.random.choice(X_b_train.shape[0], N_u, replace=False)
    X_b_train = X_b_train[idx, :]
    u_b_train = u_b_train[idx, :]

    X_f_train = X_star.copy()
    xx = X_f_train[:, 0]
    tt = X_f_train[:, 1]

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 定义的神经网络
    net = Net(50, lb, ub)
    net.lb = net.lb.to(device)
    net.ub = net.ub.to(device)
    net = net.to(device)
    net.apply(init_weights1)
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
    iteration = 50000

    x0 = torch.tensor(X_i_train, dtype=torch.float32)
    y0_r = torch.tensor(u_i_train, dtype=torch.float32)
    x1 = torch.tensor(X_b_train, dtype=torch.float32)
    y1_r = torch.tensor(u_b_train, dtype=torch.float32)
    ca_all_zeros = autograd.Variable(torch.from_numpy(np.zeros((102912, 1))).float(), requires_grad=False)

    x0 = x0.to(device)
    y0_r = y0_r.to(device)
    x1 = x1.to(device)
    y1_r = y1_r.to(device)
    ca_all_zeros = ca_all_zeros.to(device)
    # for epoch in range(iteration + 1):
    #     optimizer.zero_grad()
    #     y0 = net(x0)
    #     mse_i = mse_loss_function(y0, y0_r)
    #
    #     y1 = net(x1)
    #     x2 = x1[:, 0:1]
    #     t2 = x1[:, 1:2]
    #     dy1 = PDE1(x2, t2, net)
    #     x1[:, 0] = -x1[:, 0]
    #     y2 = net(x1)
    #     x3 = x1[:, 0:1]
    #     t3 = x1[:, 1:2]
    #     dy2 = PDE1(x3, t3, net)
    #     mse_b1 = mse_loss_function(y1, y2)
    #     mse_b2 = mse_loss_function(dy1, dy2)
    #     # x1[:, 0] = -x1[:, 0]
    #     # y1_x = PDE1(x1[:, 0], x1[:, 1], net)
    #     # y2_x = PDE1(-x1[:, 0], x1[:, 1], net)
    #     # mse_b3 = mse_loss_function(y1_x, y2_x）
    #     # X_f1_train = lb + (ub - lb) * lhs(2, N_f)
    #     # X_f2_train = lb + (ub - lb) * lhs(2, N_f)
    #     # # X_f1_train = lb1 + (ub1 - lb1) *lhs(2, 5000)
    #     # # X_f_train = np.concatenate((X_f_train, X_f1_train), axis= 0)
    #
    #     ca_out = PDE(xx, tt, net)
    #     mse_f = mse_loss_function(ca_out, ca_all_zeros)
    #
    #     loss = mse_b1  + mse_b2 + mse_i + mse_f
    #     # torch.log(net.eb * net.ec * net.ei)
    #     # print(net.eb, net.ec, net.ei)
    #     loss.backward()
    #     optimizer.step()
    #
    #     if epoch % 100 == 0:
    #         print(epoch, "Traning Loss:", loss.data)
    #         print(f'times {epoch}  -  loss: {loss.item()}')
    # torch.save(net.state_dict(), "./model_pinn")
    net.load_state_dict(torch.load("model_pinn"))


    X_star = torch.tensor(X_star, dtype = torch.float32)
    X_star_GPU = X_star.to(device)

    u_pred = net(X_star_GPU)
    u_pred = u_pred.cpu()

    error_u = np.linalg.norm(u_star - u_pred.detach().numpy(), 2) / np.linalg.norm(u_star, 2)
    mae_error = np.sum(np.abs(u_star - u_pred.detach().numpy())) / np.size(u_star)
    rmse_error = np.linalg.norm(u_star - u_pred.detach().numpy(), 2) / np.sqrt(np.size(u_star))
    print('Error u: %e' % (error_u))
    print(mae_error)
    print(rmse_error)
    u_pred = u_pred.detach().numpy()
    u_pred = np.reshape(u_pred, (201, 512))
    fig, ax = newfig(2.0, 0.9)
    ax.axis('off')
    # u_pred = Exact
    gs1 = gridspec.GridSpec(1, 2)
    ax = plt.subplot(gs1[0, 0], projection = '3d')
    # ax = fig.add_subplot(2, 3, 1, projection='3d')
    ax.plot_surface(X, T, Exact, cmap='viridis', edgecolor='none')
    ax.set_title('Allen-Cahn_clean-AS')
    ax = plt.subplot(gs1[0, 1], projection = '3d')
    # ax1 = fig.add_subplot(2, 3, 2, projection='3d')
    ax.plot_surface(X, T, u_pred, cmap='viridis', edgecolor='none')
    ax.set_title('Allen-Cahn_clean-PS')
    plt.show()
    # savefig("./figures/img_predict_adapinn+")


    U_pred = griddata(X_star.numpy(), u_pred.detach().numpy().flatten(), (X, T), method='cubic')
    Error = np.abs(Exact - U_pred)
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################

    fig, ax = newfig(1.0, 1.1)
    ax.axis('off')

    ####### Row 0: u(t,x) ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.1, right=0.9, wspace=0.5)
    ax = plt.subplot(gs0[0, 0])

    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_title('$u(t,x)$', fontsize=10)
    # ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (u_train.shape[0]), markersize=4,
    #        clip_on=False)

    # line = np.linspace(x.min(), x.max(), 2)[:, None]
    # ax.plot(t[25] * np.ones((2, 1)), line, 'w-', linewidth=1)
    # ax.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
    # ax.plot(t[75] * np.ones((2, 1)), line, 'w-', linewidth=1)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc='upper left')


    ax = plt.subplot(gs0[0, 1])

    h = ax.imshow(Exact.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_title('$u(t,x)$', fontsize=10)
    # ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (u_train.shape[0]), markersize=4,
    #        clip_on=False)

    # line = np.linspace(x.min(), x.max(), 2)[:, None]
    # ax.plot(t[25] * np.ones((2, 1)), line, 'w-', linewidth=1)
    # ax.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
    # ax.plot(t[75] * np.ones((2, 1)), line, 'w-', linewidth=1)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False)

    ####### Row 1: u(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, Exact[25, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred[25, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = 0.25$', fontsize=10)
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, Exact[50, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred[50, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title('$t = 0.50$', fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, Exact[75, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred[75, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title('$t = 0.75$', fontsize=10)
    plt.show()
    # savefig('./figures/Allen-Cahn_pinn')