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
        self.output_layer = nn.Linear(NN, 1)
        self.lb = lb
        self.lb = self.lb.astype(np.float32)
        self.lb = torch.tensor(self.lb)
        self.ub = ub
        self.ub = self.ub.astype(np.float32)
        self.ub = torch.tensor(self.ub)
    def forward(self, x):
        out1 = torch.tanh(self.input_layer(x))
        out2 = torch.tanh(self.h1_layer(out1))
        out_final = self.output_layer(out2)
        return out_final
def PDE(x, t, net):
    x = torch.from_numpy(x)
    x = torch.tensor(x, dtype = torch.float32)
    t = torch.from_numpy(t)
    t = torch.tensor(t, dtype = torch.float32)
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
    pi_ = torch.pi
    tmp1 = torch.sin(t * pi_ * 2).to(device)
    tmp2 = (4 * (torch.pi * torch.pi * (x * (x - 1))) - 2).to(device)
    r = -1 * (u_tt[0].to(device) + u_xx[0].to(device)) - tmp1 * tmp2
    return r
if __name__ == "__main__":
    # 1. Data Process
    noise = 0.0
    N_u = 100
    N_f = 20000
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    yy = np.outer(y, np.ones(100))
    xx = np.outer(x, np.ones(100)).T
    Exact = xx * (xx - 1) * np.sin(2 * np.pi * yy) + 1
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
    xxx = X_f_train[:, 0:1]
    ttt = X_f_train[:, 1:2]
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 定义的神经网络
    net = Net(90, lb, ub)
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
    ca_all_zeros = autograd.Variable(torch.from_numpy(np.zeros((20000, 1))).float(), requires_grad=False)

    x0 = x0.to(device)
    y0_r = y0_r.to(device)
    ca_all_zeros = ca_all_zeros.to(device)
    start_time = time.time()
    for epoch in range(iteration + 1):
        optimizer.zero_grad()
        y0 = net(x0)
        mse_b = mse_loss_function(y0, y0_r)
        ca_out = PDE(xxx, ttt, net)
        mse_f = mse_loss_function(ca_out, ca_all_zeros)
        loss = 100 * mse_b + mse_f
        # torch.log(net.eb * net.ec * net.ei)
        # print(net.eb, net.ec, net.ei)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(epoch, "Traning Loss:", loss.data)
            print(f'times {epoch}  -  loss: {loss.item()}')
    elapsed = time.time() - start_time
    X_star = torch.tensor(X_star, dtype = torch.float32)
    X_star_GPU = X_star.to(device)
    u_pred = net(X_star_GPU)
    # u_pred = u_pred.cpu()
    error_u = np.linalg.norm(u_star - u_pred.detach().numpy(), 2) / np.linalg.norm(u_star, 2)
    print('Error u: %e' % (error_u))
    print(elapsed / 60, elapsed % 60)
    #
    # fig = plt.figure()
    # u_pred = u_pred.detach().numpy()
    # u_pred = np.reshape(u_pred, (100, 100))
    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    # ax.plot_surface(xx, yy, u_pred, cmap='viridis', edgecolor='none')
    # ax.set_title('Possion-AS')
    # bx = fig.add_subplot(1, 2, 2, projection='3d')
    # bx.plot_surface(xx, yy, Exact, cmap='viridis', edgecolor='none')
    # bx.set_title('Possion-PS')
    # plt.savefig('Possion_MindSpore.pdf')
    # plt.savefig('Possion_MindSpore.eps')
    # U_pred = griddata(X_star.numpy(), u_pred.detach().numpy().flatten(), (xx, yy), method='cubic')
    # Error = np.abs(Exact - U_pred)
    # ######################################################################
    # ############################# Plotting ###############################
    # ######################################################################
    #
    # fig, ax = newfig(1.0, 1.1)
    # ax.axis('off')
    #
    # ####### Row 0: u(t,x) ##################
    # gs0 = gridspec.GridSpec(1, 2)
    # gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.1, right=0.9, wspace=0.5)
    # ax = plt.subplot(gs0[0, 0])
    #
    # h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
    #               extent=[y.min(), y.max(), x.min(), x.max()],
    #               origin='lower', aspect='auto')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="4%", pad=0.05)
    # fig.colorbar(h, cax=cax)
    # ax.set_title('$u(t,x)$', fontsize=10)
    # # ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (u_train.shape[0]), markersize=4,
    # #        clip_on=False)
    #
    # # line = np.linspace(x.min(), x.max(), 2)[:, None]
    # # ax.plot(t[25] * np.ones((2, 1)), line, 'w-', linewidth=1)
    # # ax.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
    # # ax.plot(t[75] * np.ones((2, 1)), line, 'w-', linewidth=1)
    #
    # ax.set_xlabel('$t$')
    # ax.set_ylabel('$x$')
    # ax.legend(frameon=False, loc='upper left')
    #
    #
    # ax = plt.subplot(gs0[0, 1])
    #
    # h = ax.imshow(Exact.T, interpolation='nearest', cmap='rainbow',
    #               extent=[y.min(), y.max(), x.min(), x.max()],
    #               origin='lower', aspect='auto')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="4%", pad=0.05)
    # fig.colorbar(h, cax=cax)
    # ax.set_title('$u(t,x)$', fontsize=10)
    # # ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (u_train.shape[0]), markersize=4,
    # #        clip_on=False)
    #
    # # line = np.linspace(x.min(), x.max(), 2)[:, None]
    # # ax.plot(t[25] * np.ones((2, 1)), line, 'w-', linewidth=1)
    # # ax.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
    # # ax.plot(t[75] * np.ones((2, 1)), line, 'w-', linewidth=1)
    #
    # ax.set_xlabel('$t$')
    # ax.set_ylabel('$x$')
    # ax.legend(frameon=False)
    #
    # ####### Row 1: u(t,x) slices ##################
    # gs1 = gridspec.GridSpec(1, 3)
    # gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    #
    # ax = plt.subplot(gs1[0, 0])
    # ax.plot(x, Exact[25, :], 'b-', linewidth=2, label='Exact')
    # ax.plot(x, U_pred[25, :], 'r--', linewidth=2, label='Prediction')
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$u(t,x)$')
    # ax.set_title('$t = 0.25$', fontsize=10)
    # ax.axis('square')
    # ax.set_xlim([-1.1, 1.1])
    # ax.set_ylim([-1.1, 1.1])
    #
    # ax = plt.subplot(gs1[0, 1])
    # ax.plot(x, Exact[50, :], 'b-', linewidth=2, label='Exact')
    # ax.plot(x, U_pred[50, :], 'r--', linewidth=2, label='Prediction')
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$u(t,x)$')
    # ax.axis('square')
    # ax.set_xlim([-1.1, 1.1])
    # ax.set_ylim([-1.1, 1.1])
    # ax.set_title('$t = 0.50$', fontsize=10)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
    #
    # ax = plt.subplot(gs1[0, 2])
    # ax.plot(x, Exact[75, :], 'b-', linewidth=2, label='Exact')
    # ax.plot(x, U_pred[75, :], 'r--', linewidth=2, label='Prediction')
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$u(t,x)$')
    # ax.axis('square')
    # ax.set_xlim([-1.1, 1.1])
    # ax.set_ylim([-1.1, 1.1])
    # ax.set_title('$t = 0.75$', fontsize=10)
    #
    # savefig('./figures/Burgers_pinn')