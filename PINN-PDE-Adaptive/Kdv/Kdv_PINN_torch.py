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
import math
from torch.utils.tensorboard import SummaryWriter
np.random.seed(1234)
class Net(nn.Module):
    def __init__(self, NN, lb, ub):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(2, NN)
        self.h1_layer = nn.Linear(NN, NN)
        self.h2_layer = nn.Linear(NN, NN)
        self.output_layer = nn.Linear(NN, 1)
        self.lb = lb
        self.lb = self.lb.astype(np.float32)
        self.lb = torch.tensor(self.lb)
        self.ub = ub
        self.ub = self.ub.astype(np.float32)
        self.ub = torch.tensor(self.ub)
        # self.a1 = nn.Parameter(torch.tensor([1.0], requires_grad = True))
        # self.a2 = nn.Parameter(torch.tensor([1.0], requires_grad = True))
        # self.a3 = nn.Parameter(torch.tensor([1.0], requires_grad = True))
        self.eb = nn.Parameter(torch.tensor([1.0], requires_grad = True))
        self.ei = nn.Parameter(torch.tensor([1.0], requires_grad = True))
        self.ec = nn.Parameter(torch.tensor([1.0], requires_grad = True))
    def forward(self, x):
        x1 = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        out1 = torch.tanh(self.input_layer(x1))
        out2 = torch.tanh(self.h1_layer(out1))
        out3 = torch.tanh(self.h2_layer(out2))
        out_final = self.output_layer(out3)
        return out_final
def init_weights1(model):
    if isinstance(model, nn.Linear):
        torch.nn.init.xavier_uniform(model.weight)
        model.bias.data.fill_(0)
def f(x, t):
    c = 3
    x0 = 0
    u = (math.sqrt(c) / 2) * (x - x0 - c * t)
    u1 = 2.0 / (np.exp(u) + np.exp(-u))
    u2 = (c / 2) * u1 * u1
    return u2

def PDE(x, t, net):
    x = torch.from_numpy(x)
    x = torch.tensor(x, dtype = torch.float32)
    t = torch.from_numpy(t)
    t = torch.tensor(t, dtype = torch.float32)
    x.requires_grad_(True)
    t.requires_grad_(True)
    X = torch.stack((x, t), 1)
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
    u_t2 = torch.stack(u_t, 1).to(device)
    u_x = autograd.grad(outputs = u,
                        inputs = x,
                        grad_outputs = torch.ones_like(u),
                        retain_graph = True,
                        create_graph = True,
                        allow_unused = True)
    u_x2 = torch.stack(u_x, 1).to(device)
    u_xx = autograd.grad(outputs = u_x,
                        inputs = x,
                        grad_outputs = torch.ones_like(u_x[0]),
                        retain_graph = True,
                        create_graph = True,
                        allow_unused = True)
    u_xx2 = torch.stack(u_xx, 1).to(device)
    u_xxx = autograd.grad(outputs = u_xx,
                        inputs = x,
                        grad_outputs = torch.ones_like(u_xx[0]),
                        retain_graph = True,
                        create_graph = True,
                        allow_unused = True)
    u_xxx2 = torch.stack(u_xxx, 1).to(device)
    return u_t2 + 6 * u * u_x2 + u_xxx2

if __name__ == "__main__":
    # 1. Data Process
    N_u = 200
    N_f = 50000
    t = np.linspace(0, 5, 500)
    x = np.linspace(-20, 20, 1000)
    T = np.outer(t, np.ones(1000))
    X = np.outer(x, np.ones(500)).T
    Exact = f(X, T)
    X_star = np.hstack((X.flatten()[: ,None], T.flatten()[: ,None]))
    u_star = Exact.flatten()[: ,None]
    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)
    xx1 = np.hstack((X[0:1 ,:].T, T[0:1, :].T))
    uu1 = Exact[0:1, :].T
    xx2 = np.hstack((X[: ,0:1], T[: ,0:1]))
    uu2 = Exact[: ,0:1]
    xx3 = np.hstack((X[: ,-1:], T[: ,-1:]))
    uu3 = Exact[: ,-1:]

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_u_train = np.vstack([xx2, xx3])
    u_train = np.vstack([uu2, uu3])
    X_b_train = xx1
    u_b_train = uu1
    X_f_train = lb + (ub -lb) *lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx ,:]
    xx = X_f_train[:, 0]
    tt = X_f_train[:, 1]
    x0 = torch.tensor(X_u_train, dtype=torch.float32)
    y0_r = torch.tensor(u_train, dtype=torch.float32)
    x1 = torch.tensor(X_b_train, dtype=torch.float32)
    y1_r = torch.tensor(u_b_train, dtype=torch.float32)
    ca_all_zeros = autograd.Variable(torch.from_numpy(np.zeros((51000, 1))).float(), requires_grad=False)
    x0 = x0.to(device)
    y0_r = y0_r.to(device)
    x1 = x1.to(device)
    y1_r = y1_r.to(device)
    ca_all_zeros = ca_all_zeros.to(device)

    # 定义的神经网络
    net = Net(50, lb, ub)
    net.lb = net.lb.to(device)
    net.ub = net.ub.to(device)
    net = net.to(device)
    net.apply(init_weights1)
    # 定义的损失函数
    mse_loss_function = torch.nn.MSELoss(reduction='mean')
    mse_loss_function = mse_loss_function.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
    iteration = 50000

    loss_sum = []
    start_time = time.time()
    for epoch in range(iteration + 1):
        optimizer.zero_grad()
        y0 = net(x0)
        mse_b = mse_loss_function(y0, y0_r)

        y1 = net(x1)
        mse_i = mse_loss_function(y1, y1_r)

        ca_out = PDE(xx, tt, net)
        mse_f = mse_loss_function(ca_out, ca_all_zeros)
        # loss = 1 / (2 * net.eb * net.eb) * mse_b + 1 / (2 * net.ei * net.ei) * mse_i + \
        #        1 / (2 * net.ec * net.ec) * mse_f
        loss = 2 * mse_b + 2 * mse_i + 0.2 * mse_f
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(epoch, "Traning Loss:", loss.data)
            print(f'times {epoch}  -  loss: {loss.item()}')
        loss_sum.append([epoch + 1, loss.item()])
    elapsed = time.time() - start_time
    loss_sum = np.array(loss_sum)
    np.savetxt("./log/PINN_loss.csv", loss_sum, delimiter=',')
    # torch.save(net.state_dict(), "./Model_pinn_ada_fun")
    # net.load_state_dict(torch.load("./Model_pinn_ada_fun"))
    # X_star = torch.tensor(X_star, dtype = torch.float32)
    # X_star_GPU = X_star.to(device)
    # u_pred = net(X_star_GPU)
    # u_pred = u_pred.cpu()
    # error_u = np.linalg.norm(u_star - u_pred.detach().numpy(), 2) / np.linalg.norm(u_star, 2)
    # mae_error = np.sum(np.abs(u_star - u_pred.detach().numpy())) / np.size(u_star)
    # rmse_error = np.linalg.norm(u_star - u_pred.detach().numpy(), 2) / np.sqrt(np.size(u_star))
    # print('Error u: %e' % (error_u))
    # print(mae_error)
    # print(rmse_error)
    # print(elapsed)
    # u_pred = np.reshape(u_pred.detach().numpy(), (500, 1000))

    # fig, ax = newfig(2.0, 0.9)
    # ax.axis('off')
    # gs1 = gridspec.GridSpec(1, 2)
    # ax = plt.subplot(gs1[0, 0], projection = '3d')
    # # ax = fig.add_subplot(2, 3, 1, projection='3d')
    # ax.plot_surface(X, T, Exact, cmap='viridis', edgecolor='none')
    # ax.set_title('Kdv-AS')
    # ax = plt.subplot(gs1[0, 1], projection = '3d')
    # # ax1 = fig.add_subplot(2, 3, 2, projection='3d')
    # ax.plot_surface(X, T, u_pred, cmap='viridis', edgecolor='none')
    # ax.set_title('Kdv-PS')
    # savefig("./figures/img_predict")
    # savefig("./figures/img_predict")

    # u_pred = u_star
    # U_pred = griddata(X_star.numpy(), u_pred.flatten(), (X, T), method='cubic')
    # Error = np.abs(Exact - U_pred)
    #
    # # ######################################################################
    # # ############################# Plotting ###############################
    # # ######################################################################
    #
    # fig, ax = newfig(1.0, 1.1)
    # ax.axis('off')
    #
    # ####### Row 0: u(t,x) ##################
    # gs0 = gridspec.GridSpec(1, 2)
    # gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.1, right=0.9, wspace=0.5)
    # ax = plt.subplot(gs0[0, 0])
    #
    # h = ax.imshow(Exact.T, interpolation='nearest', cmap='rainbow',
    #               extent=[t.min(), t.max(), x.min(), x.max()],
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
    # ax.set_xlabel('$t$')
    # ax.set_ylabel('$x$')
    # ax.legend(frameon=False, loc='upper left')
    #
    # ax = plt.subplot(gs0[0, 1])
    # h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
    #               extent=[t.min(), t.max(), x.min(), x.max()],
    #               origin='lower', aspect='auto')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="4%", pad=0.05)
    # fig.colorbar(h, cax=cax)
    # ax.set_title('$u(t,x)$', fontsize=10)
    # # ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (u_train.shape[0]), markersize=4,
    # #        clip_on=False)
    #
    # # line = np.linspace(x.min(), x.max(), 2)[:, None]
    # # ax.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
    # # ax.plot(t[100] * np.ones((2, 1)), line, 'w-', linewidth=1)
    # # ax.plot(t[150] * np.ones((2, 1)), line, 'w-', linewidth=1)
    #
    # ax.set_xlabel('$t$')
    # ax.set_ylabel('$x$')
    # ax.legend(frameon=False)
    # ####### Row 1: u(t,x) slices ##################
    # gs1 = gridspec.GridSpec(1, 3)
    #
    # gs1.update(top=1 - 2 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    # ax = plt.subplot(gs1[0, 0])
    # ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    # ax.plot(x, Exact[100, :], 'b-', linewidth=2, label='Exact')
    # ax.plot(x, U_pred[100, :], 'r--', linewidth=2, label='Prediction')
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$u(t,x)$')
    # ax.set_title('$t = 1.0$', fontsize=10)
    #
    # ax = plt.subplot(gs1[0, 1])
    # ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    # ax.plot(x, Exact[300, :], 'b-', linewidth=2, label='Exact')
    # ax.plot(x, U_pred[300, :], 'r--', linewidth=2, label='Prediction')
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$u(t,x)$')
    # ax.set_title('$t = 3.0$', fontsize=10)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
    #
    # ax = plt.subplot(gs1[0, 2])
    # # ax.axis('auto')
    # ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    # ax.plot(x, Exact[400, :], 'b-', linewidth=2, label='Exact')
    # ax.plot(x, U_pred[400, :], 'r--', linewidth=2, label='Prediction')
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$u(t,x)$')
    #
    # ax.set_title('$t = 4.0$', fontsize=10)
    #
    # savefig('./figures/Kdv_ada_ca_pinn')