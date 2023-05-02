import numpy as np
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
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
        self.output_layer = nn.Linear(NN, 1)
        self.a1 = nn.Parameter(torch.tensor([1.0], requires_grad = True))
        self.a2 = nn.Parameter(torch.tensor([1.0], requires_grad = True))
        self.a3 = nn.Parameter(torch.tensor([1.0], requires_grad = True))
        self.lb = lb
        self.lb = self.lb.astype(np.float32)
        self.lb = torch.tensor(self.lb)
        self.ub = ub
        self.ub = self.ub.astype(np.float32)
        self.ub = torch.tensor(self.ub)
        self.n = 1
        self.eb = nn.Parameter(torch.tensor([1.0], requires_grad = True))
        self.ec = nn.Parameter(torch.tensor([1.0], requires_grad = True))
    def forward(self, x):
        x1 = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        out1 = torch.tanh(self.n * self.a1 * self.input_layer(x1))
        out2 = torch.tanh(self.n * self.a2 * self.h1_layer(out1))
        out3 = torch.tanh(self.n * self.a3 * self.h2_layer(out2))
        out_final = self.output_layer(out3)
        return out_final
def init_weights1(model):
    if isinstance(model, nn.Linear):
        torch.nn.init.xavier_uniform(model.weight)
        model.bias.data.fill_(0)

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
    u_tt = autograd.grad(outputs = u_t,
                        inputs = t,
                        grad_outputs = torch.ones_like(u_t[0]),
                        retain_graph = True,
                        create_graph = True,
                        allow_unused = True)
    u_tt2 = torch.stack(u_tt, 1).to(device)
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
    pi = torch.pi
    tmp1 = torch.sin(pi * t).to(device)
    tmp2 = ((torch.pi * torch.pi * (x * (x - 1))) - 2).to(device)

    return -(u_tt2 + u_xx2) - tmp1 * tmp2

if __name__ == "__main__":
    # 1. Data Process
    noise = 0.0

    N_u = 100
    N_f = 10000


    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    yy = np.outer(y, np.ones(100))
    xx = np.outer(x, np.ones(100)).T
    Exact = xx * (xx - 1) * np.sin(np.pi * yy)
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
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        loss = 1 / (2 * net.eb * net.eb) * mse_b + 1 / (2 * net.ec * net.ec) * mse_f

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
    np.savetxt("./log/PINN_loss_ada_fun_loss.csv", loss_sum, delimiter=',')
    torch.save(net.state_dict(), "./Model_pinn_ada_fun_loss")
    net.load_state_dict(torch.load("./Model_pinn_ada_fun_loss"))
    X_star = torch.tensor(X_star, dtype = torch.float32)
    X_star_GPU = X_star.to(device)
    u_pred = net(X_star_GPU)
    u_pred = u_pred.cpu()
    error_u = np.linalg.norm(u_star - u_pred.detach().numpy(), 2) / np.linalg.norm(u_star, 2)
    print('Error u: %e' % (error_u))
    # print(elapsed)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    u_pred = u_pred.detach().numpy()
    u_pred = np.reshape(u_pred, (100, 100))
    ax.plot_surface(x, y, u_pred, cmap='viridis', edgecolor='none')
    ax.set_title('Possion-AS')
    plt.show()