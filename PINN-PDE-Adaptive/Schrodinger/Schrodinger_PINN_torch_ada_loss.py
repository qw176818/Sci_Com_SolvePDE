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
from Schrodinger_PINN_torch import Net as NetPINN
np.random.seed(1234)
class Net(nn.Module):
    def __init__(self, NN, lb, ub):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(2, NN)
        self.h1_layer = nn.Linear(NN, NN)
        self.h2_layer = nn.Linear(NN, NN)
        self.output_layer = nn.Linear(NN, 2)
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
        self.ei = nn.Parameter(torch.tensor([1.0], requires_grad = True))
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
    U = net(X)
    u = U[:, 0:1]
    v = U[:, 1:2]

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

    v_t = autograd.grad(outputs = v,
                        inputs = t,
                        grad_outputs = torch.ones_like(v),
                        retain_graph = True,
                        create_graph = True,
                        allow_unused = True)
    v_t2 = torch.stack(v_t, 1).to(device)
    v_x = autograd.grad(outputs = v,
                        inputs = x,
                        grad_outputs = torch.ones_like(v),
                        retain_graph = True,
                        create_graph = True,
                        allow_unused = True)
    v_x2 = torch.stack(v_x, 1).to(device)
    v_xx = autograd.grad(outputs = v_x,
                        inputs = x,
                        grad_outputs = torch.ones_like(v_x[0]),
                        retain_graph = True,
                        create_graph = True,
                        allow_unused = True)
    v_xx2 = torch.stack(v_xx, 1).to(device)
    f_u = u_t2 + 0.5 * v_xx2 + (u * u + v * v) * v
    f_v = v_t2 - 0.5 * u_xx2 - (u * u + v * v) * u
    return f_u, f_v
def Make_b_mse(xb1, xb2, tb, net):
    xb1 = torch.tensor(xb1, dtype = torch.float32, requires_grad=True).to(device)
    xb2 = torch.tensor(xb2, dtype = torch.float32, requires_grad=True).to(device)
    tb = torch.tensor(tb, dtype = torch.float32, requires_grad=True).to(device)
    vb1 = torch.concat((xb1, tb), dim=1)
    vb2 = torch.concat((xb2, tb), dim=1)
    h1 = net(vb1)
    h2 = net(vb2)
    u1 = h1[:, 0]
    v1 = h1[:, 1]
    u2 = h2[:, 0]
    v2 = h2[:, 1]
    hx1 = autograd.grad(outputs = u1,
                        inputs = xb1,
                        grad_outputs = torch.ones_like(u1),
                        retain_graph = True,
                        create_graph = True,
                        allow_unused = True)
    hx1 = hx1[0].to(device)

    hx2 = autograd.grad(outputs = u2,
                        inputs = xb2,
                        grad_outputs = torch.ones_like(u2),
                        retain_graph = True,
                        create_graph = True,
                        allow_unused = True)
    hx2 = hx2[0].to(device)
    return h1, h2, hx1, hx2
if __name__ == "__main__":
    noise = 0.0

    # Doman bounds
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi / 2])

    N0 = 50
    N_b = 50
    N_f = 20000

    data = scipy.io.loadmat('./NLS.mat')

    t = data['tt'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u ** 2 + Exact_v ** 2)
    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact_u.T.flatten()[:, None]
    v_star = Exact_v.T.flatten()[:, None]
    h_star = Exact_h.T.flatten()[:, None]

    ###########################
    # t = 0 随机选取50个点
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x, :]
    u0 = Exact_u[idx_x, 0:1]
    v0 = Exact_v[idx_x, 0:1]
    t0 = np.zeros([50, 1])
    X_u_train = np.concatenate((x0, t0), axis = 1)
    u_train = np.concatenate((u0, v0), axis = 1)
    # x 边界随机选取50个点
    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t, :]
    xb1 = -5 * np.ones([50, 1])
    xb2 = 5 * np.ones([50, 1])

    # 可行域内部随机选取20000个点
    X_f = lb + (ub - lb) * lhs(2, N_f)

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
    x0 = torch.tensor(X_u_train, dtype=torch.float32)
    y0_r = torch.tensor(u_train, dtype=torch.float32)
    ca_all_zeros = autograd.Variable(torch.from_numpy(np.zeros((N_f, 1))).float(), requires_grad=False)
    x0 = x0.to(device)
    y0_r = y0_r.to(device)
    ca_all_zeros = ca_all_zeros.to(device)
    # loss_sum = []
    # for epoch in range(iteration + 1):
    #     optimizer.zero_grad()
    #     y0 = net(x0)
    #     mse_i = mse_loss_function(y0, y0_r)
    #     h1, h2, hx1, hx2 = Make_b_mse(xb1, xb2, tb, net)
    #     mse_b1 = mse_loss_function(h1, h2)
    #     mse_b2 = mse_loss_function(hx1, hx2)
    #     mse_b = mse_b1 + mse_b2
    #     f_u, f_v = PDE(X_f[:, 0], X_f[:, 1], net)
    #     mse_f1 = mse_loss_function(f_u, ca_all_zeros)
    #     mse_f2 = mse_loss_function(f_v, ca_all_zeros)
    #     mse_f = mse_f1 + mse_f2
    #     loss = 1 / (2 * net.eb * net.eb) * mse_b + 1 / (2 * net.ei * net.ei) * mse_i + \
    #            1 / (2 * net.ec * net.ec) * mse_f
    #     loss.backward()
    #     optimizer.step()
    #
    #     if epoch % 100 == 0:
    #         print(epoch, "Traning Loss:", loss.data)
    #         print(f'times {epoch}  -  loss: {loss.item()}')
    #     loss_sum.append([epoch + 1, loss.item()])
    # loss_sum = np.array(loss_sum)
    # np.savetxt("./log/PINN_Ada_Loss_Fun_loss.csv", loss_sum, delimiter=',')
    # torch.save(net.state_dict(), "./model_pinn_ada_funloss")
    net.load_state_dict(torch.load("./model_pinn_ada_funloss"))
    X_star = torch.tensor(X_star, dtype=torch.float32)
    X_star_GPU = X_star.to(device)
    pred = net(X_star_GPU)
    u_pred = pred[:, 0:1].cpu().detach().numpy()
    v_pred = pred[:, 1:2].cpu().detach().numpy()
    h_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_h = np.linalg.norm(h_star - h_pred, 2) / np.linalg.norm(h_star, 2)
    mae_u_error = np.sum(np.abs(u_star - u_pred)) / np.size(u_star)
    rmse_u_error = np.linalg.norm(u_star - u_pred, 2) / np.sqrt(np.size(u_star))
    mae_v_error = np.sum(np.abs(v_star - v_pred)) / np.size(v_star)
    rmse_v_error = np.linalg.norm(v_star - v_pred, 2) / np.sqrt(np.size(v_star))
    mae_h_error = np.sum(np.abs(h_star - h_pred) / np.size(h_star))
    rmse_h_error = np.linalg.norm(h_star - h_pred, 2) / np.sqrt(np.size(h_star))
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error h: %e' % (error_h))

    print('Error u: %e' % (mae_u_error))
    print('Error v: %e' % (mae_v_error))
    print('Error h: %e' % (mae_h_error))


    print('Error u: %e' % (rmse_u_error))
    print('Error v: %e' % (rmse_v_error))
    print('Error h: %e' % (rmse_h_error))
    net1 = NetPINN(50, lb, ub)
    net1.to(device)
    net1.lb = net1.lb.to(device)
    net1.ub = net1.ub.to(device)
    net1.load_state_dict(torch.load("./model_pinn"))

    pred_pinn = net1(X_star_GPU)
    u_pred_pinn = pred_pinn[:, 0:1].cpu().detach().numpy()
    v_pred_pinn = pred_pinn[:, 1:2].cpu().detach().numpy()
    h_pred_pinn = np.sqrt(u_pred_pinn ** 2 + v_pred_pinn ** 2)


    # FU_pred = griddata(X_star, f_u_pred.flatten(), (X, T), method='cubic')
    # FV_pred = griddata(X_star, f_v_pred.flatten(), (X, T), method='cubic')
    u_pred = np.reshape(u_pred, (201, 256))
    v_pred = np.reshape(v_pred, (201, 256))
    h_pred = np.reshape(h_pred, (201, 256))
    u_pred_pinn = np.reshape(u_pred_pinn, (201, 256))
    v_pred_pinn = np.reshape(v_pred_pinn, (201, 256))
    h_pred_pinn = np.reshape(h_pred_pinn, (201, 256))
    Error_u = abs(Exact_u.T - u_pred)
    Error_v = abs(Exact_v.T - v_pred)
    Error_h = abs(Exact_h.T - h_pred)
    Error_u_pinn = abs(Exact_u.T - u_pred_pinn)
    Error_v_pinn = abs(Exact_v.T - v_pred_pinn)
    Error_h_pinn = abs(Exact_h.T - h_pred_pinn)

    # ## AdaPINN的真实解与解析解三维图像
    # # fig, ax = newfig(2.0, 0.9)
    # # ax.axis('off')
    # # gs1 = gridspec.GridSpec(2, 3)
    # # ax = plt.subplot(gs1[0, 0], projection = '3d')
    # # # ax = fig.add_subplot(2, 3, 1, projection='3d')
    # # ax.plot_surface(X, T, Exact_u.T, cmap='viridis', edgecolor='none')
    # # ax.set_title('Schrodinger-AS-u')
    # # ax = plt.subplot(gs1[0, 1], projection = '3d')
    # # # ax1 = fig.add_subplot(2, 3, 2, projection='3d')
    # # ax.plot_surface(X, T, Exact_v.T, cmap='viridis', edgecolor='none')
    # # ax.set_title('Schrodinger-AS-v')
    # # ax = plt.subplot(gs1[0, 2], projection = '3d')
    # # # ax2 = fig.add_subplot(2, 3, 3, projection='3d')
    # # ax.plot_surface(X, T, Exact_h.T, cmap='viridis', edgecolor='none')
    # # ax.set_title('Schrodinger-AS-h')
    # # # bx = fig.add_subplot(2, 3, 4, projection='3d')
    # # ax = plt.subplot(gs1[1, 0], projection = '3d')
    # # ax.plot_surface(X, T, u_pred.T, cmap='viridis', edgecolor='none')
    # # ax.set_title('Schrodinger-PS-u')
    # # # bx1 = fig.add_subplot(2, 3, 5, projection='3d')
    # # ax = plt.subplot(gs1[1, 1], projection = '3d')
    # # ax.plot_surface(X, T, v_pred.T, cmap='viridis', edgecolor='none')
    # # ax.set_title('Schrodinger-PS-v')
    # # # bx2 = fig.add_subplot(2, 3, 6, projection='3d')
    # # ax = plt.subplot(gs1[1, 2], projection = '3d')
    # # ax.plot_surface(X, T, h_pred.T, cmap='viridis', edgecolor='none')
    # # ax.set_title('Schrodinger-PS-h')
    # # savefig("./figures/img_predict")
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################
    # x0 = x0.cpu().detach().numpy()
    # X0 = np.concatenate((x0, 0 * x0), 1)  # (x0, 0)
    # X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
    # X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)
    # X_u_train = np.vstack([X0, X_lb, X_ub])

    # # Error可视化
    # Error_u_ = griddata(X_star, Error_u_pinn.flatten(), (X, T), method='cubic')
    # Error_v_ = griddata(X_star, Error_v_pinn.flatten(), (X, T), method='cubic')
    # Error_h_ = griddata(X_star, Error_h_pinn.flatten(), (X, T), method='cubic')
    # Error_U_pred = griddata(X_star, Error_u.flatten(), (X, T), method='cubic')
    # Error_V_pred = griddata(X_star, Error_v.flatten(), (X, T), method='cubic')
    # Error_h_pred = griddata(X_star, Error_h.flatten(), (X, T), method='cubic')
    # fig1, ax = newfig(2.5, 1.5)
    # ax.axis('off')
    # gs0 = gridspec.GridSpec(3, 2)
    # ax = plt.subplot(gs0[0, 0])
    # ax.set_title('Error U-PINN')
    # h = ax.imshow(Error_u_, interpolation='nearest', cmap='YlGnBu',
    #               extent=[lb[1], ub[1], lb[0], ub[0]],
    #               origin='lower', aspect='auto')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig1.colorbar(h, cax=cax)
    # ax = plt.subplot(gs0[0, 1])
    # ax.set_title('Error U-AdaPINN')
    # h = ax.imshow(Error_U_pred, interpolation='nearest', cmap='YlGnBu',
    #               extent=[lb[1], ub[1], lb[0], ub[0]],
    #               origin='lower', aspect='auto')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig1.colorbar(h, cax=cax)
    # ax = plt.subplot(gs0[1, 0])
    # ax.set_title('Error V-PINN')
    # h = ax.imshow(Error_v_, interpolation='nearest', cmap='YlGnBu',
    #               extent=[lb[1], ub[1], lb[0], ub[0]],
    #               origin='lower', aspect='auto')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig1.colorbar(h, cax=cax)
    # ax = plt.subplot(gs0[1, 1])
    # ax.set_title('Error V-AdaPINN')
    # h = ax.imshow(Error_V_pred, interpolation='nearest', cmap='YlGnBu',
    #               extent=[lb[1], ub[1], lb[0], ub[0]],
    #               origin='lower', aspect='auto')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig1.colorbar(h, cax=cax)
    # ax = plt.subplot(gs0[2, 0])
    # ax.set_title('Error H-PINN')
    # h = ax.imshow(Error_h_, interpolation='nearest', cmap='YlGnBu',
    #               extent=[lb[1], ub[1], lb[0], ub[0]],
    #               origin='lower', aspect='auto')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig1.colorbar(h, cax=cax)
    # ax = plt.subplot(gs0[2, 1])
    # ax.set_title('Error H-AdaPINN')
    # h = ax.imshow(Error_h_pred, interpolation='nearest', cmap='YlGnBu',
    #               extent=[lb[1], ub[1], lb[0], ub[0]],
    #               origin='lower', aspect='auto')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig1.colorbar(h, cax=cax)
    # savefig('./figures/Error_predict')


    Exact_u = griddata(X_star, Exact_u.T.flatten(), (X, T), method='cubic')
    Exact_v = griddata(X_star, Exact_v.T.flatten(), (X, T), method='cubic')
    Exact_h = griddata(X_star, Exact_h.T.flatten(), (X, T), method='cubic')
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    V_pred = griddata(X_star, v_pred.flatten(), (X, T), method='cubic')
    H_pred = griddata(X_star, h_pred.flatten(), (X, T), method='cubic')
    fig, ax = newfig(2.5, 1.5)
    ax.axis('off')
    gs0 = gridspec.GridSpec(3, 2)
    ax = plt.subplot(gs0[0, 0])
    ax.set_title('Exact U')
    h = ax.imshow(Exact_u.T, interpolation='nearest', cmap='rainbow',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax = plt.subplot(gs0[0, 1])
    ax.set_title('Predict U')

    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax = plt.subplot(gs0[1, 0])
    ax.set_title('Exact V')
    h = ax.imshow(Exact_v.T, interpolation='nearest', cmap='YlGnBu',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax = plt.subplot(gs0[1, 1])
    ax.set_title('Predict V')
    h = ax.imshow(V_pred.T, interpolation='nearest', cmap='YlGnBu',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax = plt.subplot(gs0[2, 0])
    ax.set_title('Exact H')
    h = ax.imshow(Exact_h.T, interpolation='nearest', cmap='rainbow',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax = plt.subplot(gs0[2, 1])
    ax.set_title('Predict H')
    h = ax.imshow(H_pred.T, interpolation='nearest', cmap='rainbow',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    savefig('./figures/NLS_predict')

    # ####### Row 0: h(t,x) ##################
    gs0 = gridspec.GridSpec(1, 1)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(H_pred.T, interpolation='nearest', cmap='rainbow',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    # ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=4,
    #         clip_on=False)

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[50] * np.ones((2, 1)), line, 'k--', linewidth=1)
    ax.plot(t[100] * np.ones((2, 1)), line, 'k--', linewidth=1)
    ax.plot(t[150] * np.ones((2, 1)), line, 'k--', linewidth=1)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    leg = ax.legend(frameon=False, loc='best')
    #    plt.setp(leg.get_texts(), color='w')
    ax.set_title('$|h(t,x)|$', fontsize=10)

    ####### Row 1: h(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, Exact_h[50, ], 'b-', linewidth=2, label='Exact')
    ax.plot(x, H_pred[50, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.set_title('$t = %.2f$' % (t[50]), fontsize=10)
    ax.axis('square')
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, Exact_h[100, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, H_pred[100, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])
    ax.set_title('$t = %.2f$' % (t[100]), fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])

    ax.plot(x, Exact_h[150, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, H_pred[150, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])
    ax.set_title('$t = %.2f$' % (t[150]), fontsize=10)

    savefig('./figures/NLS')