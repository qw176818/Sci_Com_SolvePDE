import math
import mindspore.ops
import numpy as np
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import context
from pyDOE import lhs
from mindspore.common.initializer import XavierUniform
import time
import matplotlib.pyplot as plt
import os
context.set_context(mode=context.PYNATIVE_MODE)
class NetMS(nn.Cell):
    def __init__(self, NN, lb, ub):
        super(NetMS, self).__init__()
        self.input_layer = nn.Dense(2, NN, weight_init = XavierUniform())
        self.h1_layer = nn.Dense(NN, NN, weight_init = XavierUniform())
        self.h2_layer = nn.Dense(NN, NN, weight_init = XavierUniform())
        self.h3_layer = nn.Dense(NN, NN, weight_init = XavierUniform())
        self.h4_layer = nn.Dense(NN, NN, weight_init = XavierUniform())
        self.h5_layer = nn.Dense(NN, NN, weight_init = XavierUniform())
        self.output_layer = nn.Dense(NN, 1, weight_init = XavierUniform())
        self.a1 = ms.Parameter(ms.Tensor([1.0], mindspore.float32))
        self.a2 = ms.Parameter(ms.Tensor([1.0], mindspore.float32))
        self.a3 = ms.Parameter(ms.Tensor([1.0], mindspore.float32))
        self.a4 = ms.Parameter(ms.Tensor([1.0], mindspore.float32))
        self.a5 = ms.Parameter(ms.Tensor([1.0], mindspore.float32))
        self.a6 = ms.Parameter(ms.Tensor([1.0], mindspore.float32))

        self.n = 1
        self.tanh = nn.Tanh()
        self.lb = lb
        self.lb = self.lb.astype(np.float32)
        self.lb = ms.Tensor(self.lb)
        self.ub = ub
        self.ub = self.ub.astype(np.float32)
        self.ub = ms.Tensor(self.ub)

        self.eb = ms.Parameter(ms.Tensor([1.0], mindspore.float32))
        self.ec = ms.Parameter(ms.Tensor([1.0], mindspore.float32))
    def construct(self, x):
        out1 = self.tanh(self.n * self.a1 * self.input_layer(x))
        out2 = self.tanh(self.n * self.a2 * self.h1_layer(out1))
        out3 = self.tanh(self.n * self.a3 * self.h2_layer(out2))
        out4 = self.tanh(self.n * self.a4 * self.h3_layer(out3))
        out5 = self.tanh(self.n * self.a5 * self.h4_layer(out4))
        out6 = self.tanh(self.n * self.a6 * self.h5_layer(out5))
        out7 = self.output_layer(out6)
        return out7
def forward_fn1(xxx, ttt):
    x = ops.concat((xxx, ttt), axis=1)
    y = net(x)
    return y
def PDE(xxx, ttt):
    xxx = ms.Tensor(xxx, dtype=ms.float32)
    ttt = ms.Tensor(ttt, dtype=ms.float32)
    grad_fn1 = ms.ops.grad(forward_fn1, grad_position=(0, 1), weights=None)
    secondgrad = ms.ops.grad(grad_fn1, grad_position=(0, 1), weights=None)
    dx = grad_fn1(xxx, ttt)
    dxx = secondgrad(xxx, ttt)
    tmp1 = ms.ops.sin(ttt * math.pi * 2)
    tmp2 = (4 * (math.pi * math.pi * (xxx * (xxx - 1))) - 2)
    r = -1 * (dxx[1] + dxx[0]) - tmp1 * tmp2
    return r
def forward_fn(x0, y0_r, xxx, ttt, ca_all_zeros):
    y0 = net(x0)
    mse_b_1 = mse_loss_function(y0, y0_r)
    ca_out = PDE(xxx, ttt)
    mse_b_3 = mse_loss_function(ca_out, ca_all_zeros)
    loss = 1 / (2 * net.eb * net.eb) * mse_b_1 + 1 / (2 * net.ec * net.ec) * mse_b_3
    return loss
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
    xxx = mindspore.Tensor.from_numpy(xxx)
    ttt = mindspore.Tensor.from_numpy(ttt)
    # 定义的神经网络
    net = NetMS(50, lb, ub)
    # 定义的损失函数
    mse_loss_function = nn.MSELoss(reduction='mean')
    # 定义的优化器
    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=1e-4)
    iteration = 50000
    x0 = ms.Tensor(X_u_train, dtype=ms.float32)
    y0_r = ms.Tensor(u_train, dtype=ms.float32)
    zero = ops.Zeros()
    ca_all_zeros = zero((20000, 1), mindspore.float32)
    start_time = time.time()
    for epoch in range(iteration + 1):
        grad_fn = mindspore.ops.value_and_grad(forward_fn, grad_position=None, weights=optimizer.parameters, has_aux=False)
        loss, grad = grad_fn(x0, y0_r, xxx, ttt, ca_all_zeros)
        optimizer(grad)
        if epoch % 100 == 0:
            print(epoch, "Traning Loss:", loss)
            print(f'times {epoch}  -  loss: {loss}')

    elapsed = time.time() - start_time
    # mindspore.save_checkpoint(net, "adapinn_mindspore.ckpt")
    # param_dict = mindspore.load_checkpoint("adapinn_mindspore.ckpt")
    # param_not_load = mindspore.load_param_into_net(net, param_dict)
    X_star = ms.Tensor(X_star, dtype = ms.float32)
    u_pred = net(X_star)
    error_u = np.linalg.norm(u_star - u_pred.asnumpy(), 2) / np.linalg.norm(u_star, 2)
    print(error_u, elapsed / 60, elapsed % 60)
    print(elapsed)

    # fig = plt.figure()
    # u_pred = u_pred.asnumpy()
    # u_pred = np.reshape(u_pred, (100, 100))
    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    # ax.plot_surface(xx, yy, Exact, cmap='viridis', edgecolor='none')
    # ax.set_title('Possion-AS')
    # bx = fig.add_subplot(1, 2, 2, projection='3d')
    # bx.plot_surface(xx, yy, u_pred, cmap='viridis', edgecolor='none')
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

# for epoch in range(iteration + 1):
#     zero = ops.Zeros()
#     ones = ops.Ones()
#     x0 = zero((100, 1), mindspore.float32) - 2.0
#     y0_r = 4.0 * ones((100, 1), mindspore.float32)
#     # mse_b_1 = mse_loss_function(y0, ms.ops.exp(y0_r))
#
#     x1 = zero((100, 1), mindspore.float32) + 2.0
#     y1_r = -4.0 * ones((100, 1), mindspore.float32)
#     # mse_b_2 = mse_loss_function(y1, ms.ops.exp(y1_r))
#
#     x_in = np.random.uniform(low = -2.0, high = 2.0, size = (3000, 1))
#     ca_in = ms.Tensor(x_in, dtype = mindspore.float32)
#     ca_all_zeros = zero((3000, 1),mindspore.float32)
#     x_train = ms.ops.concat((x0, x1, ca_in), 0)
#     y_real = ms.ops.concat((y0_r, y1_r, ca_all_zeros), 0)
#     # y_train = ms.ops.concat((y0, y1, ca_out), 0)
#
#     grad_fn = mindspore.ops.value_and_grad(forward_fn, grad_position=None, weights = optimizer.parameters, has_aux=True)
#     (loss, y_train), grad = grad_fn(x_train, y_real)
#     optimizer(grad)
#     if epoch % 100 == 0:
#         # y = torch.exp(-2 * ca_in)  # y 真实值
#         # y_train0 = net(ca_in)  # y 预测值
#         print(epoch, "Traning Loss:", loss.data)
#         print(f'times {epoch}  -  loss: {loss}')