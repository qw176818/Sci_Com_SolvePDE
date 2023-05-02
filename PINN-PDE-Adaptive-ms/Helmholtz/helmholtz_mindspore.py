import sys
sys.path.insert(0, '../../Utilities/')
import math
import mindspore.ops
import numpy as np
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.common.initializer import XavierUniform
from mindspore import context
from pyDOE import lhs
import time
import matplotlib.pyplot as plt
import os
context.set_context(mode=context.PYNATIVE_MODE)
np.random.seed(1234)
class NetMS(nn.Cell):
    def __init__(self, NN, lb, ub):
        super(NetMS, self).__init__()
        self.input_layer = nn.Dense(2, NN, weight_init = XavierUniform())
        self.h1_layer = nn.Dense(NN, NN, weight_init = XavierUniform())
        self.h2_layer = nn.Dense(NN, NN, weight_init = XavierUniform())
        self.h3_layer = nn.Dense(NN, NN, weight_init = XavierUniform())
        self.h4_layer = nn.Dense(NN, NN, weight_init = XavierUniform())
        self.output_layer = nn.Dense(NN, 1, weight_init = XavierUniform())
        self.tanh = nn.Tanh()
        self.lb = lb
        self.lb = self.lb.astype(np.float32)
        self.lb = ms.Tensor(self.lb)
        self.ub = ub
        self.ub = self.ub.astype(np.float32)
        self.ub = ms.Tensor(self.ub)
    def construct(self, x):
        x1 = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        out1 = self.tanh(self.input_layer(x1))
        out2 = self.tanh(self.h1_layer(out1))
        out3 = self.tanh(self.h2_layer(out2))
        out4 = self.tanh(self.h3_layer(out3))
        out5 = self.tanh(self.h4_layer(out4))
        out = self.output_layer(out5)
        return out
def forward_fn1(xxx, ttt):
    x = ops.concat((xxx, ttt), axis=1)
    y = net(x)
    return y
def PDE(xxx, ttt, net):
    grad_fn1 = ms.ops.grad(forward_fn1, grad_position=(0, 1), weights=None)
    secondgrad = ms.ops.grad(grad_fn1, grad_position=(0, 1), weights=None)
    X = ms.ops.concat((xxx, ttt), 1)
    u = net(X)
    dx = grad_fn1(xxx, ttt)
    dxx = secondgrad(xxx, ttt)
    tmp1 = ((6 - math.pi * math.pi - 36 * math.pi * math.pi) * \
           ms.ops.sin(math.pi * xxx))
    tmp2 = ms.ops.sin((math.pi * 6 * ttt))
    r = (dxx[1] + dxx[0] + 6 * u) - tmp1 * tmp2
    return r
def forward_fn(x0, y0_r, xxx, ttt, ca_all_zeros):
    y0 = net(x0)
    mse_b_1 = mse_loss_function(y0, y0_r)
    ca_out = PDE(xxx, ttt, net)
    mse_b_3 = mse_loss_function(ca_out, ca_all_zeros)
    loss = 10000 * mse_b_1 + mse_b_3
    return loss
if __name__ == "__main__":
    # 1. Data Process
    noise = 0.0
    N_u = 100
    N_f = 10000
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    yy = np.outer(y, np.ones(100))
    xx = np.outer(x, np.ones(100)).T
    Exact = np.sin(np.pi * xx) * np.sin(6 * np.pi * yy)
    X_star = np.hstack((xx.flatten()[: ,None], yy.flatten()[: ,None]))
    u_star = Exact.flatten()[: ,None]
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
    ttt = X_f_train[:, 1:]
    xxx = ms.Tensor(xxx, dtype=ms.float32)
    ttt = ms.Tensor(ttt, dtype=ms.float32)
    # 定义的神经网络
    net = NetMS(30, lb, ub)
    # 定义的损失函数
    mse_loss_function = ms.nn.MSELoss(reduction='mean')
    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=1e-4)
    iteration = 50000
    x0 = ms.Tensor(X_u_train, dtype=ms.float32)
    y0_r = ms.Tensor(u_train, dtype=ms.float32)
    zero = ops.Zeros()
    ca_all_zeros = zero((10000, 1), mindspore.float32)
    start_time = time.time()
    for epoch in range(iteration + 1):
        grad_fn = mindspore.ops.value_and_grad(forward_fn, grad_position=None, weights=optimizer.parameters, has_aux=False)
        loss, grad = grad_fn(x0, y0_r, xxx, ttt, ca_all_zeros)
        optimizer(grad)
        if epoch % 100 == 0:
            print(epoch, "Traning Loss:", loss)
            print(f'times {epoch}  -  loss: {loss}')
    elapsed = time.time() - start_time
    # mindspore.save_checkpoint(net, "pinn_ode_cmp_mindspore.ckpt")
    X_star = ms.Tensor(X_star, dtype = ms.float32)
    u_pred = net(X_star)
    error_u = np.linalg.norm(u_star - u_pred.asnumpy(), 2) / np.linalg.norm(u_star, 2)
    print('Error u: %e' % (error_u))
    print(elapsed / 60, elapsed % 60)
    # fig = plt.figure()
    # u_pred = u_pred.asnumpy()
    # u_pred = np.reshape(u_pred, (100, 100))
    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    # ax.plot_surface(xx, yy, Exact, cmap='viridis', edgecolor='none')
    # ax.set_title('Helmholtz-AS')
    # bx = fig.add_subplot(1, 2, 2, projection='3d')
    # bx.plot_surface(xx, yy, u_pred, cmap='viridis', edgecolor='none')
    # bx.set_title('Helmholtz-PS')
    # plt.savefig('helmholtz_MindSpore.pdf')
    # plt.savefig('helmholtz_MindSpore.eps')