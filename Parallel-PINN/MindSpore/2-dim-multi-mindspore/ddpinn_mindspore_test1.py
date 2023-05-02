import time
import mindspore.ops
import numpy as np
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor, dtype as mstype
import math
import matplotlib.pyplot as plt
from mindspore.common.initializer import XavierUniform
context.set_context(mode=context.PYNATIVE_MODE)
class NetMS(nn.Cell):
    def __init__(self, NN):
        super(NetMS, self).__init__()
        self.input_layer = nn.Dense(1, NN, weight_init = XavierUniform())
        self.h1_layer = nn.Dense(NN, NN, weight_init = XavierUniform())
        self.h2_layer = nn.Dense(NN, NN, weight_init = XavierUniform())
        self.h3_layer = nn.Dense(NN, NN, weight_init = XavierUniform())
        self.h4_layer = nn.Dense(NN, NN, weight_init = XavierUniform())
        # self.h5_layer = nn.Dense(NN, NN, weight_init = XavierUniform())
        # self.h6_layer = nn.Dense(NN, NN, weight_init = XavierUniform())
        self.output_layer = nn.Dense(NN, 1, weight_init = XavierUniform())
        self.tanh = nn.Tanh()
    def construct(self, x):
        out1 = self.tanh(self.input_layer(x))
        out2 = self.tanh(self.h1_layer(out1))
        out3 = self.tanh(self.h2_layer(out2))
        out4 = self.tanh(self.h3_layer(out3))
        out5 = self.tanh(self.h4_layer(out4))
        # out6 = self.tanh(self.h3_layer(out5))
        # out7 = self.tanh(self.h4_layer(out6))
        out8 = self.output_layer(out5)
        return out8
w1 = 1
w2 = 7
net1 = NetMS(60)
pi = Tensor(math.pi, mstype.float32)
def ffn1(x):
    y = ops.tanh(w2 * (x + pi)) * net1(x)
    return y
def PDE1(x):
    # grad_fn1 = mindspore.ops.grad(net, grad_position=0, weights=None, has_aux=False)
    grad_fn1 = ms.ops.grad(ffn1, grad_position=0, weights=None, has_aux= False)
    # secondgrad = ms.ops.grad(grad_fn1, grad_position=0, weights=None, has_aux=False)
    y_x = grad_fn1(x)
    # y_xx = secondgrad(x)
    tmp = w1 * ops.cos(w1 * x) + w2 * ops.cos(w2 * x)
    res = y_x - tmp
    return res
def forward_fn1(ca_in, ca_all_zeros):
    ca_out = PDE1(ca_in)
    mse_c = mse_loss_function(ca_out, ca_all_zeros)
    return mse_c
mse_loss_function = nn.MSELoss(reduction = 'mean')
optimizer = nn.optim.Adam(net1.trainable_params(), learning_rate = 1e-4)
iteration = 50001
for epoch in range(iteration + 1):
    x_in = np.random.uniform(low = -4.5, high = 0.5, size = (400, 1))
    ca_in = ms.Tensor(x_in, dtype = mindspore.float32)
    ca_all_zeros = ops.zeros((400, 1), mindspore.float32)
    grad_fn = mindspore.ops.value_and_grad(forward_fn1, grad_position=None, weights = optimizer.parameters, has_aux = False)
    loss, grad = grad_fn(ca_in, ca_all_zeros)
    optimizer(grad)
    if epoch % 100 == 0:
        # y = torch.exp(-2 * ca_in)  # y 真实值
        # y_train0 = net(ca_in)  # y 预测值
        print(epoch, "Traning Loss:", loss)
        print(f'times {epoch}  -  loss: {loss}')
    if epoch % 10000 == 0:
        plt.cla()
        pre_in1 = ops.linspace(ms.Tensor(-4.5), ms.Tensor(-0.5), 9000)
        pre_in1 = ops.reshape(pre_in1, (9000, 1))
        y = ops.sin(w1 * pre_in1) + ops.sin(w2 * pre_in1)
        y_train0 = ops.tanh(w2 * (pre_in1 + pi)) * net1(pre_in1)
        plt.xlabel(f'x-{epoch}')
        plt.ylabel('sin(x)')
        plt.plot(pre_in1.asnumpy(), y.asnumpy(), linewidth=0.5)
        plt.plot(pre_in1.asnumpy(), y_train0.asnumpy(), c='red', linewidth=0.5)
        plt.pause(0.1)
mindspore.save_checkpoint(net1, "ddpinn_sub_d.ckpt")





# # param_dict = mindspore.load_checkpoint("pinn_ode_cmp_mindspore.ckpt")
# # param_not_load = mindspore.load_param_into_net(net, param_dict)
# # print(param_not_load)
# x = np.linspace(-2, 2, 3000)
# x = x.reshape((3000, 1))
# x_r = ms.Tensor(x, dtype = mindspore.float32)
# y = net(x_r)
# y_r = np.exp(-2 * x)
# error_u = np.linalg.norm(y_r - y.asnumpy(), 2) / np.linalg.norm(y_r, 2)
# print(error_u, eclapse / 60, eclapse % 60)