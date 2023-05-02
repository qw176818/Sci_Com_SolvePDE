import time

import mindspore.ops
import numpy as np
import mindspore as ms
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import context
import matplotlib.pyplot as plt
context.set_context(mode=context.PYNATIVE_MODE)
class NetMS(nn.Cell):
    def __init__(self, NN):
        super(NetMS, self).__init__()
        self.input_layer = nn.Dense(1, NN)
        self.h1_layer = nn.Dense(NN, NN)
        self.h2_layer = nn.Dense(NN, NN)
        self.h3_layer = nn.Dense(NN, NN)
        self.h4_layer = nn.Dense(NN, NN)
        self.output_layer = nn.Dense(NN, 1)
        self.tanh = nn.Tanh()
    def construct(self, x):
        out1 = self.tanh(self.input_layer(x))
        out2 = self.tanh(self.h1_layer(out1))
        out3 = self.tanh(self.h2_layer(out2))
        out4 = self.tanh(self.h3_layer(out3))
        out5 = self.tanh(self.h4_layer(out4))
        out6 = self.output_layer(out5)
        return out6
net = NetMS(90)
mse_loss_function = nn.MSELoss(reduction = 'mean')
optimizer = nn.optim.Adam(net.trainable_params(), learning_rate = 1e-4)
def PDE(x, net):
    grad_fn1 = mindspore.ops.grad(net, grad_position=0, weights=None, has_aux=False)
    y_x = grad_fn1(x)
    y1_x = y_x + 2 * ms.ops.exp(-2 * x)
    return y1_x
iteration = 30000
def forward_fn(x0, y0_r, x1, y1_r, ca_in, ca_all_zeros):
    y0 = net(x0)
    mse_b_1 = mse_loss_function(y0, ms.ops.exp(y0_r))
    y1 = net(x1)
    mse_b_2 = mse_loss_function(y1, ms.ops.exp(y1_r))
    ca_out = PDE(ca_in, net)
    mse_b_3 = mse_loss_function(ca_out, ca_all_zeros)
    loss = mse_b_1 + mse_b_2 + mse_b_3
    return loss
net.set_train()
start_time = time.time()
zero = ops.Zeros()
ones = ops.Ones()
x0 = zero((100, 1), mindspore.float32) - 2.0
y0_r = 4.0 * ones((100, 1), mindspore.float32)
x1 = zero((100, 1), mindspore.float32) + 2.0
y1_r = -4.0 * ones((100, 1), mindspore.float32)
for epoch in range(iteration + 1):
    x_in = np.random.uniform(low = -2.0, high = 2.0, size = (3000, 1))
    ca_in = ms.Tensor(x_in, dtype = mindspore.float32)
    ca_all_zeros = zero((3000, 1),mindspore.float32)
    grad_fn = mindspore.ops.value_and_grad(forward_fn, grad_position=None, weights = optimizer.parameters, has_aux = False)
    loss, grad = grad_fn(x0, y0_r, x1, y1_r, ca_in, ca_all_zeros)
    optimizer(grad)
    if epoch % 100 == 0:
        # y = torch.exp(-2 * ca_in)  # y 真实值
        # y_train0 = net(ca_in)  # y 预测值
        print(epoch, "Traning Loss:", loss)
        print(f'times {epoch}  -  loss: {loss}')
eclapse = time.time() - start_time
mindspore.save_checkpoint(net, "pinn_ode_cmp_mindspore.ckpt")
# param_dict = mindspore.load_checkpoint("pinn_ode_cmp_mindspore.ckpt")
# param_not_load = mindspore.load_param_into_net(net, param_dict)
# print(param_not_load)
x = np.linspace(-2, 2, 3000)
x = x.reshape((3000, 1))
x_r = ms.Tensor(x, dtype = mindspore.float32)
y = net(x_r)
y_r = np.exp(-2 * x)
error_u = np.linalg.norm(y_r - y.asnumpy(), 2) / np.linalg.norm(y_r, 2)
print(error_u, eclapse / 60, eclapse % 60)
# plt.plot(x, y_r, color = 'green')
# plt.plot(x, y.asnumpy(), color = 'red', linestyle = '--')
# plt.show()