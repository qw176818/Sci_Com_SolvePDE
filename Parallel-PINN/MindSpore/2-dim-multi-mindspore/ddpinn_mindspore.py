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
import matplotlib as mpl
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
net2 = NetMS(60)
net3 = NetMS(60)
net4 = NetMS(60)
pi = Tensor(math.pi, mstype.float32)


def ffn1(x):
    y = ops.tanh(w2 * (x + 2 * pi)) * net1(x)
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

def ffn2(x):
    y = ops.tanh(w2 * (x + pi)) * net2(x)
    return y
def PDE2(x):
    # grad_fn1 = mindspore.ops.grad(net, grad_position=0, weights=None, has_aux=False)
    grad_fn1 = ms.ops.grad(ffn2, grad_position=0, weights=None, has_aux= False)
    # secondgrad = ms.ops.grad(grad_fn1, grad_position=0, weights=None, has_aux=False)
    y_x = grad_fn1(x)
    # y_xx = secondgrad(x)
    tmp = w1 * ops.cos(w1 * x) + w2 * ops.cos(w2 * x)
    res = y_x - tmp
    return res
def forward_fn2(ca_in, ca_all_zeros):
    ca_out = PDE2(ca_in)
    mse_c = mse_loss_function(ca_out, ca_all_zeros)
    return mse_c

def ffn3(x):
    y = ops.tanh(w2 * (x - pi)) * net3(x)
    return y
def PDE3(x):
    # grad_fn1 = mindspore.ops.grad(net, grad_position=0, weights=None, has_aux=False)
    grad_fn1 = ms.ops.grad(ffn3, grad_position=0, weights=None, has_aux= False)
    # secondgrad = ms.ops.grad(grad_fn1, grad_position=0, weights=None, has_aux=False)
    y_x = grad_fn1(x)
    # y_xx = secondgrad(x)
    tmp = w1 * ops.cos(w1 * x) + w2 * ops.cos(w2 * x)
    res = y_x - tmp
    return res
def forward_fn3(ca_in, ca_all_zeros):
    ca_out = PDE3(ca_in)
    mse_c = mse_loss_function(ca_out, ca_all_zeros)
    return mse_c

def ffn4(x):
    y = ops.tanh(w2 * (x - 2 * pi)) * net4(x)
    return y
def PDE4(x):
    # grad_fn1 = mindspore.ops.grad(net, grad_position=0, weights=None, has_aux=False)
    grad_fn1 = ms.ops.grad(ffn4, grad_position=0, weights=None, has_aux= False)
    # secondgrad = ms.ops.grad(grad_fn1, grad_position=0, weights=None, has_aux=False)
    y_x = grad_fn1(x)
    # y_xx = secondgrad(x)
    tmp = w1 * ops.cos(w1 * x) + w2 * ops.cos(w2 * x)
    res = y_x - tmp
    return res
def forward_fn4(ca_in, ca_all_zeros):
    ca_out = PDE4(ca_in)
    mse_c = mse_loss_function(ca_out, ca_all_zeros)
    return mse_c

mse_loss_function = nn.MSELoss(reduction = 'mean')
optimizer1 = nn.optim.Adam(net1.trainable_params(), learning_rate = 1e-4)
optimizer2 = nn.optim.Adam(net2.trainable_params(), learning_rate = 1e-4)
optimizer3 = nn.optim.Adam(net3.trainable_params(), learning_rate = 1e-4)
optimizer4 = nn.optim.Adam(net4.trainable_params(), learning_rate = 1e-4)
iteration = 50001
start_time = time.time()
for epoch in range(iteration + 1):
    x_in = np.random.uniform(low = -8.0, high = -4.0, size = (600, 1))
    ca_in = ms.Tensor(x_in, dtype = mindspore.float32)
    ca_all_zeros = ops.zeros((600, 1), mindspore.float32)
    grad_fn = mindspore.ops.value_and_grad(forward_fn1, grad_position=None, weights = optimizer1.parameters, has_aux = False)
    loss1, grad = grad_fn(ca_in, ca_all_zeros)
    optimizer1(grad)

    x_in = np.random.uniform(low = -4.5, high = 0.5, size = (400, 1))
    ca_in = ms.Tensor(x_in, dtype = mindspore.float32)
    ca_all_zeros = ops.zeros((400, 1), mindspore.float32)
    grad_fn = mindspore.ops.value_and_grad(forward_fn2, grad_position=None, weights = optimizer2.parameters, has_aux = False)
    loss2, grad = grad_fn(ca_in, ca_all_zeros)
    optimizer2(grad)

    x_in = np.random.uniform(low = -0.5, high = 4.5, size = (400, 1))
    ca_in = ms.Tensor(x_in, dtype = mindspore.float32)
    ca_all_zeros = ops.zeros((400, 1), mindspore.float32)
    grad_fn = mindspore.ops.value_and_grad(forward_fn3, grad_position=None, weights = optimizer3.parameters, has_aux = False)
    loss3, grad = grad_fn(ca_in, ca_all_zeros)
    optimizer3(grad)

    x_in = np.random.uniform(low = 4.0, high = 8.0, size = (600, 1))
    ca_in = ms.Tensor(x_in, dtype = mindspore.float32)
    ca_all_zeros = ops.zeros((600, 1), mindspore.float32)
    grad_fn = mindspore.ops.value_and_grad(forward_fn4, grad_position=None, weights = optimizer4.parameters, has_aux = False)
    loss4, grad = grad_fn(ca_in, ca_all_zeros)
    optimizer4(grad)

    if epoch % 100 == 0:
        # y = torch.exp(-2 * ca_in)  # y 真实值
        # y_train0 = net(ca_in)  # y 预测值
        print(epoch, "Traning Loss:", loss1, loss2, loss3, loss4)
        print(f'times {epoch}  -  loss: {loss1, loss2, loss3, loss4}')
    # if epoch % 10000 == 0:
    #     plt.cla()
    #     pre_in1 = ops.linspace(ms.Tensor(-4.5), ms.Tensor(-0.5), 9000)
    #     pre_in1 = ops.reshape(pre_in1, (9000, 1))
    #     y = ops.sin(w1 * pre_in1) + ops.sin(w2 * pre_in1)
    #     y_train0 = ops.tanh(w2 * (pre_in1 + pi)) * net1(pre_in1)
    #     plt.xlabel(f'x-{epoch}')
    #     plt.ylabel('sin(x)')
    #     plt.plot(pre_in1.asnumpy(), y.asnumpy(), linewidth=0.5)
    #     plt.plot(pre_in1.asnumpy(), y_train0.asnumpy(), c='red', linewidth=0.5)
    #     plt.pause(0.1)
eclapse = time.time() - start_time
print(eclapse)
mindspore.save_checkpoint(net1, "ddpinn_sub_d1.ckpt")
mindspore.save_checkpoint(net2, "ddpinn_sub_d2.ckpt")
mindspore.save_checkpoint(net3, "ddpinn_sub_d3.ckpt")
mindspore.save_checkpoint(net4, "ddpinn_sub_d4.ckpt")



















param_dict = mindspore.load_checkpoint("ddpinn_sub_d1.ckpt")
param_not_load1 = mindspore.load_param_into_net(net1, param_dict)
param_dict = mindspore.load_checkpoint("ddpinn_sub_d2.ckpt")
param_not_load2 = mindspore.load_param_into_net(net2, param_dict)
param_dict = mindspore.load_checkpoint("ddpinn_sub_d3.ckpt")
param_not_load3 = mindspore.load_param_into_net(net3, param_dict)
param_dict = mindspore.load_checkpoint("ddpinn_sub_d4.ckpt")
param_not_load4 = mindspore.load_param_into_net(net4, param_dict)


colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']*100
a = [[-8.0, -4.0], [-4.5, 0.5], [-0.5, 4.5], [4.0, 8.0]]
bias = [2, 1, -1, -2]
net = [net1, net2, net3, net4]
num_p = 500
for i in range(4):
    xpre = ops.linspace(ms.Tensor(a[i][0]), ms.Tensor(a[i][1]), num_p)
    xpre = ops.reshape(xpre, (num_p, 1))
    ypre = ops.tanh(w2 * (xpre + pi * bias[i])) * net[i](xpre)
    y_star = ops.sin(w1 * xpre) + ops.sin(w2 * xpre)
    sub_l2_norm = np.linalg.norm(y_star.asnumpy() - ypre.asnumpy(), 2) \
                   / np.linalg.norm(y_star.asnumpy(), 2)
    print("sub" + str(i + 1) + "_l2_norm: ")
    print(sub_l2_norm)
a2 = [[-8.0, -4.5], [-4.5, -4.0], [-4.0, -0.5], [-0.5, 0.5], [0.5, 4.0], [4.0, 4.5], [4.5, 8.0]]
# net = [net1, net, net2, net, net3, net, net4, net, net5]
colors = ['tab:blue', 'tab:orange', 'tab:olive', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']*100
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
fig = plt.figure(figsize = (10.0, 5.0))
x = ops.linspace(ms.Tensor(-8.0), ms.Tensor(8.0), num_p)
x = ops.reshape(x, (num_p, 1))
y_star = ops.sin(w1 * x) + ops.sin(w2 * x)
plt.plot(x.asnumpy(), y_star.asnumpy(), color='green', label='Exact')
num_p = 1000
for i, d in enumerate(a2):
    x = ops.linspace(ms.Tensor(a2[i][0]), ms.Tensor(a2[i][1]), num_p)
    x = ops.reshape(x, (num_p, 1))

    if i % 2 == 0:
        y = ops.tanh(w2 * (x + bias[int(i / 2)] * pi)) * (net[int(i / 2)](x))
        plt.plot(x.asnumpy(), y.asnumpy(),  colors[int(i / 2)], label='Sub-d'+str(int((i / 2) + 1)))
    else:
        y1 = ops.tanh(w2 * (x + bias[int(i / 2)] * pi)) * net[int(i / 2)](x)
        y2 = ops.tanh(w2 * (x + bias[int((i + 1) / 2)] * pi)) * net[int((i + 1) / 2)](x)
        y = (y1 + y2) / 2
        plt.plot(x.asnumpy(), y.asnumpy(),  "pink", label='Cross-R'+str(int((i + 1) / 2)))

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('x')
plt.ylabel('u')
plt.ylim(-2.2, 2.2)
plt.xlim(-8.0, 12.0)
plt.title("DDPINN $w_{1} = 1, w_{2} = 7$")
plt.legend(fontsize = 13)
# plt.show()
plt.savefig("DDPINN-2-multy-12_ms.pdf")
plt.savefig("DDPINN-2-multy-12_ms.eps")
num1 = 225
num2 = 33
x_sum = np.array([[0]])
y_sum = np.array([[0]])
y_sumr = np.array([[0]])
for i, d in enumerate(a2):
    if i % 2 == 0:
        x = ops.linspace(ms.Tensor(a2[i][0]), ms.Tensor(a2[i][1]), num1)
        x = ops.reshape(x, (num1, 1))
        y_r = (ops.sin(w1 * x) + ops.sin(w2 * x)).asnumpy()
        y = (ops.tanh(w2 * (x + bias[int(i / 2)] * pi)) * (net[int(i / 2)](x))).asnumpy()
        x = x.asnumpy()
        y_sum = np.concatenate((y_sum, y))
        x_sum = np.concatenate((x_sum, x))
        y_sumr = np.concatenate((y_sumr, y_r))
    else:
        x = ops.linspace(ms.Tensor(a2[i][0]), ms.Tensor(a2[i][1]), num2)
        x = ops.reshape(x, (num2, 1))
        y_r = (ops.sin(w1 * x) + ops.sin(w2 * x)).asnumpy()
        y1 =ops.tanh(w2 * (x + bias[int(i / 2)] * pi)) * net[int(i / 2)](x)
        y2 = ops.tanh(w2 * (x + bias[int((i + 1) / 2)] * pi)) * net[int((i + 1) / 2)](x)
        y = (y1 + y2) / 2
        y = y.asnumpy()
        x = x.asnumpy()
        y_sum = np.concatenate((y_sum, y))
        x_sum = np.concatenate((x_sum, x))
        y_sumr = np.concatenate((y_sumr, y_r))
x_sum = x_sum[1:]
y_sum = y_sum[1:]
y_sumr = y_sumr[1:]
error = np.linalg.norm(y_sum - y_sumr, 2) / np.linalg.norm(y_sumr, 2)
print(error)