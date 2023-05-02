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
    def __init__(self, NN):
        super(NetMS, self).__init__()
        self.input_layer = nn.Dense(2, NN, weight_init = XavierUniform())
        self.h1_layer = nn.Dense(NN, NN, weight_init = XavierUniform())
        self.h2_layer = nn.Dense(NN, NN, weight_init = XavierUniform())
        self.h3_layer = nn.Dense(NN, NN, weight_init = XavierUniform())
        self.h4_layer = nn.Dense(NN, NN, weight_init = XavierUniform())
        self.output_layer = nn.Dense(NN, 1, weight_init = XavierUniform())
    def construct(self, x):
        out = x * x * x
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
if __name__ == "__main__":
    iteration = 10000
    net = NetMS(50)
    for epoch in range(iteration + 1):
        xxx = ms.Tensor([[2.0], [4.0]])
        ttt = ms.Tensor([[3.0], [5.0]])
        PDE(xxx, ttt, net)