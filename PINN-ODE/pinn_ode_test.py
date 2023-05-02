import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch import autograd

class Net(nn.Module):
    def __init__(self, NL, NN):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(1, NN)
        self.hidden_layer = nn.Linear(NN,int(NN/2))
        self.output_layer = nn.Linear(int(NN/2), 1)

    def forward(self, x):
        out = torch.tanh(self.input_layer(x))
        out = torch.tanh(self.hidden_layer(out))
        out_final = self.output_layer(out)
        return out_final


net=Net(4,20)
mse_cost_function = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(),lr=1e-4)

def ode_01(x,net):
    y=net(x)
    y_x = autograd.grad(y, x,grad_outputs=torch.ones_like(net(x)),create_graph=True)[0]
    return y-y_x   # y-y' = 0

# requires_grad=True).unsqueeze(-1)

plt.ion()
iterations=200000
for epoch in range(iterations):

    optimizer.zero_grad()

    x_0 = torch.zeros(2000, 1)
    y_0 = net(x_0)
    mse_i = mse_cost_function(y_0, torch.ones(2000, 1))


    x_in = np.random.uniform(low=0.0, high=2.0, size=(2000, 1))
    pt_x_in = autograd.Variable(torch.from_numpy(x_in).float(), requires_grad=True)
    pt_y_colection=ode_01(pt_x_in,net)
    pt_all_zeros= autograd.Variable(torch.from_numpy(np.zeros((2000,1))).float(), requires_grad=False)
    mse_f=mse_cost_function(pt_y_colection, pt_all_zeros)

    loss = mse_i + mse_f
    loss.backward()
    optimizer.step()

    if epoch%1000==0:
            y = torch.exp(pt_x_in)
            y_train0 = net(pt_x_in)
            print(epoch, "Traning Loss:", loss.data)
            print(f'times {epoch}  -  loss: {loss.item()} - y_0: {y_0}')
            plt.cla()
            plt.scatter(pt_x_in.detach().numpy(), y.detach().numpy())
            plt.scatter(pt_x_in.detach().numpy(), y_train0.detach().numpy(),c='red')
            plt.pause(0.1)

