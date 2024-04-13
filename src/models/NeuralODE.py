import math
import torch
from torchdiffeq import odeint_adjoint as odeint

######################
# A ODE model looks like
#
# z_t = z_0 + \int_0^t f_\theta(z_s) ds
#
# Where z is the hidden state of your data and f_\theta is a neural network. So the first thing we need to do is define such an f_\theta.
# That's what this ODEFunc class does.
######################
class ODEFunc(torch.nn.Module):

    def __init__(self, input_channels, hidden_channels):
        super(ODEFunc, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.linear2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.linear3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.linear4 = torch.nn.Linear(hidden_channels, hidden_channels)

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                # Initialize weights from a normal distribution with mean 0 and std 0.1
                torch.nn.init.normal_(m.weight, mean=0, std=0.1)
                # Initialize biases to zero
                torch.nn.init.constant_(m.bias, val=0)

    def forward(self, t, z):
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        z = z.relu()
        z = self.linear3(z)
        z = z.relu()
        z = self.linear4(z)
        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()

        return z
    
class NeuralODE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(NeuralODE, self).__init__()

        self.ode_func = ODEFunc(hidden_channels, hidden_channels)
        self.ode_initial = torch.nn.Linear(input_channels, hidden_channels)
        self.ode_readout = torch.nn.Linear(hidden_channels, output_channels)

    def forward(self, x):
        # x = (batch, length, input_channels)
        x = self.ode_initial(x)
        x_0 = x[:, 0, :]

        t = torch.arange(x.size(1)).float()
        z_T = odeint(self.ode_func, x_0, t, method='rk4', options={'step_size': 1})
        z_T = z_T.transpose(0, 1)

        pred_y = self.ode_readout(z_T)

        return pred_y