#############################################################################################
# @article{kidger2020neuralcde,
#    title={{N}eural {C}ontrolled {D}ifferential {E}quations for {I}rregular {T}ime {S}eries},
#    author={Kidger, Patrick and Morrill, James and Foster, James and Lyons, Terry},
#    journal={Advances in Neural Information Processing Systems},
#    year={2020}
# }
#############################################################################################

import math
import torch
import torchcde
import torch
import torch.nn as nn
import numpy as np

######################
# A CDE model looks like
#
# z_t = z_0 + \int_0^t f_\theta(z_s) dX_s
#
# Where X is your data and f_\theta is a neural network. So the first thing we need to do is define such an f_\theta.
# That's what this CDEFunc class does.
######################
class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.linear2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.linear3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.linear4 = torch.nn.Linear(hidden_channels, input_channels * hidden_channels)

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
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
        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z

class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, decoder_hidden):
        decoder_out, (decoder_hn, decoder_cn) = self.lstm(x, decoder_hidden)
        out = self.fc(decoder_out.squeeze(1))
        return out, (decoder_hn, decoder_cn)

######################
# Next, we need to package CDEFunc and Transformer Decoder up into a model.
######################
class NeuralCDELSTM(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, lstm_channels, output_channels, num_layers, interpolation="cubic"):
        super(NeuralCDELSTM, self).__init__()

        self.cde_func = CDEFunc(input_channels, hidden_channels)
        self.cde_initial = torch.nn.Linear(input_channels, hidden_channels)
        self.cde_readout = torch.nn.Linear(hidden_channels, lstm_channels)

        self.decoder = LSTMDecoder(input_channels-1, lstm_channels, num_layers, output_channels)

        self.interpolation = interpolation
        self.num_layers = num_layers
        self.lstm_channels = lstm_channels
        self.output_channels = output_channels

    def forward(self, coeffs):
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        ######################
        X0 = X.evaluate(X.interval[0])
        z0 = self.cde_initial(X0)

        ######################
        # Actually solve the CDE.
        ######################
        interval = torch.tensor(np.linspace(0, 126, 127))
        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.cde_func,
                              adjoint=True,
                              t=X.interval,
                              method='rk4',
                              options={'step_size': 1})

        # Initialize decoder hidden state with encoder last hidden state
        z_T = z_T[:,1]
        z_T = self.cde_readout(z_T)
        z_T = z_T.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        decoder_hidden = (z_T, z_T)
        
        # Initialize decoder input with zeros
        decoder_input = torch.zeros(coeffs.size(0), 1, self.output_channels).to(coeffs.device)

        # List to store decoder outputs
        decoder_outputs = []
        
        # Decoder loop
        for _ in range(182-coeffs.size(1)-1):
            decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_out)
            decoder_input = decoder_out.unsqueeze(1)
        
        # Stack decoder outputs
        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        
        return decoder_outputs