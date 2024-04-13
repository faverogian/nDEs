cd import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, lengths):
        # Pack the sequence
        x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # RNN
        out, _ = self.rnn(x_packed)
        # Unpack sequence
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # Only want last time step hidden states
        out = out[range(out.shape[0]), lengths - 1, :]
        # Linear layer
        out = self.fc(out)
        return out

class RNNModelReg(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModelReg, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, lengths, hidden=None):
        # If no initial hidden state is provided, default to None (RNN will use zero state)
        x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, hidden = self.rnn(x_packed, hidden)  # Pass the initial hidden state if provided
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = self.fc(out)
        return out, hidden  # Return both output and the last hidden state
