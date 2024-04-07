import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, lengths):
        # Pack the sequence
        x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # LSTM
        out, (hn, cn) = self.lstm(x_packed)
        # Unpack sequence
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # Only want last time step hidden states
        out = out[range(out.shape[0]), lengths - 1, :]
        # Linear layer
        out = self.fc(out)
        return out