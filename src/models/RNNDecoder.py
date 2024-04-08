import torch
import torch.nn as nn

class RNNDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, decoder_hidden=None):
        if decoder_hidden is None:
            decoder_hidden = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        decoder_out, decoder_hidden = self.rnn(x, decoder_hidden)
        out = self.fc(decoder_out.squeeze(1))
        return out, decoder_hidden
    
    def autoregressive_predict(self, x, max_len):
        decoder_hidden = None
        decoder_input = x
        decoder_outputs = []
        for _ in range(max_len):
            decoder_out, decoder_hidden = self(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_out)
            decoder_input = decoder_out.unsqueeze(1)
        return torch.stack(decoder_outputs, dim=1)
