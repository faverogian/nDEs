import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
    
    def forward(self, x, lengths):
        x_packed = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        encoder_out, (encoder_hn, encoder_cn) = self.lstm(x_packed)
        encoder_out, _ = rnn_utils.pad_packed_sequence(encoder_out, batch_first=True)
        return encoder_out, (encoder_hn, encoder_cn)

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

class LSTMEncoderDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMEncoderDecoder, self).__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, layer_dim)
        self.decoder = LSTMDecoder(input_dim-1, hidden_dim, layer_dim, output_dim)
    
    def forward(self, x, lengths):
        encoder_out, encoder_hidden = self.encoder(x, lengths)
        
        # Initialize decoder hidden state with encoder last hidden state
        decoder_hidden = encoder_hidden

        # Initialize decoder input with zeros
        decoder_input = torch.zeros(x.size(0), 1, x.size(2)-1).to(x.device)
        
        # List to store decoder outputs
        decoder_outputs = []
        
        # Decoder loop
        for _ in range(182-x.size(1)):
            decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_out)
            decoder_input = decoder_out.unsqueeze(1)
        
        # Stack decoder outputs
        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        
        return decoder_outputs