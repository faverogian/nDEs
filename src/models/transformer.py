import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, num_heads, dropout):
        super(Transformer, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layer_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, attention_mask=None):
        # Embedding layer
        x = self.embedding(x)
        
        # Transformer Encoder
        x = x.permute(1, 0, 2)  # Change from (batch_size, seq_len, input_dim) to (seq_len, batch_size, input_dim)
        if attention_mask is not None:
            attention_mask = attention_mask.permute(1, 0)
            x = x * attention_mask.unsqueeze(-1)
        x = self.encoder(x)
        
        # Take the output of the last layer
        x = x[-1, :, :]  # Take the last layer's output for each time step
        
        # Fully connected layer
        x = self.fc(x)
        return x