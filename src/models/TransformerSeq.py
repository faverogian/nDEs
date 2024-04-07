import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout):
        super(TransformerEncoder, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask=None):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout, output_dim=None):
        super(TransformerDecoder, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        if output_dim is not None:
            self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, memory=None, tgt_key_padding_mask=None):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  

        if memory is None:
            memory_shape = (x.size(0), x.size(1), x.size(2))  # Swap batch and sequence dimensions
            memory = torch.zeros(memory_shape, dtype=x.dtype, device=x.device)

        x = self.decoder(x, memory, tgt_key_padding_mask=tgt_key_padding_mask)

        if hasattr(self, 'fc'):
            x = self.fc(x)
        
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(input_dim, hidden_dim, num_heads, num_layers, dropout)
        self.decoder = TransformerDecoder(input_dim, hidden_dim, num_heads, num_layers, dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        encoder_output = self.encoder(src, src_key_padding_mask)
        decoder_output = self.decoder(tgt, encoder_output, tgt_key_padding_mask)
        output = self.fc(decoder_output)
        return output