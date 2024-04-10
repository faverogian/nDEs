import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout):
        super(TransformerEncoder, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.hidden_dim = hidden_dim

    def get_padding_mask(self, x):
        # Create a padding mask
        padding_mask = x[:, :, 0] != 0
        return padding_mask

    def forward(self, x):
        key_padding_mask = self.get_padding_mask(x)
        x = self.embedding(x) * math.sqrt(self.hidden_dim)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout, output_dim=None):
        super(TransformerDecoder, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        if output_dim is not None:
            self.fc = nn.Linear(hidden_dim, output_dim)

        self.hidden_dim = hidden_dim

    def get_padding_mask(self, x):
        # Create a padding mask
        padding_mask = x[:, :, 0] != 0
        return padding_mask

    def make_causal_mask(self, x):
        # Create a causal mask
        causal_mask = torch.ones(x.shape[1], x.shape[1])
        causal_mask = torch.tril(causal_mask)
        causal_mask = causal_mask.bool()
        return causal_mask

    def forward(self, x, memory=None):
        causal_mask = self.make_causal_mask(x).to(x.device)
        key_padding_mask = self.get_padding_mask(x).to(x.device)
        x = self.embedding(x) * math.sqrt(self.hidden_dim)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  

        x = self.decoder(x, memory, tgt_mask=causal_mask, tgt_key_padding_mask=key_padding_mask)

        if hasattr(self, 'fc'):
            x = self.fc(x)
        
        return x
    
    def autoregressive_predict(self, x, memory=None, max_len=None):
        for i in range(max_len):
            x_i = self.forward(x, memory)
            x_i = x_i[-1:, :, :]
            x_i = x_i.permute(1, 0, 2)
            x = torch.cat([x, x_i], dim=1)
        return x


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout):
        super(Transformer, self).__init__()
        
        self.transformer = nn.Transformer(d_model=hidden_dim, 
                                          nhead=num_heads, 
                                          num_encoder_layers=num_layers, 
                                          num_decoder_layers=num_layers, 
                                          dropout=dropout)
        
        self.pos_encoder = PositionalEncoding(d_model=hidden_dim, dropout=dropout)
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dim_model = hidden_dim

    def forward(self, src, tgt, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        # Get the embedding and positional encoding of the source and target sequences
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        # Permute to obtain size (sequence length, batch_size, dim_model)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        out = self.transformer(src=src, 
                               tgt=tgt, 
                               tgt_mask=tgt_mask, 
                               src_key_padding_mask=src_padding_mask, 
                               tgt_key_padding_mask=tgt_padding_mask)
        out = self.fc(out)

        return out
    
    def get_causal_mask(self, sz):
        # Create a causal mask
        causal_mask = torch.ones(sz, sz)
        causal_mask = torch.tril(causal_mask)
        causal_mask = causal_mask.float()
        causal_mask = causal_mask.masked_fill(causal_mask == 0, float('-inf'))
        causal_mask = causal_mask.masked_fill(causal_mask == 1, 0.0)

        return causal_mask
    
    def get_padding_mask(self, x):
        # Create a padding mask
        padding_mask = x[:, :, -1] == 0
        return padding_mask