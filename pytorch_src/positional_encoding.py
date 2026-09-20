import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional Encoding as described in Section 3.5 of Vaswani et al. (2017).

    Formulas:
        PE_(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE_(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, d_model: int = 512, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape to (1, max_len, d_model) for broadcasting over batch dimension
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
