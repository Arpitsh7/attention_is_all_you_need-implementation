import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Networks as described in Section 3.3 of Vaswani et al. (2017).

    Formula:
        FFN(x) = max(0, x * W_1 + b_1) * W_2 + b_2
    """

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))
