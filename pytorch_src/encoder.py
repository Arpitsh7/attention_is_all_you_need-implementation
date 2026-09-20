import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """
    An individual Encoder Layer as described in Section 3.1 of Vaswani et al. (2017).

    Each layer has two sub-layers:
        1. Multi-Head Self-Attention mechanism
        2. Position-wise PositionwiseFeedForward network
    Residual connections are applied around each sub-layer, followed by Layer Normalization.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            mask: Optional src_mask of shape (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len)

        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
            attn_weights: Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        # Sub-layer 1: Self-Attention
        attn_out, attn_weights = self.self_attn(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # Sub-layer 2: Feed Forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))

        return x, attn_weights


class TransformerEncoder(nn.Module):
    """
    Encoder stack of N identical EncoderLayers as described in Section 3.1 of Vaswani et al. (2017).
    """

    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            mask: Optional src_mask tensor

        Returns:
            x: Final encoder outputs (batch_size, seq_len, d_model)
            all_attentions: List of attention weight tensors for each layer
        """
        all_attentions = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask=mask)
            all_attentions.append(attn_weights)

        x = self.norm(x)
        return x, all_attentions
