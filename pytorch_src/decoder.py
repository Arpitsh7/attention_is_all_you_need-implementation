import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """
    An individual Decoder Layer as described in Section 3.1 of Vaswani et al. (2017).

    Each layer has three sub-layers:
        1. Masked Multi-Head Self-Attention (prevents attending to subsequent positions)
        2. Multi-Head Cross-Attention (queries from decoder layer, keys & values from encoder output)
        3. Position-wise PositionwiseFeedForward network
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (batch_size, tgt_seq_len, d_model)
            encoder_output: Tensor of shape (batch_size, src_seq_len, d_model)
            tgt_mask: Causal/padding mask for target (batch_size, 1, tgt_seq_len, tgt_seq_len)
            memory_mask: Source padding mask for cross-attention (batch_size, 1, 1, src_seq_len)

        Returns:
            output: Tensor of shape (batch_size, tgt_seq_len, d_model)
            self_attn_weights: Attention weights of decoder self-attention
            cross_attn_weights: Attention weights of encoder-decoder cross-attention
        """
        # Sub-layer 1: Masked Self-Attention
        self_attn_out, self_attn_weights = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))

        # Sub-layer 2: Encoder-Decoder Cross-Attention
        cross_attn_out, cross_attn_weights = self.cross_attn(
            query=x, key=encoder_output, value=encoder_output, mask=memory_mask
        )
        x = self.norm2(x + self.dropout2(cross_attn_out))

        # Sub-layer 3: Feed Forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_out))

        return x, self_attn_weights, cross_attn_weights


class TransformerDecoder(nn.Module):
    """
    Decoder stack of N identical DecoderLayers as described in Section 3.1 of Vaswani et al. (2017).
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
            [DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            x: Tensor of shape (batch_size, tgt_seq_len, d_model)
            encoder_output: Tensor of shape (batch_size, src_seq_len, d_model)
            tgt_mask: Target causal / padding mask
            memory_mask: Source padding mask

        Returns:
            x: Decoder final output (batch_size, tgt_seq_len, d_model)
            self_attentions: List of self-attention weights for each layer
            cross_attentions: List of cross-attention weights for each layer
        """
        self_attentions = []
        cross_attentions = []

        for layer in self.layers:
            x, self_attn, cross_attn = layer(
                x, encoder_output=encoder_output, tgt_mask=tgt_mask, memory_mask=memory_mask
            )
            self_attentions.append(self_attn)
            cross_attentions.append(cross_attn)

        x = self.norm(x)
        return x, self_attentions, cross_attentions
