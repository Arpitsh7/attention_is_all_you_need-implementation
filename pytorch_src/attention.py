import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention as described in Section 3.2.1 of Vaswani et al. (2017).

    Formula:
        Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Tensor of shape (batch_size, num_heads, seq_len_q, d_k)
            key:   Tensor of shape (batch_size, num_heads, seq_len_k, d_k)
            value: Tensor of shape (batch_size, num_heads, seq_len_k, d_v)
            mask:  Optional Tensor broadcastable to (batch_size, num_heads, seq_len_q, seq_len_k).
                   Positions with 0 (or False) will be masked out (filled with -inf before softmax).

        Returns:
            context: Tensor of shape (batch_size, num_heads, seq_len_q, d_v)
            attn_weights: Attention weights of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        d_k = query.size(-1)
        # Compute scaled dot products: (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # Mask out positions where mask is 0 / False with -inf
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Output context representation
        context = torch.matmul(attn_weights, value)
        return context, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module as described in Section 3.2.2 of Vaswani et al. (2017).

    Instead of performing a single attention function with d_model-dimensional keys, values and queries,
    we linearly project the queries, keys and values h times with different, learned linear projections.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key:   (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_k, d_model)
            mask:  Optional mask tensor (batch_size, 1, seq_len_q, seq_len_k) or (batch_size, 1, 1, seq_len_k)

        Returns:
            output: (batch_size, seq_len_q, d_model)
            attn_weights: Attention weights across all heads (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)

        # 1) Linear projections and reshape to (batch_size, num_heads, seq_len, d_k)
        q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2) Scaled Dot-Product Attention
        context, attn_weights = self.attention(q, k, v, mask=mask)

        # 3) Concatenate heads and apply final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(context)

        return output, attn_weights
