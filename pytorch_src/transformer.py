import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from .positional_encoding import PositionalEncoding
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


def generate_causal_mask(size: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Generate a lower triangular mask for target sequence to prevent attending to future tokens.

    Args:
        size: Target sequence length (tgt_seq_len)
        device: PyTorch device

    Returns:
        Tensor of shape (1, 1, size, size) where 1 indicates allowed attention and 0 indicates masked attention.
    """
    mask = torch.tril(torch.ones((size, size), device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
    return mask


class Transformer(nn.Module):
    """
    Complete Transformer Sequence-to-Sequence Model architecture from Vaswani et al. (2017).
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_idx: int = 0,
        tgt_pad_idx: int = 0,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        share_embeddings: bool = False,
    ):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.d_model = d_model

        # Token Embeddings
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
        self.pos_decoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)

        # Encoder & Decoder
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Final projection to target vocabulary logits
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # Optional parameter sharing (Section 3.4)
        if share_embeddings:
            assert src_vocab_size == tgt_vocab_size, "Vocab sizes must match to share embeddings"
            self.tgt_embed.weight = self.src_embed.weight
            self.generator.weight = self.tgt_embed.weight

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters with Glorot / Xavier uniform initialization as in the paper."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Create padding mask for source sequence: (batch_size, 1, 1, src_seq_len)
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Create target mask combining padding mask and causal lower-triangular mask.
        Shape: (batch_size, 1, tgt_seq_len, tgt_seq_len)
        """
        batch_size, tgt_len = tgt.size()
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = generate_causal_mask(tgt_len, device=tgt.device)
        return tgt_pad_mask & causal_mask

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            src: Source token tensor of shape (batch_size, src_seq_len)
            tgt: Target token tensor of shape (batch_size, tgt_seq_len)
            src_mask: Optional custom source mask
            tgt_mask: Optional custom target mask

        Returns:
            logits: Output target token predictions of shape (batch_size, tgt_seq_len, tgt_vocab_size)
            attention_info: Dict containing encoder and decoder attention weight lists
        """
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)

        # Section 3.4: Multiply embedding weights by sqrt(d_model)
        src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_decoder(self.tgt_embed(tgt) * math.sqrt(self.d_model))

        # Encode
        memory, enc_attentions = self.encoder(src_emb, mask=src_mask)

        # Decode
        dec_output, dec_self_attns, dec_cross_attns = self.decoder(
            tgt_emb, encoder_output=memory, tgt_mask=tgt_mask, memory_mask=src_mask
        )

        logits = self.generator(dec_output)

        attention_info = {
            "encoder_attentions": enc_attentions,
            "decoder_self_attentions": dec_self_attns,
            "decoder_cross_attentions": dec_cross_attns,
        }

        return logits, attention_info

    @torch.no_grad()
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
        memory, _ = self.encoder(src_emb, mask=src_mask)
        return memory

    @torch.no_grad()
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)
        tgt_emb = self.pos_decoder(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        out, _, _ = self.decoder(tgt_emb, encoder_output=memory, tgt_mask=tgt_mask, memory_mask=src_mask)
        return out
