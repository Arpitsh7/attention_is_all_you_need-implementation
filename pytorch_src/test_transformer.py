import unittest
import torch
from pytorch_src.attention import ScaledDotProductAttention, MultiHeadAttention
from pytorch_src.positional_encoding import PositionalEncoding
from pytorch_src.encoder import EncoderLayer, TransformerEncoder
from pytorch_src.decoder import DecoderLayer, TransformerDecoder
from pytorch_src.transformer import Transformer, generate_causal_mask


class TestTransformerModules(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 2
        self.src_len = 5
        self.tgt_len = 6
        self.d_model = 64
        self.num_heads = 4
        self.vocab_size = 50

    def test_scaled_dot_product_attention(self):
        attn = ScaledDotProductAttention(dropout=0.0)
        d_k = self.d_model // self.num_heads
        q = torch.randn(self.batch_size, self.num_heads, self.src_len, d_k)
        k = torch.randn(self.batch_size, self.num_heads, self.src_len, d_k)
        v = torch.randn(self.batch_size, self.num_heads, self.src_len, d_k)

        out, weights = attn(q, k, v)
        self.assertEqual(out.shape, (self.batch_size, self.num_heads, self.src_len, d_k))
        self.assertEqual(weights.shape, (self.batch_size, self.num_heads, self.src_len, self.src_len))
        
        # Check softmax rows sum to 1.0
        row_sums = weights.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums))

    def test_multi_head_attention(self):
        mha = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads, dropout=0.0)
        x = torch.randn(self.batch_size, self.src_len, self.d_model)

        out, weights = mha(x, x, x)
        self.assertEqual(out.shape, (self.batch_size, self.src_len, self.d_model))
        self.assertEqual(weights.shape, (self.batch_size, self.num_heads, self.src_len, self.src_len))

    def test_positional_encoding(self):
        pe = PositionalEncoding(d_model=self.d_model, dropout=0.0, max_len=100)
        x = torch.zeros(self.batch_size, self.src_len, self.d_model)
        out = pe(x)
        self.assertEqual(out.shape, (self.batch_size, self.src_len, self.d_model))
        # Ensure encoding is not all zeros
        self.assertTrue(torch.abs(out).sum() > 0)

    def test_causal_mask(self):
        mask = generate_causal_mask(4)
        self.assertEqual(mask.shape, (1, 1, 4, 4))
        # Lower triangular should be 1/True, upper triangle 0/False
        expected = torch.tensor([
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1]
        ], dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        torch.testing.assert_close(mask, expected)

    def test_full_transformer_forward_backward(self):
        model = Transformer(
            src_vocab_size=self.vocab_size,
            tgt_vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=128,
        )

        src = torch.randint(1, self.vocab_size, (self.batch_size, self.src_len))
        tgt = torch.randint(1, self.vocab_size, (self.batch_size, self.tgt_len))

        logits, attn_info = model(src, tgt)
        self.assertEqual(logits.shape, (self.batch_size, self.tgt_len, self.vocab_size))
        self.assertIn("encoder_attentions", attn_info)
        self.assertEqual(len(attn_info["encoder_attentions"]), 2)

        # Test backward pass
        loss = logits.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient for {name} is None")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")


if __name__ == "__main__":
    unittest.main()
