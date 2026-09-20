import React, { useState } from 'react';
import { Code, Copy, Check, FileText } from 'lucide-react';

export default function CodeExplorer() {
  const [activeFile, setActiveFile] = useState('attention.py');
  const [copied, setCopied] = useState(false);

  const files = {
    'attention.py': {
      title: 'Scaled Dot-Product & Multi-Head Attention',
      section: 'Section 3.2',
      code: `import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention:
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
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, value)
        return context, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        context, attn_weights = self.attention(q, k, v, mask=mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(context), attn_weights`,
    },
    'positional_encoding.py': {
      title: 'Sinusoidal Positional Encoding',
      section: 'Section 3.5',
      code: `import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    PE_(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE_(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """
    def __init__(self, d_model: int = 512, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)`,
    },
    'transformer.py': {
      title: 'Complete Transformer Sequence-to-Sequence Model',
      section: 'Section 3.1 - 3.4',
      code: `import math
import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.pos_decoder = PositionalEncoding(d_model=d_model, dropout=dropout)

        self.encoder = TransformerEncoder(num_layers=num_encoder_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff)
        self.decoder = TransformerDecoder(num_layers=num_decoder_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff)

        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_decoder(self.tgt_embed(tgt) * math.sqrt(self.d_model))

        memory, enc_attns = self.encoder(src_emb, mask=src_mask)
        output, dec_self, dec_cross = self.decoder(tgt_emb, encoder_output=memory, tgt_mask=tgt_mask, memory_mask=src_mask)

        logits = self.generator(output)
        return logits, {"encoder_attentions": enc_attns, "decoder_self_attentions": dec_self, "decoder_cross_attentions": dec_cross}`,
    },
    'train.py': {
      title: 'Noam Optimizer & Training Loop',
      section: 'Section 5.3',
      code: `class NoamOpt:
    """
    lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    """
    def __init__(self, d_model: int, warmup: int, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.d_model = d_model

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self.optimizer.step()

    def rate(self, step: int = None) -> float:
        if step is None:
            step = self._step
        return self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))`,
    },
  };

  const current = files[activeFile] || files['attention.py'];

  const handleCopy = () => {
    navigator.clipboard.writeText(current.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="glass-panel p-6">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Code className="h-5 w-5 text-indigo-400" />
              PyTorch Implementation Source Explorer
            </h2>
            <p className="text-xs text-gray-400 mt-1">
              Located in <code className="text-indigo-300">/pytorch_src</code> with 100% equation fidelity to Vaswani et al. (2017).
            </p>
          </div>

          <button
            onClick={handleCopy}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-gray-700 text-xs text-gray-200 transition"
          >
            {copied ? <Check className="h-4 w-4 text-emerald-400" /> : <Copy className="h-4 w-4 text-gray-400" />}
            <span>{copied ? 'Copied!' : 'Copy Code Snippet'}</span>
          </button>
        </div>

        {/* File Tabs */}
        <div className="flex flex-wrap items-center gap-2 mt-4 pt-4 border-t border-gray-800">
          {Object.keys(files).map((fileName) => (
            <button
              key={fileName}
              onClick={() => setActiveFile(fileName)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-mono transition ${
                activeFile === fileName
                  ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-semibold shadow-md'
                  : 'bg-slate-900 border border-gray-800 text-gray-400 hover:text-white'
              }`}
            >
              <FileText className="h-3.5 w-3.5" />
              <span>{fileName}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Code Viewer Panel */}
      <div className="glass-panel p-6 overflow-x-auto">
        <div className="flex items-center justify-between mb-4 border-b border-gray-800 pb-3">
          <h3 className="text-sm font-semibold text-gray-200">{current.title}</h3>
          <span className="text-xs font-mono bg-indigo-500/20 text-indigo-300 px-2 py-0.5 rounded border border-indigo-500/40">
            {current.section}
          </span>
        </div>

        <pre className="p-4 rounded-xl bg-slate-950 border border-gray-800 text-xs text-gray-200 font-mono leading-relaxed overflow-x-auto">
          <code>{current.code}</code>
        </pre>
      </div>
    </div>
  );
}
