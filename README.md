# ⚡ Attention Is All You Need — PyTorch & Interactive Visualizer

[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![React + Vite](https://img.shields.io/badge/React-18-61dafb?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![Tests Passing](https://img.shields.io/badge/Tests-5%2F5%20Passed-10b981?style=for-the-badge&logo=pytest&logoColor=white)](#-testing--verification)
[![Paper Citation](https://img.shields.io/badge/Vaswani--et--al.-2017-6366f1?style=for-the-badge&logo=google-scholar&logoColor=white)](https://arxiv.org/abs/1706.03762)
[![License: MIT](https://img.shields.io/badge/License-MIT-amber?style=for-the-badge)](LICENSE)

A clean, modular **PyTorch implementation** of the landmark Transformer architecture from **"Attention Is All You Need"** (Vaswani et al., NIPS 2017), paired with a **Stunning Interactive Web Visualizer** to inspect multi-head self-attention matrices, positional encoding waveforms, tensor flow diagrams, and step-by-step autoregressive generation.

---

## 📌 Table of Contents
- [✨ Key Features](#-key-features)
- [🏗️ Transformer Architecture](#️-transformer-architecture)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [PyTorch Core Usage](#pytorch-core-usage)
  - [Running the Interactive Web Visualizer](#running-the-interactive-web-visualizer)
- [🧪 Testing & Verification](#-testing--verification)
- [📐 Mathematical Equations & Paper Mapping](#-mathematical-equations--paper-mapping)
- [📜 Citation & References](#-citation--references)
- [📄 License](#-license)

---

## ✨ Key Features

### 🧠 1. Modular PyTorch Core (`pytorch_src/`)
- **Exact Equation Fidelity**: Every tensor operation directly matches the formulas in Vaswani et al. (2017).
- **Scaled Dot-Product Attention**: $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ with custom mask support.
- **Multi-Head Attention**: $h=8$ parallel attention projection heads with dimension $d_k = d_v = 64$.
- **Sinusoidal Positional Encoding**: Fixed sine and cosine frequencies for position representation.
- **Position-wise Feed-Forward Networks**: Two linear projections with inner dimension $d_{ff} = 2048$ and ReLU activation.
- **Residual Connections & LayerNorm**: Pre/Post-LN support wrapping each sub-layer.
- **Noam LR Scheduler & Label Smoothing**: Learning rate warmup and KL-divergence label smoothing loss.
- **Autoregressive Greedy Decoding**: Fast sequence generation helper for inference.

### 🌐 2. Interactive Web Visualizer (`web_app/`)
- **Self-Attention Matrix Heatmap**: Input custom sentences and visualize real-time softmax attention heatmaps per head.
- **Token Focus Inspection**: Select any Query token to see ranked Key token allocations.
- **Positional Encoding Explorer**: 2D heatmaps and interactive sine/cosine frequency wave charts.
- **Transformer Block Inspector**: Clickable block diagram (Vaswani et al. Figure 1) with live tensor shape tracking.
- **Autoregressive Sandbox**: Step-by-step decoding simulator with causal lower-triangular mask visualizer.
- **PyTorch Code Explorer**: Interactive source viewer highlighting line-by-line equation mappings.

---

## 🏗️ Transformer Architecture

```
                          ┌───────────────────────────┐
                          │   Output Probabilities    │
                          └─────────────▲─────────────┘
                                        │
                          ┌─────────────┴─────────────┐
                          │   Linear & Softmax Gen    │
                          └─────────────▲─────────────┘
                                        │
                                        │ (Decoder Output)
                          ┌─────────────┴─────────────┐
                          │    Decoder Stack (N=6)    │
                          │ ┌───────────────────────┐ │
                          │ │ Position-wise FFN     │ │
                          │ ├───────────────────────┤ │
                          │ │ Cross-Attention       │◄┼─────────┐
                          │ ├───────────────────────┤ │         │
                          │ │ Masked Self-Attention │ │         │
                          │ └───────────────────────┘ │         │
                          └─────────────▲─────────────┘         │
                                        │                       │ (Encoder Memory)
                                        │ (Target Embeddings)   │
                          ┌─────────────┴─────────────┐         │
                          │ Target Embedding + PE     │         │
                          └─────────────▲─────────────┘         │
                                        │                       │
                                 Target Sequence                │
                                                                │
                          ┌───────────────────────────┐         │
                          │    Encoder Stack (N=6)    ├─────────┘
                          │ ┌───────────────────────┐ │
                          │ │ Position-wise FFN     │ │
                          │ ├───────────────────────┤ │
                          │ │ Multi-Head Self-Attn  │ │
                          │ └───────────────────────┘ │
                          └─────────────▲─────────────┘
                                        │
                                        │ (Source Embeddings)
                          ┌─────────────┴─────────────┐
                          │  Source Embedding + PE    │
                          └─────────────▲─────────────┘
                                        │
                                 Source Sequence
```

---

## 📁 Project Structure

```text
attention-is-all-you-need/
├── pytorch_src/                   # Modular PyTorch Core Implementation
│   ├── __init__.py                # Package initialization & exports
│   ├── attention.py               # ScaledDotProductAttention & MultiHeadAttention
│   ├── positional_encoding.py     # Sinusoidal Positional Encoding module
│   ├── feed_forward.py            # PositionwiseFeedForward network module
│   ├── encoder.py                 # EncoderLayer and TransformerEncoder stack
│   ├── decoder.py                 # DecoderLayer and TransformerDecoder stack
│   ├── transformer.py             # Top-level Transformer Seq2Seq model & masking
│   ├── dataset.py                 # Synthetic sequence dataset for copy/reverse tasks
│   ├── train.py                   # NoamOpt, LabelSmoothingLoss, and training loop
│   └── test_transformer.py        # PyTorch unit test suite
├── web_app/                       # Interactive React + Vite Web Visualizer
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx                      # Navigation header & paper metadata
│   │   │   ├── SelfAttentionVisualizer.jsx     # Attention heatmaps & token focus
│   │   │   ├── PositionalEncodingVisualizer.jsx# Waveform plots & 2D PE matrix
│   │   │   ├── ArchitectureInspector.jsx       # Interactive Figure 1 diagram
│   │   │   ├── InferenceSandbox.jsx            # Autoregressive generation sandbox
│   │   │   └── CodeExplorer.jsx                # Interactive PyTorch code browser
│   │   ├── utils/
│   │   │   └── transformerMath.js              # Mathematical routines & tokenizers
│   │   ├── App.jsx                             # Main visualizer container
│   │   └── index.css                           # Modern dark glassmorphism design system
│   ├── index.html
│   └── package.json
├── requirements.txt               # PyTorch Python dependencies
└── README.md                      # GitHub Repository Documentation
```

---

## 🚀 Getting Started

### Prerequisites
- **Python**: `3.9+` with PyTorch `2.0+`
- **Node.js**: `18.0+` & `npm`

---

### PyTorch Core Usage

#### Installation
```bash
git clone https://github.com/your-username/attention-is-all-you-need.git
cd attention-is-all-you-need
pip install -r requirements.txt
```

#### Quick Start Code Example
```python
import torch
from pytorch_src import Transformer

# Initialize Transformer model
model = Transformer(
    src_vocab_size=1000,
    tgt_vocab_size=1000,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048,
    dropout=0.1
)

# Dummy inputs: batch_size=2, src_len=10, tgt_len=12
src = torch.randint(1, 1000, (2, 10))
tgt = torch.randint(1, 1000, (2, 12))

# Forward pass
logits, attn_info = model(src, tgt)

print("Logits shape:", logits.shape) 
# Output: torch.Size([2, 12, 1000])

print("Encoder layers attention count:", len(attn_info["encoder_attentions"]))
# Output: 6
```

#### Running Synthetic Model Training
To train the Transformer on a sequence copy task:
```bash
python -m pytorch_src.train
```

---

### Running the Interactive Web Visualizer

```bash
cd web_app
npm install
npm run dev
```

Open your browser at **`http://localhost:5173`** (or the URL printed in terminal) to explore:
1. **Multi-Head Self-Attention Matrix Heatmaps**: Type custom sentences and inspect attention weights across all 8 heads.
2. **Positional Encoding Frequency Curves**: Explore how sine/cosine wavelengths vary with dimension $d_{model}$.
3. **Architecture Block Flow**: Click through the Transformer Encoder and Decoder layers.
4. **Step-by-Step Autoregressive Decoder**: Visualize token generation with causal masking.

---

## 🧪 Testing & Verification

The repository includes a comprehensive PyTorch `unittest` suite covering shape invariants, gradient backpropagation, causal masking logic, and numerical correctness.

To run tests:
```bash
python -m unittest pytorch_src.test_transformer
```

### Test Suite Summary
| Test Case | Description | Status |
| :--- | :--- | :---: |
| `test_scaled_dot_product_attention` | Verifies attention output tensor shapes and row-wise softmax normalization ($\sum \text{softmax} = 1.0$). | ✅ PASS |
| `test_multi_head_attention` | Verifies head splitting and concatenation dimensions across dynamic sequence lengths. | ✅ PASS |
| `test_positional_encoding` | Validates sinusoidal tensor additions and value bounds. | ✅ PASS |
| `test_causal_mask` | Ensures lower-triangular causal mask prevents future token attention leakage. | ✅ PASS |
| `test_full_transformer_forward_backward` | Verifies full model end-to-end forward pass and non-NaN gradient backward pass. | ✅ PASS |

---

## 📐 Mathematical Equations & Paper Mapping

| Section | Topic | Equation |
| :--- | :--- | :--- |
| **Section 3.2.1** | Scaled Dot-Product Attention | $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$ |
| **Section 3.2.2** | Multi-Head Attention | $$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O \quad \text{where } \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$ |
| **Section 3.3** | Position-wise Feed-Forward | $$FFN(x) = \max(0, x W_1 + b_1) W_2 + b_2$$ |
| **Section 3.4** | Embeddings & Softmax Scale | $\text{Embedding Weights multiplied by } \sqrt{d_{model}}$ |
| **Section 3.5** | Positional Encoding | $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$ |
| **Section 5.3** | Optimizer & LR Schedule | $$lrate = d_{model}^{-0.5} \cdot \min\left(step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5}\right)$$ |

---

## 📜 Citation & References

If you find this codebase or visualizer useful in your research or learning, please cite the original paper:

```bibtex
@inproceedings{vaswani2017attention,
  title     = {Attention is All you Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems (NIPS)},
  volume    = {30},
  pages     = {5998--6008},
  year      = {2017},
  url       = {https://arxiv.org/abs/1706.03762}
}
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
