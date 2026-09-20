import React, { useState } from 'react';
import { Layers, ArrowDown, Cpu, Sparkles, CheckCircle } from 'lucide-react';

export default function ArchitectureInspector() {
  const [selectedBlock, setSelectedBlock] = useState('mha');

  const blocks = {
    embed: {
      title: 'Token Embeddings',
      section: 'Section 3.4',
      formula: 'Embedding(x) * sqrt(d_model)',
      inputShape: '(Batch_Size, Seq_Len)',
      outputShape: '(Batch_Size, Seq_Len, d_model = 512)',
      description:
        'Converts discrete input and target token IDs into dense d_model-dimensional vectors. In Section 3.4, embedding weights are multiplied by sqrt(d_model) to balance the magnitudes of embeddings and positional encodings.',
    },
    pe: {
      title: 'Positional Encoding',
      section: 'Section 3.5',
      formula: 'x + PE(pos, 2i)',
      inputShape: '(Batch_Size, Seq_Len, d_model)',
      outputShape: '(Batch_Size, Seq_Len, d_model)',
      description:
        'Adds fixed sinusoidal functions of sine and cosine across different frequencies so the model learns relative token positions without recurrence or convolutions.',
    },
    mha: {
      title: 'Multi-Head Self-Attention',
      section: 'Section 3.2.2',
      formula: 'MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O',
      inputShape: '(Batch_Size, Seq_Len, d_model)',
      outputShape: '(Batch_Size, Seq_Len, d_model)',
      description:
        'Linearly projects queries, keys, and values h=8 times with d_k = d_v = 64 dimensions. Allows the model to jointly attend to information from different representation subspaces at different positions.',
    },
    ffn: {
      title: 'Position-wise Feed-Forward Network',
      section: 'Section 3.3',
      formula: 'FFN(x) = max(0, x W_1 + b_1) W_2 + b_2',
      inputShape: '(Batch_Size, Seq_Len, d_model = 512)',
      outputShape: '(Batch_Size, Seq_Len, d_model = 512)',
      description:
        'Applied to each position separately and identically. Consists of two linear transformations with a ReLU activation in between. Inner layer dimensionality d_ff = 2048.',
    },
    layernorm: {
      title: 'Residual Connections & Layer Normalization',
      section: 'Section 3.1',
      formula: 'LayerNorm(x + Sublayer(x))',
      inputShape: '(Batch_Size, Seq_Len, d_model)',
      outputShape: '(Batch_Size, Seq_Len, d_model)',
      description:
        'A residual connection surrounds each sub-layer, followed by layer normalization. Facilitates deep gradient flow during training.',
    },
    masked_mha: {
      title: 'Masked Multi-Head Self-Attention',
      section: 'Section 3.2.3',
      formula: 'Attention(Q, K, V) with Causal Mask (-inf for future positions)',
      inputShape: '(Batch_Size, Tgt_Seq_Len, d_model)',
      outputShape: '(Batch_Size, Tgt_Seq_Len, d_model)',
      description:
        'Ensures predictions for position i depend only on known outputs at positions less than i by masking out future tokens before softmax.',
    },
    cross_mha: {
      title: 'Encoder-Decoder Cross-Attention',
      section: 'Section 3.2.3',
      formula: 'Q = Decoder Layer Output, K & V = Encoder Final Memory Output',
      inputShape: 'Decoder: (Batch, Tgt_Len, 512), Memory: (Batch, Src_Len, 512)',
      outputShape: '(Batch_Size, Tgt_Seq_Len, d_model)',
      description:
        'Queries come from the previous decoder layer, while Keys and Values come from the final encoder output. Allows every decoder position to attend to all positions in the input sequence.',
    },
    generator: {
      title: 'Linear Projection & Softmax Output',
      section: 'Section 3.4',
      formula: 'Softmax(x W_proj + b)',
      inputShape: '(Batch_Size, Tgt_Seq_Len, d_model)',
      outputShape: '(Batch_Size, Tgt_Seq_Len, Vocab_Size)',
      description:
        'Linear transformation converts decoder output vectors into log-odds probabilities over the target vocabulary, followed by softmax.',
    },
  };

  const current = blocks[selectedBlock] || blocks.mha;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="glass-panel p-6">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <Layers className="h-5 w-5 text-indigo-400" />
          Interactive Transformer Architecture Diagram (Vaswani et al. Figure 1)
        </h2>
        <p className="text-xs text-gray-400 mt-1">
          Click any block in the Encoder or Decoder stack to inspect tensor shape transformations, formulas, and layer specifications.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Visual Diagram Column */}
        <div className="lg:col-span-7 glass-panel p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Encoder Column */}
            <div className="space-y-3 p-4 rounded-xl bg-slate-900/80 border border-indigo-500/20">
              <div className="flex items-center justify-between border-b border-indigo-500/30 pb-2">
                <span className="text-xs font-bold text-indigo-400 uppercase tracking-wider">
                  Encoder Stack (N = 6)
                </span>
                <span className="text-[10px] bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 px-2 py-0.5 rounded">
                  Source Input
                </span>
              </div>

              {/* FFN Sublayer */}
              <button
                onClick={() => setSelectedBlock('ffn')}
                className={`w-full p-3 rounded-lg border text-left text-xs font-medium transition ${
                  selectedBlock === 'ffn'
                    ? 'bg-purple-600 border-purple-400 text-white shadow-lg shadow-purple-500/20'
                    : 'bg-slate-800/80 border-gray-700 text-gray-200 hover:border-purple-500'
                }`}
              >
                Feed Forward Network (d_ff = 2048)
              </button>

              <button
                onClick={() => setSelectedBlock('layernorm')}
                className={`w-full p-2 rounded-lg border text-center text-[11px] font-mono transition ${
                  selectedBlock === 'layernorm'
                    ? 'bg-indigo-600 border-indigo-400 text-white'
                    : 'bg-slate-800/40 border-gray-800 text-gray-400 hover:text-gray-200'
                }`}
              >
                Add & Norm
              </button>

              {/* MHA Sublayer */}
              <button
                onClick={() => setSelectedBlock('mha')}
                className={`w-full p-3 rounded-lg border text-left text-xs font-medium transition ${
                  selectedBlock === 'mha'
                    ? 'bg-indigo-600 border-indigo-400 text-white shadow-lg shadow-indigo-500/20'
                    : 'bg-slate-800/80 border-gray-700 text-gray-200 hover:border-indigo-500'
                }`}
              >
                Multi-Head Self-Attention (h = 8)
              </button>

              <button
                onClick={() => setSelectedBlock('layernorm')}
                className={`w-full p-2 rounded-lg border text-center text-[11px] font-mono transition ${
                  selectedBlock === 'layernorm'
                    ? 'bg-indigo-600 border-indigo-400 text-white'
                    : 'bg-slate-800/40 border-gray-800 text-gray-400 hover:text-gray-200'
                }`}
              >
                Add & Norm
              </button>

              <div className="flex justify-center my-1">
                <ArrowDown className="h-4 w-4 text-indigo-400 animate-bounce" />
              </div>

              {/* PE & Embedding */}
              <button
                onClick={() => setSelectedBlock('pe')}
                className={`w-full p-2.5 rounded-lg border text-left text-xs transition ${
                  selectedBlock === 'pe'
                    ? 'bg-amber-500 text-slate-950 font-bold border-amber-300'
                    : 'bg-slate-800/60 border-gray-700 text-gray-300 hover:border-amber-500'
                }`}
              >
                + Positional Encoding
              </button>

              <button
                onClick={() => setSelectedBlock('embed')}
                className={`w-full p-2.5 rounded-lg border text-left text-xs transition ${
                  selectedBlock === 'embed'
                    ? 'bg-cyan-500 text-slate-950 font-bold border-cyan-300'
                    : 'bg-slate-800/60 border-gray-700 text-gray-300 hover:border-cyan-500'
                }`}
              >
                Input Embedding (scaled by √d_model)
              </button>
            </div>

            {/* Decoder Column */}
            <div className="space-y-3 p-4 rounded-xl bg-slate-900/80 border border-purple-500/20">
              <div className="flex items-center justify-between border-b border-purple-500/30 pb-2">
                <span className="text-xs font-bold text-purple-400 uppercase tracking-wider">
                  Decoder Stack (N = 6)
                </span>
                <span className="text-[10px] bg-purple-500/10 text-purple-300 border border-purple-500/20 px-2 py-0.5 rounded">
                  Target Output
                </span>
              </div>

              <button
                onClick={() => setSelectedBlock('generator')}
                className={`w-full p-2.5 rounded-lg border text-left text-xs font-bold transition ${
                  selectedBlock === 'generator'
                    ? 'bg-emerald-500 border-emerald-300 text-slate-950'
                    : 'bg-slate-800/80 border-gray-700 text-gray-200 hover:border-emerald-500'
                }`}
              >
                Linear & Softmax Generator
              </button>

              <button
                onClick={() => setSelectedBlock('ffn')}
                className={`w-full p-2.5 rounded-lg border text-left text-xs font-medium transition ${
                  selectedBlock === 'ffn'
                    ? 'bg-purple-600 border-purple-400 text-white'
                    : 'bg-slate-800/80 border-gray-700 text-gray-200 hover:border-purple-500'
                }`}
              >
                Feed Forward Network
              </button>

              {/* Cross Attention Block */}
              <button
                onClick={() => setSelectedBlock('cross_mha')}
                className={`w-full p-3 rounded-lg border text-left text-xs font-medium transition ${
                  selectedBlock === 'cross_mha'
                    ? 'bg-cyan-500 border-cyan-300 text-slate-950 font-bold shadow-lg shadow-cyan-500/20'
                    : 'bg-slate-800/80 border-gray-700 text-gray-200 hover:border-cyan-500'
                }`}
              >
                Encoder-Decoder Cross-Attention
              </button>

              {/* Masked MHA Block */}
              <button
                onClick={() => setSelectedBlock('masked_mha')}
                className={`w-full p-3 rounded-lg border text-left text-xs font-medium transition ${
                  selectedBlock === 'masked_mha'
                    ? 'bg-indigo-600 border-indigo-400 text-white shadow-lg shadow-indigo-500/20'
                    : 'bg-slate-800/80 border-gray-700 text-gray-200 hover:border-indigo-500'
                }`}
              >
                Masked Multi-Head Self-Attention
              </button>

              <button
                onClick={() => setSelectedBlock('embed')}
                className={`w-full p-2.5 rounded-lg border text-left text-xs transition ${
                  selectedBlock === 'embed'
                    ? 'bg-cyan-500 text-slate-950 font-bold border-cyan-300'
                    : 'bg-slate-800/60 border-gray-700 text-gray-300'
                }`}
              >
                Output Embedding (Shifted Right)
              </button>
            </div>
          </div>
        </div>

        {/* Selected Component Specification Card */}
        <div className="lg:col-span-5 glass-panel p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3 border-b border-gray-800 pb-3">
              <h3 className="text-base font-bold text-white">{current.title}</h3>
              <span className="text-xs px-2 py-1 rounded bg-indigo-500/20 border border-indigo-500/40 text-indigo-300 font-mono">
                {current.section}
              </span>
            </div>

            {/* Formula box */}
            <div className="p-3 rounded-xl bg-slate-900 border border-gray-800 mb-4 font-mono text-xs text-amber-300">
              <span className="text-[10px] text-gray-500 block mb-1">Mathematical Formula:</span>
              {current.formula}
            </div>

            {/* Tensor Shapes */}
            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="p-3 rounded-xl bg-slate-900/60 border border-gray-800/80">
                <span className="text-[10px] text-gray-400 block mb-0.5">Input Tensor Shape:</span>
                <span className="text-xs font-mono text-cyan-400 font-semibold">{current.inputShape}</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/60 border border-gray-800/80">
                <span className="text-[10px] text-gray-400 block mb-0.5">Output Tensor Shape:</span>
                <span className="text-xs font-mono text-emerald-400 font-semibold">{current.outputShape}</span>
              </div>
            </div>

            {/* Description */}
            <p className="text-xs text-gray-300 leading-relaxed bg-slate-900/40 p-4 rounded-xl border border-gray-800">
              {current.description}
            </p>
          </div>

          <div className="mt-4 p-3 rounded-xl bg-slate-900 border border-gray-800 text-xs text-gray-400 flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-amber-400 shrink-0" />
            <span>Hyperparameters: d_model = 512, h = 8, d_k = 64, d_ff = 2048, Dropout = 0.1</span>
          </div>
        </div>
      </div>
    </div>
  );
}
