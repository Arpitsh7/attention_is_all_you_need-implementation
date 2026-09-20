import React, { useState, useMemo } from 'react';
import { computeMultiHeadSelfAttention } from '../utils/transformerMath';
import { Eye, Info, RefreshCw, ArrowRight } from 'lucide-react';

export default function SelfAttentionVisualizer() {
  const [text, setText] = useState("The animal didn't cross the street because it was too tired");
  const [selectedHead, setSelectedHead] = useState(-1); // -1 for average, 0-7 for specific heads
  const [selectedTokenIndex, setSelectedTokenIndex] = useState(0);
  const [hoveredCell, setHoveredCell] = useState(null);

  const presets = [
    "The animal didn't cross the street because it was too tired",
    "Attention is all you need for sequence to sequence modeling",
    "Multi head attention computes scaled dot product in parallel",
  ];

  const { tokens, heads, avgMatrix } = useMemo(() => {
    return computeMultiHeadSelfAttention(text, 8);
  }, [text]);

  const activeMatrix = selectedHead === -1 ? avgMatrix : heads[selectedHead]?.matrix || avgMatrix;

  // Compute color based on softmax weight (0.0 to 1.0)
  const getCellColor = (val) => {
    if (val < 0.05) return 'bg-slate-900/60 border-slate-800 text-slate-600';
    if (val < 0.15) return 'bg-indigo-950/80 border-indigo-900/50 text-indigo-300';
    if (val < 0.30) return 'bg-indigo-800/80 border-indigo-600 text-indigo-100';
    if (val < 0.50) return 'bg-purple-600 border-purple-400 text-white font-semibold';
    if (val < 0.70) return 'bg-cyan-500 border-cyan-300 text-slate-950 font-bold';
    return 'bg-amber-400 border-amber-200 text-slate-950 font-extrabold shadow-lg shadow-amber-400/30';
  };

  return (
    <div className="space-y-6">
      {/* Header Info */}
      <div className="glass-panel p-6">
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4 mb-4">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Eye className="h-5 w-5 text-indigo-400" />
              Scaled Dot-Product & Multi-Head Self-Attention
            </h2>
            <p className="text-xs text-gray-400 mt-1">
              Equation 1: <code className="text-indigo-300">Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V</code>
            </p>
          </div>

          {/* Preset buttons */}
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-gray-400">Presets:</span>
            {presets.map((preset, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setText(preset);
                  setSelectedTokenIndex(0);
                }}
                className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 border border-gray-700 text-xs text-gray-300 transition"
              >
                Sample {idx + 1}
              </button>
            ))}
          </div>
        </div>

        {/* Text Input */}
        <div className="relative">
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type any sentence to see real-time attention weights..."
            className="w-full px-4 py-3 rounded-xl bg-slate-900 border border-gray-700 text-sm text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition"
          />
        </div>

        {/* Head Selector Bar */}
        <div className="flex flex-wrap items-center gap-2 mt-4 pt-4 border-t border-gray-800">
          <span className="text-xs font-semibold text-gray-400 mr-2">Attention Heads (h=8):</span>
          <button
            onClick={() => setSelectedHead(-1)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition ${
              selectedHead === -1
                ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-md'
                : 'bg-slate-800/80 text-gray-400 hover:text-white'
            }`}
          >
            Average (All 8 Heads)
          </button>
          {[0, 1, 2, 3, 4, 5, 6, 7].map((headIdx) => {
            const headLabels = [
              "H1: Adjacent", "H2: Coreference", "H3: Verbs", "H4: Syntax",
              "H5: Objects", "H6: Modifiers", "H7: Positional", "H8: Semantic"
            ];
            return (
              <button
                key={headIdx}
                onClick={() => setSelectedHead(headIdx)}
                className={`px-2.5 py-1.5 rounded-lg text-xs font-mono transition ${
                  selectedHead === headIdx
                    ? 'bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20'
                    : 'bg-slate-800/60 text-gray-400 hover:text-white border border-gray-800'
                }`}
              >
                {headLabels[headIdx]}
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Visualizer Split Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Heatmap Matrix Card */}
        <div className="lg:col-span-7 glass-panel p-6 overflow-x-auto">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-gray-200">
              Attention Weight Matrix {selectedHead === -1 ? '(Averaged)' : `(Head ${selectedHead + 1})`}
            </h3>
            {hoveredCell && (
              <span className="text-xs font-mono bg-indigo-500/20 border border-indigo-500/40 text-indigo-300 px-2 py-1 rounded-md">
                "{tokens[hoveredCell.i]}" → "{tokens[hoveredCell.j]}" : {(hoveredCell.val * 100).toFixed(1)}%
              </span>
            )}
          </div>

          <div className="min-w-[400px]">
            {/* Header Column Labels */}
            <div className="flex items-center mb-2 pl-24">
              {tokens.map((tok, j) => (
                <div
                  key={j}
                  className="flex-1 text-center text-[10px] font-mono text-gray-400 truncate px-0.5 transform -rotate-45 origin-bottom-left"
                  title={tok}
                >
                  {tok}
                </div>
              ))}
            </div>

            {/* Matrix Rows */}
            <div className="space-y-1">
              {tokens.map((tokQuery, i) => (
                <div key={i} className="flex items-center gap-1">
                  {/* Row Label (Query Token) */}
                  <button
                    onClick={() => setSelectedTokenIndex(i)}
                    className={`w-24 text-right pr-2 text-xs font-mono truncate transition rounded px-1 py-0.5 ${
                      selectedTokenIndex === i
                        ? 'text-cyan-400 font-bold bg-cyan-950/40 border-r-2 border-cyan-400'
                        : 'text-gray-400 hover:text-white'
                    }`}
                    title={tokQuery}
                  >
                    {tokQuery}
                  </button>

                  {/* Softmax Weights */}
                  {tokens.map((_, j) => {
                    const val = activeMatrix[i]?.[j] || 0;
                    return (
                      <div
                        key={j}
                        onMouseEnter={() => setHoveredCell({ i, j, val })}
                        onMouseLeave={() => setHoveredCell(null)}
                        className={`heatmap-cell flex-1 h-9 rounded flex items-center justify-center border text-[10px] font-mono cursor-pointer ${getCellColor(
                          val
                        )}`}
                      >
                        {(val * 100).toFixed(0)}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Interactive Token Alignment Card */}
        <div className="lg:col-span-5 glass-panel p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-semibold text-gray-200 mb-1">
              Token Attention Focus
            </h3>
            <p className="text-xs text-gray-400 mb-4">
              Select a Query token on the left to inspect which Key tokens receive the highest attention allocation.
            </p>

            <div className="p-3 rounded-xl bg-slate-900 border border-gray-800 mb-4">
              <span className="text-xs text-gray-400 block mb-1">Active Query Token:</span>
              <div className="flex items-center gap-2">
                <span className="text-lg font-bold font-mono text-cyan-400">
                  "{tokens[selectedTokenIndex]}"
                </span>
                <span className="text-xs text-gray-500 font-mono">
                  (Index {selectedTokenIndex})
                </span>
              </div>
            </div>

            {/* Token Attention Allocation List */}
            <div className="space-y-2 max-h-[320px] overflow-y-auto pr-1">
              {tokens
                .map((keyTok, j) => ({
                  token: keyTok,
                  weight: activeMatrix[selectedTokenIndex]?.[j] || 0,
                  index: j,
                }))
                .sort((a, b) => b.weight - a.weight)
                .map((item, rank) => (
                  <div
                    key={item.index}
                    className="flex items-center justify-between p-2 rounded-lg bg-slate-900/60 border border-gray-800/80 text-xs"
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-mono text-gray-500 w-4">#{rank + 1}</span>
                      <span className="font-mono text-gray-200">"{item.token}"</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="w-24 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-cyan-400 rounded-full"
                          style={{ width: `${item.weight * 100}%` }}
                        />
                      </div>
                      <span className="font-mono text-cyan-400 font-semibold w-12 text-right">
                        {(item.weight * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
            </div>
          </div>

          <div className="mt-4 p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-xs text-indigo-300 flex items-start gap-2">
            <Info className="h-4 w-4 text-indigo-400 shrink-0 mt-0.5" />
            <p>
              Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
