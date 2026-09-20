import React, { useState, useMemo } from 'react';
import { computePositionalEncodingMatrix } from '../utils/transformerMath';
import { Zap, Info, Sliders } from 'lucide-react';

export default function PositionalEncodingVisualizer() {
  const [seqLen, setSeqLen] = useState(20);
  const [dModel, setDModel] = useState(64);
  const [selectedDim, setSelectedDim] = useState(0);

  const peMatrix = useMemo(() => {
    return computePositionalEncodingMatrix(seqLen, dModel);
  }, [seqLen, dModel]);

  // Map value [-1.0, 1.0] to a vibrant heatmap color
  const getPEColor = (val) => {
    // Map -1..1 to 0..1
    const norm = (val + 1) / 2;
    if (norm < 0.25) return 'bg-cyan-900 border-cyan-700 text-cyan-200';
    if (norm < 0.45) return 'bg-indigo-950 border-indigo-800 text-indigo-300';
    if (norm < 0.55) return 'bg-slate-900 border-gray-800 text-gray-500';
    if (norm < 0.75) return 'bg-purple-950 border-purple-800 text-purple-300';
    return 'bg-amber-500 border-amber-300 text-slate-950 font-bold';
  };

  return (
    <div className="space-y-6">
      {/* Header Card */}
      <div className="glass-panel p-6">
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Zap className="h-5 w-5 text-amber-400" />
              Sinusoidal Positional Encoding Visualizer
            </h2>
            <p className="text-xs text-gray-400 mt-1">
              Section 3.5: Since the model contains no recurrence or convolution, positional encodings inject token order into embeddings.
            </p>
          </div>

          <div className="flex items-center gap-4 bg-slate-900 px-4 py-2 rounded-xl border border-gray-800">
            <div className="flex items-center gap-2">
              <Sliders className="h-4 w-4 text-indigo-400" />
              <span className="text-xs text-gray-300 font-medium">Controls:</span>
            </div>
            
            <div className="flex items-center gap-2">
              <label className="text-xs text-gray-400">Seq Length ({seqLen}):</label>
              <input
                type="range"
                min="8"
                max="32"
                value={seqLen}
                onChange={(e) => setSeqLen(Number(e.target.value))}
                className="w-24 accent-indigo-500"
              />
            </div>

            <div className="flex items-center gap-2">
              <label className="text-xs text-gray-400">d_model ({dModel}):</label>
              <input
                type="range"
                min="16"
                max="64"
                step="8"
                value={dModel}
                onChange={(e) => setDModel(Number(e.target.value))}
                className="w-24 accent-purple-500"
              />
            </div>
          </div>
        </div>

        {/* Formulas */}
        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3 pt-4 border-t border-gray-800">
          <div className="p-3 rounded-lg bg-slate-900 border border-gray-800 text-xs font-mono text-cyan-300">
            <span className="text-gray-400 font-sans block mb-1">Even Dimensions (2i):</span>
            PE_(pos, 2i) = sin(pos / 10000^(2i / d_model))
          </div>
          <div className="p-3 rounded-lg bg-slate-900 border border-gray-800 text-xs font-mono text-amber-300">
            <span className="text-gray-400 font-sans block mb-1">Odd Dimensions (2i+1):</span>
            PE_(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
          </div>
        </div>
      </div>

      {/* Grid view of PE matrix */}
      <div className="glass-panel p-6 overflow-x-auto">
        <h3 className="text-sm font-semibold text-gray-200 mb-2">
          Positional Encoding Matrix heatmap (pos = 0..{seqLen - 1}, d = 0..{dModel - 1})
        </h3>
        <p className="text-xs text-gray-400 mb-4">
          Each row represents a sequence position. Each column represents an embedding dimension. Notice how low dimensions oscillate rapidly while high dimensions change slowly.
        </p>

        <div className="min-w-[600px] space-y-1">
          {/* Column Dimension Headers */}
          <div className="flex items-center pl-16 mb-1">
            {Array.from({ length: dModel }).map((_, i) => (
              <div
                key={i}
                onClick={() => setSelectedDim(i)}
                className={`flex-1 text-center text-[9px] font-mono cursor-pointer transition ${
                  selectedDim === i ? 'text-amber-400 font-bold' : 'text-gray-500 hover:text-gray-300'
                }`}
              >
                {i % 4 === 0 ? `d${i}` : '.'}
              </div>
            ))}
          </div>

          {/* PE Matrix Rows */}
          {peMatrix.map((row, pos) => (
            <div key={pos} className="flex items-center gap-1">
              <span className="w-14 text-right pr-2 text-xs font-mono text-gray-400">
                pos {pos}
              </span>

              {row.map((val, dim) => (
                <div
                  key={dim}
                  title={`pos ${pos}, dim ${dim}: ${val.toFixed(4)} (${dim % 2 === 0 ? 'sine' : 'cosine'})`}
                  className={`flex-1 h-5 rounded-[2px] border text-[9px] font-mono flex items-center justify-center transition hover:scale-125 hover:z-10 ${getPEColor(
                    val
                  )}`}
                >
                  {val > 0 ? '+' : '-'}
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Waveform curve preview card */}
      <div className="glass-panel p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-gray-200">
            Sinusoidal Frequency Curves for Dimension Pair ({selectedDim}, {selectedDim % 2 === 0 ? selectedDim + 1 : selectedDim - 1})
          </h3>
          <span className="text-xs text-indigo-300 font-mono">
            Wavelength λ ≈ 2π × 10000^({(selectedDim / dModel).toFixed(2)})
          </span>
        </div>

        {/* Waveform plot simulation */}
        <div className="h-40 bg-slate-950 rounded-xl border border-gray-800 p-4 flex items-end gap-1 relative overflow-hidden">
          <div className="absolute top-2 left-3 text-[10px] font-mono text-cyan-400 flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-cyan-400 inline-block" /> Sine Curve (Dim {selectedDim % 2 === 0 ? selectedDim : selectedDim - 1})
          </div>
          <div className="absolute top-2 right-3 text-[10px] font-mono text-amber-400 flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-amber-400 inline-block" /> Cosine Curve (Dim {selectedDim % 2 === 0 ? selectedDim + 1 : selectedDim})
          </div>

          {peMatrix.map((row, pos) => {
            const sineDim = selectedDim % 2 === 0 ? selectedDim : selectedDim - 1;
            const cosDim = sineDim + 1 < dModel ? sineDim + 1 : sineDim;
            const sinVal = row[sineDim] || 0;
            const cosVal = row[cosDim] || 0;

            const sinHeight = ((sinVal + 1) / 2) * 100;
            const cosHeight = ((cosVal + 1) / 2) * 100;

            return (
              <div key={pos} className="flex-1 flex flex-col items-center gap-1 h-full justify-end group">
                <div className="w-full flex items-end justify-center gap-0.5 h-28">
                  <div
                    className="w-1.5 bg-cyan-400 rounded-t transition-all group-hover:bg-cyan-300"
                    style={{ height: `${sinHeight}%` }}
                  />
                  <div
                    className="w-1.5 bg-amber-400 rounded-t transition-all group-hover:bg-amber-300"
                    style={{ height: `${cosHeight}%` }}
                  />
                </div>
                <span className="text-[9px] font-mono text-gray-500">{pos}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
