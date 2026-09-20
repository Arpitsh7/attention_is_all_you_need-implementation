import React, { useState } from 'react';
import { getCausalMaskMatrix } from '../utils/transformerMath';
import { Cpu, Play, RotateCcw, Lock, Check } from 'lucide-react';

export default function InferenceSandbox() {
  const [promptTokens, setPromptTokens] = useState(["<SOS>", "Attention", "is", "all"]);
  const [generatedTokens, setGeneratedTokens] = useState(["you", "need"]);
  const [currentStep, setCurrentStep] = useState(0);

  const candidatePool = [
    { token: "need", prob: 0.842 },
    { token: "required", prob: 0.091 },
    { token: "models", prob: 0.043 },
    { token: "that", prob: 0.015 },
    { token: "there", prob: 0.009 },
  ];

  const fullSequence = [...promptTokens, ...generatedTokens.slice(0, currentStep)];
  const causalMatrix = getCausalMaskMatrix(fullSequence.length);

  const handleNextStep = () => {
    if (currentStep < generatedTokens.length) {
      setCurrentStep((prev) => prev + 1);
    }
  };

  const handleReset = () => {
    setCurrentStep(0);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="glass-panel p-6">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Cpu className="h-5 w-5 text-cyan-400" />
              Autoregressive Decoding & Causal Mask Sandbox
            </h2>
            <p className="text-xs text-gray-400 mt-1">
              Section 3.1 & 3.2.3: The decoder generates one token at a time autoregressively. Future positions are masked with <code className="text-cyan-300">-inf</code> before softmax.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleNextStep}
              disabled={currentStep >= generatedTokens.length}
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-600 text-slate-950 font-bold text-xs shadow-lg shadow-cyan-500/20 disabled:opacity-40 transition"
            >
              <Play className="h-4 w-4 fill-current" />
              Generate Step {currentStep + 1}
            </button>

            <button
              onClick={handleReset}
              className="p-2 rounded-xl bg-slate-800 border border-gray-700 text-gray-300 hover:text-white transition"
              title="Reset Sandbox"
            >
              <RotateCcw className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Sequence Generation Flow */}
        <div className="lg:col-span-7 glass-panel p-6 space-y-4">
          <h3 className="text-sm font-semibold text-gray-200">
            Autoregressive Decoder Sequence Stream
          </h3>

          {/* Tokens pill container */}
          <div className="p-4 rounded-xl bg-slate-950 border border-gray-800 flex flex-wrap items-center gap-2 min-h-[70px]">
            {fullSequence.map((tok, idx) => {
              const isPrompt = idx < promptTokens.length;
              return (
                <div
                  key={idx}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold flex items-center gap-1.5 border transition-all ${
                    isPrompt
                      ? 'bg-slate-800 border-gray-700 text-gray-300'
                      : 'bg-cyan-500/20 border-cyan-400 text-cyan-300 shadow-md shadow-cyan-500/10'
                  }`}
                >
                  {isPrompt ? (
                    <span className="text-[10px] text-gray-500">[Context]</span>
                  ) : (
                    <span className="text-[10px] text-cyan-400 font-bold">[Gen #{idx - promptTokens.length + 1}]</span>
                  )}
                  <span>"{tok}"</span>
                </div>
              );
            })}

            {currentStep < generatedTokens.length && (
              <div className="px-3 py-1.5 rounded-lg text-xs font-mono text-gray-500 border border-dashed border-gray-700 animate-pulse">
                [Predicting next token...]
              </div>
            )}
          </div>

          {/* Causal Mask Matrix View */}
          <div className="pt-2">
            <h4 className="text-xs font-semibold text-gray-400 mb-2 flex items-center gap-1.5">
              <Lock className="h-3.5 w-3.5 text-indigo-400" />
              Causal Lower-Triangular Mask Matrix ({fullSequence.length} × {fullSequence.length})
            </h4>

            <div className="p-3 rounded-xl bg-slate-900 border border-gray-800 overflow-x-auto">
              <div className="space-y-1 min-w-[300px]">
                {causalMatrix.map((row, i) => (
                  <div key={i} className="flex items-center gap-1">
                    <span className="w-16 text-right pr-2 text-[10px] font-mono text-gray-500 truncate">
                      {fullSequence[i]}
                    </span>
                    {row.map((val, j) => (
                      <div
                        key={j}
                        className={`flex-1 h-6 rounded border text-[9px] font-mono flex items-center justify-center ${
                          val === 1
                            ? 'bg-indigo-600/80 border-indigo-400 text-white font-bold'
                            : 'bg-slate-950 border-gray-900 text-gray-700'
                        }`}
                      >
                        {val === 1 ? '1' : '-inf'}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Softmax Probability Distribution Card */}
        <div className="lg:col-span-5 glass-panel p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-semibold text-gray-200 mb-1">
              Top Softmax Output Predictions
            </h3>
            <p className="text-xs text-gray-400 mb-4">
              Logits projected through generator <code className="text-amber-300">W_vocab</code> at position {fullSequence.length}.
            </p>

            <div className="space-y-2.5">
              {candidatePool.map((cand, idx) => (
                <div
                  key={idx}
                  className={`p-3 rounded-xl border transition ${
                    idx === 0
                      ? 'bg-cyan-950/40 border-cyan-500/50 text-white'
                      : 'bg-slate-900/60 border-gray-800 text-gray-300'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5 text-xs font-mono">
                    <div className="flex items-center gap-2">
                      {idx === 0 && <Check className="h-3.5 w-3.5 text-cyan-400" />}
                      <span className="font-bold">"{cand.token}"</span>
                    </div>
                    <span className="text-cyan-400 font-semibold">{(cand.prob * 100).toFixed(1)}%</span>
                  </div>

                  <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        idx === 0 ? 'bg-gradient-to-r from-cyan-400 to-indigo-500' : 'bg-gray-600'
                      }`}
                      style={{ width: `${cand.prob * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-4 p-3 rounded-xl bg-slate-900 border border-gray-800 text-xs text-gray-400">
            Greedy sampling selects candidate #1 (<code className="text-cyan-300">"need"</code>) with highest probability.
          </div>
        </div>
      </div>
    </div>
  );
}
