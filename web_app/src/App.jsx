import React, { useState } from 'react';
import Navbar from './components/Navbar';
import SelfAttentionVisualizer from './components/SelfAttentionVisualizer';
import PositionalEncodingVisualizer from './components/PositionalEncodingVisualizer';
import ArchitectureInspector from './components/ArchitectureInspector';
import InferenceSandbox from './components/InferenceSandbox';
import CodeExplorer from './components/CodeExplorer';

export default function App() {
  const [activeTab, setActiveTab] = useState('attention');

  const renderContent = () => {
    switch (activeTab) {
      case 'attention':
        return <SelfAttentionVisualizer />;
      case 'positional':
        return <PositionalEncodingVisualizer />;
      case 'architecture':
        return <ArchitectureInspector />;
      case 'inference':
        return <InferenceSandbox />;
      case 'code':
        return <CodeExplorer />;
      default:
        return <SelfAttentionVisualizer />;
    }
  };

  return (
    <div className="min-h-screen flex flex-col justify-between">
      <div>
        <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />
        <main className="max-w-7xl mx-auto px-6 py-8">
          {renderContent()}
        </main>
      </div>

      {/* Footer */}
      <footer className="border-t border-gray-800/80 bg-slate-950/80 py-6 px-6 mt-12">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-gray-500">
          <div>
            <span className="font-semibold text-gray-400">Attention Is All You Need Implementation</span>
            <span className="mx-2">•</span>
            <span>Vaswani et al. (2017) NIPS 2017</span>
          </div>

          <div className="flex items-center gap-4 font-mono">
            <span>PyTorch 2.x</span>
            <span>React + Vite</span>
            <span className="text-emerald-400 font-semibold">100% Equation Fidelity</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
