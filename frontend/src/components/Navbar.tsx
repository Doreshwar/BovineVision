import React from 'react';
import { Cpu, Zap } from 'lucide-react';


interface NavbarProps {
  onOpenModelInfo: () => void;
  device: string;
}

export const Navbar: React.FC<NavbarProps> = ({ onOpenModelInfo, device }) => {
  const isCuda = device.toLowerCase().includes('cuda');

  return (
    <header className="sticky top-0 z-40 bg-slate-900/90 backdrop-blur-md border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-amber-500 to-amber-300 flex items-center justify-center text-slate-950 shadow-lg shadow-amber-500/20 font-black text-xl">
            🐂
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-lg text-white tracking-tight">BovineVision</span>
              <span className="text-xs px-2 py-0.5 rounded-full font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/30">
                AI 1.0
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">
              Custom PyTorch CNN & Decoupled RAG Knowledge System
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Compute Acceleration Badge */}
          <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border ${
            isCuda
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
              : 'bg-slate-800 text-slate-300 border-slate-700'
          }`}>
            <Zap className="w-3.5 h-3.5" />
            <span>{isCuda ? 'NVIDIA CUDA GPU' : 'CPU Inference'}</span>
          </div>

          {/* Model Architecture Inspector Button */}
          <button
            onClick={onOpenModelInfo}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 hover:border-slate-600 transition-all shadow-sm"
          >
            <Cpu className="w-3.5 h-3.5 text-amber-400" />
            <span>Architecture & Metrics</span>
          </button>
        </div>
      </div>
    </header>
  );
};
