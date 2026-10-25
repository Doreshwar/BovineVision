import React from 'react';
import { AlertTriangle, CheckCircle2, ShieldAlert, BarChart3, Info } from 'lucide-react';
import type { ClassificationResult } from '../types';


interface PredictionResultsProps {
  result: ClassificationResult | null;
  isLoading: boolean;
}

export const PredictionResults: React.FC<PredictionResultsProps> = ({ result, isLoading }) => {
  if (isLoading) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4 animate-pulse">
        <div className="h-6 bg-slate-800 rounded w-1/3"></div>
        <div className="h-24 bg-slate-800/60 rounded-xl"></div>
        <div className="space-y-2">
          <div className="h-10 bg-slate-800/40 rounded-lg"></div>
          <div className="h-10 bg-slate-800/40 rounded-lg"></div>
          <div className="h-10 bg-slate-800/40 rounded-lg"></div>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-8 text-center flex flex-col items-center justify-center gap-3 min-h-[300px]">
        <div className="w-12 h-12 rounded-xl bg-slate-800/50 flex items-center justify-center text-slate-500">
          <BarChart3 className="w-6 h-6" />
        </div>
        <div>
          <h3 className="text-sm font-semibold text-slate-300">Awaiting Image Input</h3>
          <p className="text-xs text-slate-500 mt-1 max-w-sm">
            Upload an authentic cattle photo and click analyze to see the custom PyTorch CNN's top 3 predictions.
          </p>
        </div>
      </div>
    );
  }

  const { is_accepted, rejection_message, threshold_percentage, top_prediction, top_3_predictions } = result;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-5">
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-amber-400" />
            Model Prediction Results
          </h2>
          <p className="text-xs text-slate-400">
            Top 3 candidate classifications evaluated against {threshold_percentage}% threshold
          </p>
        </div>

        {/* Accepted vs Rejected Badge */}
        {is_accepted ? (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 shadow-sm">
            <CheckCircle2 className="w-3.5 h-3.5" />
            Validated Identification
          </span>
        ) : (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-rose-500/10 text-rose-400 border border-rose-500/30 shadow-sm">
            <ShieldAlert className="w-3.5 h-3.5" />
            Detection Rejected
          </span>
        )}
      </div>

      {/* REJECTION BANNER (User requirement: "Could not detect, try another picture") */}
      {!is_accepted && (
        <div className="bg-rose-950/40 border border-rose-800/80 rounded-xl p-5 space-y-3 shadow-lg">
          <div className="flex items-start gap-3">
            <div className="p-2 rounded-lg bg-rose-900/50 text-rose-300 shrink-0">
              <AlertTriangle className="w-5 h-5" />
            </div>
            <div className="space-y-1">
              <h3 className="text-base font-extrabold text-rose-200 tracking-tight">
                {rejection_message || "Could not detect, try another picture"}
              </h3>
              <p className="text-xs text-rose-300/90 leading-relaxed">
                The highest model confidence score did not meet your required threshold of{' '}
                <strong className="text-rose-100">{threshold_percentage}%</strong>.
                To prevent incorrect breed classifications, this image has been rejected.
              </p>
            </div>
          </div>
          <div className="text-[11px] bg-rose-950/70 border border-rose-900/60 rounded-lg p-2.5 text-rose-300 flex items-center gap-2">
            <Info className="w-3.5 h-3.5 shrink-0 text-rose-400" />
            <span>
              <strong>RAG Policy Enforced:</strong> As per design constraints, unverified predictions do not trigger the RAG breed intelligence system.
            </span>
          </div>
        </div>
      )}

      {/* TOP PREDICTION HIGHLIGHT CARD (When Accepted) */}
      {is_accepted && top_prediction && (
        <div className="bg-gradient-to-br from-amber-500/10 via-slate-950 to-slate-900 border border-amber-500/30 rounded-xl p-5 shadow-md flex items-center justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500 text-slate-950 uppercase tracking-wider">
                Top Prediction
              </span>
              <span className="text-xs text-slate-400">Rank #1 Candidate</span>
            </div>
            <h3 className="text-2xl font-black text-white tracking-tight flex items-center gap-2">
              🐂 {top_prediction.breed}
            </h3>
            <p className="text-xs text-emerald-400 font-semibold flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              Passed threshold criteria ({top_prediction.confidence_percentage}% &gt;= {threshold_percentage}%)
            </p>
          </div>
          <div className="text-right">
            <span className="text-3xl font-black text-amber-400 tracking-tight">
              {top_prediction.confidence_percentage}%
            </span>
            <p className="text-[11px] text-slate-400 font-medium">Confidence Score</p>
          </div>
        </div>
      )}

      {/* TOP 3 PREDICTIONS PROGRESS LIST */}
      <div className="space-y-3">
        <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center justify-between">
          <span>Top Three Predictions</span>
          <span className="text-[11px] font-normal lowercase text-slate-500">
            {is_accepted ? 'ranked by softmax output' : 'raw candidate outputs'}
          </span>
        </label>

        <div className="space-y-2.5">
          {top_3_predictions.map((pred) => {
            const isTop = pred.rank === 1;
            const crossesThreshold = pred.confidence_percentage >= threshold_percentage;

            return (
              <div
                key={pred.breed}
                className={`p-3.5 rounded-xl border transition-all ${
                  isTop && is_accepted
                    ? 'bg-slate-950 border-amber-500/40 shadow-sm'
                    : 'bg-slate-950/60 border-slate-800'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-2.5">
                    <span className={`w-6 h-6 rounded-md flex items-center justify-center text-xs font-bold ${
                      pred.rank === 1
                        ? 'bg-amber-500 text-slate-950'
                        : pred.rank === 2
                        ? 'bg-slate-700 text-slate-200'
                        : 'bg-slate-800 text-slate-400'
                    }`}>
                      #{pred.rank}
                    </span>
                    <span className="text-sm font-bold text-slate-200">
                      {pred.breed}
                    </span>
                    {crossesThreshold ? (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium">
                        &gt;= {threshold_percentage}%
                      </span>
                    ) : (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 font-medium">
                        &lt; {threshold_percentage}%
                      </span>
                    )}
                  </div>

                  <span className={`text-sm font-extrabold ${
                    isTop && is_accepted ? 'text-amber-400' : 'text-slate-300'
                  }`}>
                    {pred.confidence_percentage}%
                  </span>
                </div>

                {/* Animated Confidence Bar */}
                <div className="w-full bg-slate-800/80 rounded-full h-2 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-700 ease-out ${
                      isTop && is_accepted
                        ? 'bg-gradient-to-r from-amber-500 to-amber-400'
                        : crossesThreshold
                        ? 'bg-emerald-500'
                        : 'bg-slate-600'
                    }`}
                    style={{ width: `${Math.min(100, Math.max(3, pred.confidence_percentage))}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
