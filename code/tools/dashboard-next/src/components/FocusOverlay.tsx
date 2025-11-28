'use client';

import { Benchmark } from '@/types';
import { formatMs } from '@/lib/utils';
import { X, Target, Clock, Activity } from 'lucide-react';

interface FocusOverlayProps {
  isOpen: boolean;
  benchmark: Benchmark | null;
  onClose: () => void;
}

export function FocusOverlay({ isOpen, benchmark, onClose }: FocusOverlayProps) {
  if (!isOpen) return null;

  const timeSaved =
    benchmark && benchmark.baseline_time_ms && benchmark.optimized_time_ms
      ? benchmark.baseline_time_ms - benchmark.optimized_time_ms
      : null;
  const displaySpeedup = benchmark?.speedup;
  const rawSpeedup = benchmark?.raw_speedup ?? displaySpeedup;
  const isCapped = benchmark?.speedup_capped && rawSpeedup !== displaySpeedup;

  return (
    <div className="fixed inset-0 z-[9999] bg-black/80 backdrop-blur-sm flex flex-col">
      <div className="flex items-center justify-between px-8 py-6 border-b border-white/10">
        <div className="flex items-center gap-3">
          <Target className="w-6 h-6 text-accent-primary" />
          <div>
            <div className="text-sm uppercase tracking-wide text-white/40">Focus Mode</div>
            <div className="text-2xl font-bold text-white">
              {benchmark ? `${benchmark.chapter}: ${benchmark.name}` : 'Top Benchmark'}
            </div>
          </div>
        </div>
        <button
          onClick={onClose}
          className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-white"
        >
          <X className="w-4 h-4" />
          Exit
        </button>
      </div>

      <div className="flex-1 overflow-auto px-8 py-8">
        {benchmark ? (
          <div className="max-w-5xl mx-auto space-y-6">
            <div className="card">
              <div className="card-body text-center py-12">
                <div className="text-6xl font-black" style={{ color: (displaySpeedup ?? 0) >= 1 ? '#00f5a0' : '#ff4757' }}>
                  {displaySpeedup?.toFixed(2)}x
                </div>
                <div className="text-lg text-white/60 mt-2">
                  Speedup Achieved
                  {isCapped && rawSpeedup !== undefined && (
                    <span className="ml-2 text-xs text-white/50">(raw {rawSpeedup.toFixed(2)}x)</span>
                  )}
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
                  <div className="p-4 bg-white/5 rounded-xl">
                    <div className="text-sm text-white/50 mb-1">Baseline</div>
                    <div className="text-2xl font-bold text-accent-tertiary">
                      {formatMs(benchmark.baseline_time_ms)}
                    </div>
                  </div>
                  <div className="p-4 bg-white/5 rounded-xl">
                    <div className="text-sm text-white/50 mb-1">Optimized</div>
                    <div className="text-2xl font-bold text-accent-success">
                      {formatMs(benchmark.optimized_time_ms)}
                    </div>
                  </div>
                  <div className="p-4 bg-white/5 rounded-xl">
                    <div className="text-sm text-white/50 mb-1">Time Saved</div>
                    <div className="text-2xl font-bold text-accent-primary">
                      {timeSaved !== null ? formatMs(timeSaved) : '—'}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="card">
                <div className="card-header">
                  <h3 className="font-medium text-white">Details</h3>
                </div>
                <div className="card-body space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-white/60">Chapter</span>
                    <span className="px-3 py-1 rounded-full bg-accent-secondary/10 text-accent-secondary text-sm">
                      {benchmark.chapter}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-white/60">Type</span>
                    <span className="text-white">{benchmark.type || 'N/A'}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-white/60">Status</span>
                    <span className="text-accent-success font-semibold">{benchmark.status}</span>
                  </div>
                </div>
              </div>

              <div className="card">
                <div className="card-header">
                  <h3 className="font-medium text-white">Highlights</h3>
                </div>
                <div className="card-body space-y-3">
                  <div className="flex items-center gap-3">
                    <Clock className="w-5 h-5 text-accent-primary" />
                    <div>
                      <div className="text-white font-medium">Latency Delta</div>
                      <div className="text-sm text-white/60">
                        {benchmark.baseline_time_ms > 0
                          ? `${((1 - benchmark.optimized_time_ms / benchmark.baseline_time_ms) * 100).toFixed(1)}% faster`
                          : 'N/A'}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <Activity className="w-5 h-5 text-accent-info" />
                    <div>
                      <div className="text-white font-medium">Optimization Goal</div>
                      <div className="text-sm text-white/60">{benchmark.optimization_goal || 'Speed'}</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="max-w-xl mx-auto text-center text-white/50">
            Select a benchmark to focus on from the table, or choose the top performer from quick actions.
          </div>
        )}
      </div>
    </div>
  );
}
