'use client';

import { useState, useMemo } from 'react';
import { Benchmark } from '@/types';
import { formatMs, getSpeedupColor, cn } from '@/lib/utils';
import { GitCompare, Search, ArrowRight, Check, X } from 'lucide-react';

interface CompareTabProps {
  benchmarks: Benchmark[];
  pinnedBenchmarks?: Set<string>;
}

export function CompareTab({ benchmarks, pinnedBenchmarks }: CompareTabProps) {
  const [leftBenchmark, setLeftBenchmark] = useState<Benchmark | null>(null);
  const [rightBenchmark, setRightBenchmark] = useState<Benchmark | null>(null);
  const [searchLeft, setSearchLeft] = useState('');
  const [searchRight, setSearchRight] = useState('');
  const [mode, setMode] = useState<'single' | 'pinned'>('single');

  const succeededBenchmarks = benchmarks.filter((b) => b.status === 'succeeded');

  const filteredLeft = useMemo(() => {
    return succeededBenchmarks.filter(
      (b) =>
        b.name.toLowerCase().includes(searchLeft.toLowerCase()) ||
        b.chapter.toLowerCase().includes(searchLeft.toLowerCase())
    );
  }, [succeededBenchmarks, searchLeft]);

  const filteredRight = useMemo(() => {
    return succeededBenchmarks.filter(
      (b) =>
        b.name.toLowerCase().includes(searchRight.toLowerCase()) ||
        b.chapter.toLowerCase().includes(searchRight.toLowerCase())
    );
  }, [succeededBenchmarks, searchRight]);

  const comparison = useMemo(() => {
    if (!leftBenchmark || !rightBenchmark) return null;
    const speedupDiff = rightBenchmark.speedup - leftBenchmark.speedup;
    const baselineDiff = rightBenchmark.baseline_time_ms - leftBenchmark.baseline_time_ms;
    const optimizedDiff = rightBenchmark.optimized_time_ms - leftBenchmark.optimized_time_ms;
    return { speedupDiff, baselineDiff, optimizedDiff };
  }, [leftBenchmark, rightBenchmark]);

  const pinnedList = useMemo(() => {
    if (!pinnedBenchmarks) return [];
    const keys = Array.from(pinnedBenchmarks);
    return keys
      .map((key) => {
        const [chapter, name] = key.split(':');
        return benchmarks.find((b) => b.chapter === chapter && b.name === name);
      })
      .filter(Boolean) as Benchmark[];
  }, [pinnedBenchmarks, benchmarks]);

  return (
    <div className="space-y-6">
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <GitCompare className="w-5 h-5 text-accent-primary" />
            <h2 className="text-lg font-semibold text-white">Benchmark Comparison</h2>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setMode('single')}
              className={cn(
                'px-3 py-1.5 rounded-lg text-sm',
                mode === 'single' ? 'bg-accent-primary/20 text-accent-primary' : 'text-white/50 hover:text-white'
              )}
            >
              Single Compare
            </button>
            <button
              onClick={() => setMode('pinned')}
              className={cn(
                'px-3 py-1.5 rounded-lg text-sm',
                mode === 'pinned' ? 'bg-accent-secondary/20 text-accent-secondary' : 'text-white/50 hover:text-white'
              )}
            >
              Pinned Grid
            </button>
          </div>
        </div>
      </div>

      {mode === 'pinned' && (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">Pinned Benchmarks ({pinnedList.length})</h3>
          </div>
          <div className="card-body">
            {pinnedList.length === 0 ? (
              <div className="text-sm text-white/50">Pin benchmarks from Overview to see them here.</div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {pinnedList.map((b, i) => (
                  <div key={`${b.chapter}-${b.name}-${i}`} className="p-4 rounded-lg bg-white/5 border border-white/10 space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="text-white font-semibold text-sm">{b.name}</div>
                      <span className="text-xs text-white/50">{b.chapter}</span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="text-center p-2 bg-white/5 rounded-lg">
                        <div className="text-white/50 text-xs">Baseline</div>
                        <div className="text-accent-tertiary font-mono">{formatMs(b.baseline_time_ms)}</div>
                      </div>
                      <div className="text-center p-2 bg-white/5 rounded-lg">
                        <div className="text-white/50 text-xs">Optimized</div>
                        <div className="text-accent-success font-mono">{formatMs(b.optimized_time_ms)}</div>
                      </div>
                    </div>
                    <div className="text-center">
                      <span className="text-lg font-bold" style={{ color: getSpeedupColor(b.speedup) }}>
                        {b.speedup.toFixed(2)}x
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Selection panels */}
      {mode === 'single' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left benchmark selector */}
          <div className="card">
            <div className="card-header">
              <h3 className="font-medium text-white">Benchmark A</h3>
              {leftBenchmark && (
                <button
                  onClick={() => setLeftBenchmark(null)}
                  className="text-white/40 hover:text-white"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>
            <div className="card-body">
              {leftBenchmark ? (
                <div className="p-4 bg-accent-primary/10 border border-accent-primary/30 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-white">{leftBenchmark.name}</span>
                    <span
                      className="text-lg font-bold"
                      style={{ color: getSpeedupColor(leftBenchmark.speedup) }}
                    >
                      {leftBenchmark.speedup.toFixed(2)}x
                    </span>
                  </div>
                  <div className="text-sm text-white/60">{leftBenchmark.chapter}</div>
                  <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <span className="text-white/40">Baseline:</span>{' '}
                      <span className="text-accent-tertiary font-mono">
                        {formatMs(leftBenchmark.baseline_time_ms)}
                      </span>
                    </div>
                    <div>
                      <span className="text-white/40">Optimized:</span>{' '}
                      <span className="text-accent-success font-mono">
                        {formatMs(leftBenchmark.optimized_time_ms)}
                      </span>
                    </div>
                  </div>
                </div>
              ) : (
                <>
                  <div className="relative mb-3">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/40" />
                    <input
                      type="text"
                      placeholder="Search benchmarks..."
                      value={searchLeft}
                      onChange={(e) => setSearchLeft(e.target.value)}
                      className="w-full pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white placeholder:text-white/40 focus:outline-none focus:border-accent-primary/50"
                    />
                  </div>
                  <div className="max-h-[300px] overflow-y-auto space-y-1">
                    {filteredLeft.slice(0, 20).map((b, i) => (
                      <button
                        key={`${b.chapter}-${b.name}-${i}`}
                        onClick={() => setLeftBenchmark(b)}
                        className="w-full flex items-center justify-between p-3 hover:bg-white/5 rounded-lg transition-colors text-left"
                      >
                        <div>
                          <div className="font-medium text-white text-sm">{b.name}</div>
                          <div className="text-xs text-white/40">{b.chapter}</div>
                        </div>
                        <span
                          className="font-bold"
                          style={{ color: getSpeedupColor(b.speedup) }}
                        >
                          {b.speedup.toFixed(2)}x
                        </span>
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Right benchmark selector */}
          <div className="card">
            <div className="card-header">
              <h3 className="font-medium text-white">Benchmark B</h3>
              {rightBenchmark && (
                <button
                  onClick={() => setRightBenchmark(null)}
                  className="text-white/40 hover:text-white"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>
            <div className="card-body">
              {rightBenchmark ? (
                <div className="p-4 bg-accent-secondary/10 border border-accent-secondary/30 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-white">{rightBenchmark.name}</span>
                    <span
                      className="text-lg font-bold"
                      style={{ color: getSpeedupColor(rightBenchmark.speedup) }}
                    >
                      {rightBenchmark.speedup.toFixed(2)}x
                    </span>
                  </div>
                  <div className="text-sm text-white/60">{rightBenchmark.chapter}</div>
                  <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <span className="text-white/40">Baseline:</span>{' '}
                      <span className="text-accent-tertiary font-mono">
                        {formatMs(rightBenchmark.baseline_time_ms)}
                      </span>
                    </div>
                    <div>
                      <span className="text-white/40">Optimized:</span>{' '}
                      <span className="text-accent-success font-mono">
                        {formatMs(rightBenchmark.optimized_time_ms)}
                      </span>
                    </div>
                  </div>
                </div>
              ) : (
                <>
                  <div className="relative mb-3">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/40" />
                    <input
                      type="text"
                      placeholder="Search benchmarks..."
                      value={searchRight}
                      onChange={(e) => setSearchRight(e.target.value)}
                      className="w-full pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white placeholder:text-white/40 focus:outline-none focus:border-accent-primary/50"
                    />
                  </div>
                  <div className="max-h-[300px] overflow-y-auto space-y-1">
                    {filteredRight.slice(0, 20).map((b, i) => (
                      <button
                        key={`${b.chapter}-${b.name}-${i}`}
                        onClick={() => setRightBenchmark(b)}
                        className="w-full flex items-center justify-between p-3 hover:bg-white/5 rounded-lg transition-colors text-left"
                      >
                        <div>
                          <div className="font-medium text-white text-sm">{b.name}</div>
                          <div className="text-xs text-white/40">{b.chapter}</div>
                        </div>
                        <span
                          className="font-bold"
                          style={{ color: getSpeedupColor(b.speedup) }}
                        >
                          {b.speedup.toFixed(2)}x
                        </span>
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Comparison results */}
      {mode === 'single' && leftBenchmark && rightBenchmark && comparison && (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">Comparison Results</h3>
          </div>
          <div className="card-body">
            <div className="flex items-center justify-center gap-8 py-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-accent-primary">
                  {leftBenchmark.speedup.toFixed(2)}x
                </div>
                <div className="text-sm text-white/50">{leftBenchmark.name}</div>
              </div>

              <div className="flex flex-col items-center gap-2">
                <ArrowRight className="w-8 h-8 text-white/20" />
                <div
                  className={cn(
                    'px-4 py-2 rounded-full font-bold',
                    comparison.speedupDiff > 0
                      ? 'bg-accent-success/20 text-accent-success'
                      : comparison.speedupDiff < 0
                      ? 'bg-accent-danger/20 text-accent-danger'
                      : 'bg-white/10 text-white/60'
                  )}
                >
                  {comparison.speedupDiff > 0 ? '+' : ''}
                  {comparison.speedupDiff.toFixed(2)}x
                </div>
              </div>

              <div className="text-center">
                <div className="text-2xl font-bold text-accent-secondary">
                  {rightBenchmark.speedup.toFixed(2)}x
                </div>
                <div className="text-sm text-white/50">{rightBenchmark.name}</div>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4 pt-4 border-t border-white/5">
              <div className="text-center p-4 bg-white/5 rounded-lg">
                <div className="text-sm text-white/50 mb-1">Speedup Difference</div>
                <div
                  className={cn(
                    'text-xl font-bold',
                    comparison.speedupDiff >= 0 ? 'text-accent-success' : 'text-accent-danger'
                  )}
                >
                  {comparison.speedupDiff >= 0 ? '+' : ''}
                  {comparison.speedupDiff.toFixed(2)}x
                </div>
              </div>
              <div className="text-center p-4 bg-white/5 rounded-lg">
                <div className="text-sm text-white/50 mb-1">Baseline Δ</div>
                <div className="text-xl font-bold text-white font-mono">
                  {formatMs(Math.abs(comparison.baselineDiff))}
                </div>
              </div>
              <div className="text-center p-4 bg-white/5 rounded-lg">
                <div className="text-sm text-white/50 mb-1">Optimized Δ</div>
                <div className="text-xl font-bold text-white font-mono">
                  {formatMs(Math.abs(comparison.optimizedDiff))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

