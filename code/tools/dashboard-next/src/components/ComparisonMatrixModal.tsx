'use client';

import { useMemo, useState } from 'react';
import { Benchmark } from '@/types';
import { formatMs, getSpeedupColor } from '@/lib/utils';
import { Download, Search, X } from 'lucide-react';
import { useToast } from './Toast';

interface ComparisonMatrixModalProps {
  isOpen: boolean;
  onClose: () => void;
  benchmarks: Benchmark[];
}

export function ComparisonMatrixModal({ isOpen, onClose, benchmarks }: ComparisonMatrixModalProps) {
  const [search, setSearch] = useState('');
  const { showToast } = useToast();

  const filtered = useMemo(() => {
    const term = search.toLowerCase();
    return benchmarks
      .filter((b) => {
        if (!term) return true;
        return b.name.toLowerCase().includes(term) || b.chapter.toLowerCase().includes(term);
      })
      .sort((a, b) => (b.speedup || 0) - (a.speedup || 0))
      .slice(0, 30);
  }, [benchmarks, search]);

  if (!isOpen) return null;

  const exportCsv = () => {
    if (filtered.length === 0) return;
    const header = 'Chapter,Benchmark,Baseline (ms),Optimized (ms),Speedup,Status';
    const rows = filtered.map((b) =>
      [b.chapter, b.name, b.baseline_time_ms?.toFixed?.(3), b.optimized_time_ms?.toFixed?.(3), b.speedup?.toFixed?.(2), b.status].join(',')
    );
    const blob = new Blob([header + '\n' + rows.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'benchmark_comparison_matrix.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast('📥 Matrix exported', 'success');
  };

  return (
    <div
      className="fixed inset-0 z-[9998] bg-black/70 backdrop-blur-sm flex items-start justify-center pt-[8vh]"
      onClick={onClose}
    >
      <div
        className="w-[1100px] max-w-[96vw] bg-brand-card border border-white/10 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[84vh]"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/5">
          <div>
            <div className="text-xs uppercase tracking-wide text-white/40">Benchmark Comparison Matrix</div>
            <div className="text-lg font-semibold text-white">{filtered.length} rows (showing top 30)</div>
          </div>
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/40" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Filter by name or chapter..."
                className="pl-9 pr-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white placeholder:text-white/40 focus:outline-none w-64"
              />
            </div>
            <button
              onClick={exportCsv}
              className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 rounded-lg text-sm text-white"
            >
              <Download className="w-4 h-4" />
              Export CSV
            </button>
            <button
              onClick={onClose}
              className="px-3 py-1.5 bg-white/5 hover:bg-white/10 rounded-lg text-sm text-white/70"
            >
              Close
            </button>
          </div>
        </div>

        <div className="overflow-auto flex-1">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/5 bg-white/[0.02]">
                <th className="px-4 py-3 text-left text-white/50 uppercase text-xs">Benchmark</th>
                <th className="px-4 py-3 text-right text-white/50 uppercase text-xs">Baseline</th>
                <th className="px-4 py-3 text-right text-white/50 uppercase text-xs">Optimized</th>
                <th className="px-4 py-3 text-right text-white/50 uppercase text-xs">Speedup</th>
                <th className="px-4 py-3 text-right text-white/50 uppercase text-xs">Delta %</th>
                <th className="px-4 py-3 text-center text-white/50 uppercase text-xs">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-4 py-6 text-center text-white/40">
                    No benchmarks match your filter.
                  </td>
                </tr>
              )}
              {filtered.map((b) => {
                const deltaPct =
                  b.baseline_time_ms && b.optimized_time_ms
                    ? ((1 - b.optimized_time_ms / b.baseline_time_ms) * 100)
                    : 0;
                return (
                  <tr key={`${b.chapter}-${b.name}`} className="hover:bg-white/[0.02]">
                    <td className="px-4 py-3">
                      <div className="font-medium text-white">{b.name}</div>
                      <div className="text-xs text-white/40">{b.chapter}</div>
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-accent-tertiary">{formatMs(b.baseline_time_ms)}</td>
                    <td className="px-4 py-3 text-right font-mono text-accent-success">{formatMs(b.optimized_time_ms)}</td>
                    <td className="px-4 py-3 text-right font-bold" style={{ color: getSpeedupColor(b.speedup) }}>
                      {b.speedup?.toFixed?.(2)}x
                    </td>
                    <td className="px-4 py-3 text-right font-bold" style={{ color: deltaPct >= 0 ? '#00f5a0' : '#ff4757' }}>
                      {deltaPct >= 0 ? '-' : '+'}
                      {Math.abs(deltaPct).toFixed(1)}%
                    </td>
                    <td className="px-4 py-3 text-center">
                      {b.status === 'succeeded' ? (
                        <span className="text-accent-success">✓</span>
                      ) : b.status === 'failed' ? (
                        <span className="text-accent-danger">✕</span>
                      ) : (
                        <span className="text-accent-warning">•</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
