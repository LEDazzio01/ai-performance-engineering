'use client';

import { useMemo } from 'react';
import { BarChart3, AlertTriangle } from 'lucide-react';
import { Benchmark } from '@/types';

interface Props {
  benchmarks: Benchmark[];
}

export function VarianceAnalysis({ benchmarks }: Props) {
  const rows = useMemo(() => {
    const groups: Record<string, Benchmark[]> = {};
    benchmarks.forEach((b) => {
      groups[b.chapter] = groups[b.chapter] || [];
      groups[b.chapter].push(b);
    });
    return Object.entries(groups).map(([chapter, list]) => {
      const times = list
        .filter((b) => typeof b.optimized_time_ms === 'number')
        .map((b) => b.optimized_time_ms as number);
      const mean = times.length ? times.reduce((a, b) => a + b, 0) / times.length : 0;
      const variance =
        times.length > 1
          ? times.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (times.length - 1)
          : 0;
      const stdev = Math.sqrt(variance);
      const cv = mean > 0 ? (stdev / mean) * 100 : 0;
      return { chapter, cv, mean };
    });
  }, [benchmarks]);

  const flagged = rows.filter((r) => r.cv > 10);

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-accent-warning" />
          <h3 className="font-medium text-white">Variance Analysis</h3>
        </div>
        {flagged.length > 0 && (
          <div className="flex items-center gap-1 text-xs text-accent-warning">
            <AlertTriangle className="w-4 h-4" /> {flagged.length} unstable chapters
          </div>
        )}
      </div>
      <div className="card-body">
        {rows.length === 0 ? (
          <div className="text-sm text-white/50">No timing data available.</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {rows.map((row, i) => (
              <div
                key={i}
                className={`p-3 rounded-lg border ${
                  row.cv > 10 ? 'border-accent-warning/30 bg-accent-warning/5' : 'border-white/10 bg-white/5'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-white font-medium">{row.chapter}</span>
                  <span className={`text-sm font-mono ${row.cv > 10 ? 'text-accent-warning' : 'text-accent-success'}`}>
                    CV {row.cv.toFixed(1)}%
                  </span>
                </div>
                <div className="text-xs text-white/50 mt-1">Mean: {row.mean.toFixed(3)} ms</div>
                {row.cv > 10 && (
                  <div className="text-xs text-accent-warning mt-1">
                    High variance detected — rerun benchmarks or pin clocks.
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

