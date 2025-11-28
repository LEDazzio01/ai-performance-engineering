'use client';

import { useMemo, useState } from 'react';
import { CheckCircle2, AlertTriangle, RefreshCw } from 'lucide-react';
import { Benchmark } from '@/types';

interface Props {
  benchmarks: Benchmark[];
  gpuName?: string;
}

export function PreflightChecklist({ benchmarks, gpuName }: Props) {
  const [timestamp, setTimestamp] = useState<Date | null>(null);

  const checks = useMemo(() => {
    const total = benchmarks.length;
    const succeeded = benchmarks.filter((b) => b.status === 'succeeded');
    const avgSpeedup =
      succeeded.length > 0
        ? succeeded.reduce((s, b) => s + (b.speedup || 0), 0) / succeeded.length
        : 0;

    return [
      {
        label: 'Benchmarks loaded',
        pass: total > 0,
        detail: `${total} benchmarks detected`,
      },
      {
        label: 'No regressions',
        pass: benchmarks.filter((b) => (b.speedup || 1) < 1).length === 0,
        detail: `${benchmarks.filter((b) => (b.speedup || 1) < 1).length} regressions`,
      },
      {
        label: 'Average speedup ≥ 1.2x',
        pass: avgSpeedup >= 1.2,
        detail: `${avgSpeedup.toFixed(2)}x`,
      },
      {
        label: 'GPU detected',
        pass: !!gpuName,
        detail: gpuName || 'Unknown GPU',
      },
    ];
  }, [benchmarks, gpuName]);

  const score = Math.round((checks.filter((c) => c.pass).length / checks.length) * 100);

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <RefreshCw className="w-5 h-5 text-accent-primary" />
          <h3 className="font-medium text-white">Pre-flight Checklist</h3>
        </div>
        <div className="text-sm text-white/60">
          Score: <span className="font-semibold text-accent-primary">{score}%</span>
        </div>
      </div>
      <div className="card-body space-y-2">
        {checks.map((c, i) => (
          <div
            key={i}
            className={`flex items-center justify-between p-3 rounded-lg border ${
              c.pass ? 'border-accent-success/20 bg-accent-success/5' : 'border-accent-warning/20 bg-accent-warning/5'
            }`}
          >
            <div>
              <div className="text-sm text-white font-medium">{c.label}</div>
              <div className="text-xs text-white/60">{c.detail}</div>
            </div>
            {c.pass ? (
              <CheckCircle2 className="w-5 h-5 text-accent-success" />
            ) : (
              <AlertTriangle className="w-5 h-5 text-accent-warning" />
            )}
          </div>
        ))}
        <div className="text-xs text-white/40">
          Last checked: {timestamp ? timestamp.toLocaleTimeString() : 'Tap refresh to run checks'}
        </div>
        <button
          onClick={() => setTimestamp(new Date())}
          className="px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-sm text-white hover:bg-white/10"
        >
          Re-run checks
        </button>
      </div>
    </div>
  );
}

