'use client';

import { useEffect, useMemo, useState } from 'react';
import { Play, Loader2, XCircle, CheckCircle, Gauge, AlertTriangle } from 'lucide-react';
import { Benchmark } from '@/types';
import { runBenchmark } from '@/lib/api';
import { useToast } from './Toast';

interface RunBenchmarkModalProps {
  isOpen: boolean;
  onClose: () => void;
  benchmarks: Benchmark[];
  onRunComplete?: () => void;
}

export function RunBenchmarkModal({ isOpen, onClose, benchmarks, onRunComplete }: RunBenchmarkModalProps) {
  const [selected, setSelected] = useState<string>('');
  const [runBaseline, setRunBaseline] = useState(true);
  const [runOptimized, setRunOptimized] = useState(true);
  const [output, setOutput] = useState<string>('');
  const [running, setRunning] = useState(false);
  const { showToast } = useToast();

  const succeeded = useMemo(
    () => benchmarks.filter((b) => b.status === 'succeeded'),
    [benchmarks]
  );

  useEffect(() => {
    if (isOpen) {
      setOutput('');
      if (succeeded.length > 0) {
        setSelected(`${succeeded[0].chapter}:${succeeded[0].name}`);
      } else if (benchmarks.length > 0) {
        setSelected(`${benchmarks[0].chapter}:${benchmarks[0].name}`);
      } else {
        setSelected('');
      }
    }
  }, [isOpen, succeeded, benchmarks]);

  if (!isOpen) return null;

  const selectedBenchmark = (selected ? benchmarks.find((b) => `${b.chapter}:${b.name}` === selected) : null);

  const handleRun = async () => {
    if (!selectedBenchmark) return;
    setRunning(true);
    setOutput(`▶️ Starting benchmark: ${selectedBenchmark.chapter}:${selectedBenchmark.name}\n`);
    try {
      const result = await runBenchmark(selectedBenchmark.chapter, selectedBenchmark.name, {
        run_baseline: runBaseline,
        run_optimized: runOptimized,
      });

      if ((result as any).success) {
        const baseline = (result as any).baseline_ms;
        const optimized = (result as any).optimized_ms;
        const speedup = (result as any).speedup;
        setOutput((prev) =>
          `${prev}\n✅ Completed successfully!\n\nBaseline: ${baseline?.toFixed?.(2) ?? 'N/A'} ms\nOptimized: ${optimized?.toFixed?.(2) ?? 'N/A'} ms\nSpeedup: ${speedup?.toFixed?.(2) ?? 'N/A'}x\n`
        );
        showToast('🏃 Benchmark finished', 'success');
        onRunComplete?.();
      } else {
        const err = (result as any).error || 'Benchmark failed';
        setOutput((prev) => `${prev}\n❌ ${err}\n`);
        showToast(err, 'error');
      }
    } catch (e) {
      setOutput((prev) =>
        `${prev}\n⚠️ Backend unavailable. You can run manually:\npython -m benchmarks.run ${selectedBenchmark.chapter} --filter "${selectedBenchmark.name}"\n`
      );
      showToast('Backend unavailable for benchmark run', 'error');
    } finally {
      setRunning(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-[9998] bg-black/70 backdrop-blur-sm flex items-start justify-center pt-[10vh]"
      onClick={onClose}
    >
      <div
        className="w-[700px] max-w-[94vw] bg-brand-card border border-white/10 rounded-2xl shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/5">
          <div className="flex items-center gap-2">
            <Gauge className="w-5 h-5 text-accent-primary" />
            <div>
              <div className="text-xs uppercase tracking-wide text-white/40">Quick Benchmark Run</div>
              <div className="text-lg font-semibold text-white">Run a single target from the UI</div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="px-3 py-1.5 bg-white/5 hover:bg-white/10 rounded-lg text-sm text-white/70"
          >
            Esc
          </button>
        </div>

        <div className="p-6 space-y-4">
          {benchmarks.length === 0 ? (
            <div className="flex items-center gap-3 p-4 bg-accent-warning/10 border border-accent-warning/20 rounded-lg text-sm text-white/70">
              <AlertTriangle className="w-4 h-4 text-accent-warning" />
              No benchmarks loaded yet. Connect the backend and refresh.
            </div>
          ) : (
            <>
              <div className="space-y-2">
                <label className="block text-sm text-white/60">Select benchmark</label>
                <select
                  value={selected}
                  onChange={(e) => setSelected(e.target.value)}
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none"
                >
                  {benchmarks.map((b) => (
                    <option key={`${b.chapter}:${b.name}`} value={`${b.chapter}:${b.name}`}>
                      {b.chapter}: {b.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <label className="flex items-center gap-2 text-white/70">
                  <input
                    type="checkbox"
                    checked={runBaseline}
                    onChange={(e) => setRunBaseline(e.target.checked)}
                    className="w-4 h-4 accent-accent-primary"
                  />
                  Run baseline
                </label>
                <label className="flex items-center gap-2 text-white/70">
                  <input
                    type="checkbox"
                    checked={runOptimized}
                    onChange={(e) => setRunOptimized(e.target.checked)}
                    className="w-4 h-4 accent-accent-primary"
                  />
                  Run optimized
                </label>
              </div>

              <div className="flex items-center gap-3">
                <button
                  onClick={handleRun}
                  disabled={!selectedBenchmark || running}
                  className="flex items-center gap-2 px-5 py-2 bg-gradient-to-r from-accent-primary to-accent-secondary text-black rounded-lg font-medium disabled:opacity-50"
                >
                  {running ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                  {running ? 'Running...' : 'Run Benchmark'}
                </button>
                <div className="text-xs text-white/50">
                  This mirrors the quick runner from the original HTML dashboard.
                </div>
              </div>

              {output && (
                <div className="rounded-lg bg-black/50 border border-white/10 p-3 font-mono text-sm text-white/80 max-h-60 overflow-auto">
                  {output.split('\n').map((line, idx) => (
                    <div key={idx} className="whitespace-pre-wrap">
                      {line}
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>

        <div className="px-6 py-4 border-t border-white/5 flex items-center justify-between text-xs text-white/40">
          <div className="flex items-center gap-2">
            <CheckCircle className="w-4 h-4 text-accent-success" />
            Runs `python -m benchmarks.run` under the hood.
          </div>
          <div className="flex items-center gap-2">
            <XCircle className="w-4 h-4 text-accent-danger" />
            Stop with Ctrl+C in the backend terminal.
          </div>
        </div>
      </div>
    </div>
  );
}
