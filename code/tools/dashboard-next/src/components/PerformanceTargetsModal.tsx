'use client';

import { useEffect, useState } from 'react';
import { CheckCircle, AlertTriangle, Gauge, Target } from 'lucide-react';
import { useToast } from './Toast';

export type PerformanceTargets = {
  minAvgSpeedup: number;
  maxRegressions: number;
  passRate: number;
  maxMemoryUtil: number;
};

interface PerformanceTargetsModalProps {
  isOpen: boolean;
  onClose: () => void;
  targets: PerformanceTargets;
  onSave: (targets: PerformanceTargets) => void;
  currentSummary?: {
    avgSpeedup?: number;
    regressions?: number;
    passRate?: number;
    memoryUtil?: number;
  };
}

const STORAGE_KEY = 'performance_targets';

export function PerformanceTargetsModal({
  isOpen,
  onClose,
  targets,
  onSave,
  currentSummary,
}: PerformanceTargetsModalProps) {
  const [localTargets, setLocalTargets] = useState<PerformanceTargets>(targets);
  const { showToast } = useToast();

  useEffect(() => {
    if (isOpen) {
      // Load saved values if present
      try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {
          setLocalTargets(JSON.parse(saved));
          return;
        }
      } catch {
        // ignore parsing errors
      }
      setLocalTargets(targets);
    }
  }, [isOpen, targets]);

  if (!isOpen) return null;

  const handleSave = () => {
    onSave(localTargets);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(localTargets));
    } catch {
      // ignore write failures (e.g., storage disabled)
    }
    showToast('🎯 Performance targets updated', 'success');
    onClose();
  };

  const budgetChecks = [
    {
      label: 'Average speedup',
      target: `${localTargets.minAvgSpeedup}x`,
      current: currentSummary?.avgSpeedup ? `${currentSummary.avgSpeedup.toFixed(2)}x` : '—',
      ok: currentSummary?.avgSpeedup !== undefined && currentSummary.avgSpeedup >= localTargets.minAvgSpeedup,
    },
    {
      label: 'Regressions allowed',
      target: `${localTargets.maxRegressions}`,
      current: currentSummary?.regressions ?? '—',
      ok: currentSummary?.regressions !== undefined && currentSummary.regressions <= localTargets.maxRegressions,
    },
    {
      label: 'Pass rate',
      target: `${localTargets.passRate}%`,
      current: currentSummary?.passRate !== undefined ? `${currentSummary.passRate.toFixed(0)}%` : '—',
      ok: currentSummary?.passRate !== undefined && currentSummary.passRate >= localTargets.passRate,
    },
    {
      label: 'Max memory util',
      target: `${localTargets.maxMemoryUtil}%`,
      current: currentSummary?.memoryUtil !== undefined ? `${currentSummary.memoryUtil.toFixed(0)}%` : '—',
      ok: currentSummary?.memoryUtil !== undefined && currentSummary.memoryUtil <= localTargets.maxMemoryUtil,
    },
  ];

  return (
    <div
      className="fixed inset-0 z-[9998] bg-black/70 backdrop-blur-sm flex items-start justify-center pt-[10vh]"
      onClick={onClose}
    >
      <div
        className="w-[720px] max-w-[94vw] bg-brand-card border border-white/10 rounded-2xl shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/5">
          <div className="flex items-center gap-2">
            <Target className="w-5 h-5 text-accent-primary" />
            <div>
              <div className="text-xs uppercase tracking-wide text-white/40">Performance Budget</div>
              <div className="text-lg font-semibold text-white">Set guardrails for every run</div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="px-3 py-1.5 bg-white/5 hover:bg-white/10 rounded-lg text-sm text-white/70"
          >
            Esc
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-6">
          <div className="space-y-4">
            <label className="block">
              <span className="block text-sm text-white/50 mb-1">Min average speedup (x)</span>
              <input
                type="number"
                step="0.1"
                min="1"
                value={localTargets.minAvgSpeedup}
                onChange={(e) => setLocalTargets({ ...localTargets, minAvgSpeedup: Number(e.target.value) })}
                className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none"
              />
            </label>

            <label className="block">
              <span className="block text-sm text-white/50 mb-1">Max regressions allowed</span>
              <input
                type="number"
                min="0"
                value={localTargets.maxRegressions}
                onChange={(e) => setLocalTargets({ ...localTargets, maxRegressions: Number(e.target.value) })}
                className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none"
              />
            </label>

            <label className="block">
              <span className="block text-sm text-white/50 mb-1">Target pass rate (%)</span>
              <input
                type="number"
                min="0"
                max="100"
                value={localTargets.passRate}
                onChange={(e) => setLocalTargets({ ...localTargets, passRate: Number(e.target.value) })}
                className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none"
              />
            </label>

            <label className="block">
              <span className="block text-sm text-white/50 mb-1">Max GPU memory usage (%)</span>
              <input
                type="number"
                min="0"
                max="100"
                value={localTargets.maxMemoryUtil}
                onChange={(e) => setLocalTargets({ ...localTargets, maxMemoryUtil: Number(e.target.value) })}
                className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none"
              />
            </label>

            <button
              onClick={handleSave}
              className="w-full mt-2 px-4 py-2 bg-gradient-to-r from-accent-primary to-accent-secondary text-black rounded-lg font-medium hover:opacity-90"
            >
              Save Targets
            </button>
          </div>

          <div className="p-4 rounded-xl bg-white/5 border border-white/10 space-y-3">
            <div className="flex items-center gap-2 text-white font-medium">
              <Gauge className="w-4 h-4 text-accent-info" />
              Current status vs targets
            </div>
            <div className="space-y-2">
              {budgetChecks.map((item) => (
                <div
                  key={item.label}
                  className="flex items-center justify-between p-3 rounded-lg bg-white/[0.03] border border-white/5"
                >
                  <div>
                    <div className="text-sm text-white/50">{item.label}</div>
                    <div className="text-white">{item.current} <span className="text-white/40">/ target {item.target}</span></div>
                  </div>
                  {item.ok ? (
                    <CheckCircle className="w-5 h-5 text-accent-success" />
                  ) : (
                    <AlertTriangle className="w-5 h-5 text-accent-warning" />
                  )}
                </div>
              ))}
            </div>
            <p className="text-xs text-white/40">
              These guardrails mirror the original dashboard: keep regressions at zero, maintain your minimum speedup, and ensure
              memory headroom.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
