'use client';

import { useState } from 'react';
import { Activity, Cpu, Globe2, Loader2, RefreshCw } from 'lucide-react';
import { runSpeedTest, runGpuBandwidthTest, runNetworkTest } from '@/lib/api';

type Result = { label: string; value: string };

export function DiagnosticsCard() {
  const [results, setResults] = useState<Record<string, Result>>({});
  const [running, setRunning] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRun = async (key: string, fn: () => Promise<any>) => {
    if (running) return;
    setRunning(key);
    setError(null);
    try {
      const res: any = await fn();
      const summary =
        typeof res === 'string'
          ? res
          : res?.summary || res?.message || JSON.stringify(res);
      setResults((prev) => ({
        ...prev,
        [key]: {
          label:
            key === 'speed'
              ? 'Disk/NFS'
              : key === 'bandwidth'
              ? 'GPU BW'
              : 'Network',
          value: summary,
        },
      }));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to run test');
    } finally {
      setRunning(null);
    }
  };

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-accent-info" />
          <h3 className="font-medium text-white">Health & Speed Tests</h3>
        </div>
        <button
          onClick={() => {
            setResults({});
            setError(null);
          }}
          className="p-2 hover:bg-white/5 rounded-lg"
        >
          <RefreshCw className="w-4 h-4 text-white/50" />
        </button>
      </div>
      <div className="card-body space-y-3">
        {error && (
          <div className="text-sm text-accent-warning">⚠️ {error}</div>
        )}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          <button
            onClick={() => handleRun('speed', runSpeedTest)}
            className="flex items-center justify-between px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-left"
          >
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-accent-primary" />
              <span className="text-sm text-white">Disk Speed</span>
            </div>
            {running === 'speed' && <Loader2 className="w-4 h-4 animate-spin text-accent-primary" />}
          </button>
          <button
            onClick={() => handleRun('bandwidth', runGpuBandwidthTest)}
            className="flex items-center justify-between px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-left"
          >
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-accent-secondary" />
              <span className="text-sm text-white">GPU Bandwidth</span>
            </div>
            {running === 'bandwidth' && <Loader2 className="w-4 h-4 animate-spin text-accent-secondary" />}
          </button>
          <button
            onClick={() => handleRun('network', runNetworkTest)}
            className="flex items-center justify-between px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-left"
          >
            <div className="flex items-center gap-2">
              <Globe2 className="w-4 h-4 text-accent-info" />
              <span className="text-sm text-white">Network Test</span>
            </div>
            {running === 'network' && <Loader2 className="w-4 h-4 animate-spin text-accent-info" />}
          </button>
        </div>
        <div className="space-y-2">
          {Object.keys(results).length === 0 ? (
            <div className="text-sm text-white/50">
              Run any test to capture a quick health snapshot.
            </div>
          ) : (
            Object.entries(results).map(([key, res]) => (
              <div
                key={key}
                className="p-3 rounded-lg bg-white/5 border border-white/10 text-sm text-white/80"
              >
                <div className="text-white font-semibold">{res.label}</div>
                <div className="text-white/70 break-words whitespace-pre-wrap">
                  {res.value}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

