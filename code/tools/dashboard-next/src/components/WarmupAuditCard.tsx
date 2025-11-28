'use client';

import { useEffect, useState } from 'react';
import { getProfilerTimeline } from '@/lib/api';
import { Flame, Loader2, AlertTriangle, RefreshCw } from 'lucide-react';
import { formatMs } from '@/lib/utils';

type TimelineEvent = { timestamp: number; label?: string; duration_ms?: number };

export function WarmupAuditCard() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [events, setEvents] = useState<TimelineEvent[]>([]);

  const load = async () => {
    try {
      setLoading(true);
      setError(null);
      const data: any = await getProfilerTimeline();
      const list = (data?.timeline || data || []) as TimelineEvent[];
      setEvents(list);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load warmup timeline');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const warmupWindow = events.slice(0, 10);
  const jitter =
    warmupWindow.length > 1
      ? Math.max(...warmupWindow.map((e) => e.duration_ms || 0)) -
        Math.min(...warmupWindow.map((e) => e.duration_ms || 0))
      : 0;

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <Flame className="w-5 h-5 text-accent-warning" />
          <h3 className="font-medium text-white">Warmup Audit (JIT/Compile)</h3>
        </div>
        <button onClick={load} className="p-2 hover:bg-white/5 rounded-lg">
          <RefreshCw className="w-4 h-4 text-white/50" />
        </button>
      </div>
      <div className="card-body space-y-3">
        {loading ? (
          <div className="flex items-center gap-2 text-white/60">
            <Loader2 className="w-4 h-4 animate-spin" />
            Inspecting timeline...
          </div>
        ) : error ? (
          <div className="flex items-center gap-2 text-accent-warning">
            <AlertTriangle className="w-4 h-4" /> {error}
          </div>
        ) : (
          <>
            <div className="flex items-center justify-between text-sm text-white/80">
              <span>Warmup events analyzed: {warmupWindow.length}</span>
              <span className="text-accent-primary font-semibold">Jitter: {formatMs(jitter)}</span>
            </div>
            <div className="space-y-2">
              {warmupWindow.length === 0 ? (
                <div className="text-sm text-white/50">No timeline events returned.</div>
              ) : (
                warmupWindow.map((e, i) => (
                  <div key={i} className="p-3 rounded-lg bg-white/5 border border-white/10 flex items-center justify-between">
                    <div>
                      <div className="text-white font-semibold">{e.label || `Event ${i + 1}`}</div>
                      <div className="text-xs text-white/50">T+{e.timestamp} ms</div>
                    </div>
                    <div className="text-accent-success font-bold">
                      {e.duration_ms !== undefined ? formatMs(e.duration_ms) : '—'}
                    </div>
                  </div>
                ))
              )}
            </div>
            <p className="text-xs text-white/40">
              Tip: Large jitter means JIT warmup is uneven. Pre-warm critical paths or pin compile caches to reduce cold-start lag.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
