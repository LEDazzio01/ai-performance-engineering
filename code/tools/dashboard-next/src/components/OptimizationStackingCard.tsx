'use client';

import { useEffect, useState } from 'react';
import { Layers, Loader2, AlertTriangle, RefreshCw } from 'lucide-react';
import { getAnalysisStacking } from '@/lib/api';

export function OptimizationStackingCard() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<any>(null);

  const load = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await getAnalysisStacking();
      setData(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load stacking data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <Layers className="w-5 h-5 text-accent-primary" />
          <h3 className="font-medium text-white">Optimization Stacking</h3>
        </div>
        <button onClick={load} className="p-2 hover:bg-white/5 rounded-lg">
          <RefreshCw className="w-4 h-4 text-white/50" />
        </button>
      </div>
      <div className="card-body">
        {loading ? (
          <div className="flex items-center gap-2 text-white/60">
            <Loader2 className="w-4 h-4 animate-spin" /> Loading stacking combos...
          </div>
        ) : error ? (
          <div className="flex items-center gap-2 text-accent-warning">
            <AlertTriangle className="w-4 h-4" /> {error}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {(data?.strategies || data?.stacking || []).slice(0, 6).map((item: any, idx: number) => (
              <div key={idx} className="p-4 rounded-lg bg-white/5 border border-white/10 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-white font-semibold">{item.name || item.strategy || `Combo ${idx + 1}`}</span>
                  {item.speedup && (
                    <span className="text-accent-success font-bold">{item.speedup.toFixed(2)}x</span>
                  )}
                </div>
                <div className="text-sm text-white/60">
                  {item.description || item.details || 'Stacked optimizations for speed/memory trade-offs.'}
                </div>
                {item.components && (
                  <div className="flex flex-wrap gap-1">
                    {item.components.map((c: string, i: number) => (
                      <span key={i} className="px-2 py-0.5 bg-accent-primary/10 text-accent-primary rounded-full text-xs">
                        {c}
                      </span>
                    ))}
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
