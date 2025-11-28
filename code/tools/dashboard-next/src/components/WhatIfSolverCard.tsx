'use client';

import { useEffect, useState } from 'react';
import { Sparkles, Loader2, AlertTriangle, RefreshCw, SlidersHorizontal } from 'lucide-react';
import { getAnalysisTradeoffs, getAnalysisRecommendations } from '@/lib/api';

export function WhatIfSolverCard() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tradeoffs, setTradeoffs] = useState<any[]>([]);
  const [recs, setRecs] = useState<any[]>([]);

  const load = async () => {
    try {
      setLoading(true);
      setError(null);
      const [t, r] = await Promise.all([
        getAnalysisTradeoffs().catch(() => null),
        getAnalysisRecommendations().catch(() => null),
      ]);
      const tradeoffList = (t as any)?.tradeoffs || (t as any)?.results || (Array.isArray(t) ? t : []);
      const recList = (r as any)?.recommendations || (r as any)?.results || (Array.isArray(r) ? r : []);
      setTradeoffs(tradeoffList || []);
      setRecs(recList || []);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load what-if data');
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
          <SlidersHorizontal className="w-5 h-5 text-accent-info" />
          <h3 className="font-medium text-white">What-If Constraint Solver</h3>
        </div>
        <button onClick={load} className="p-2 hover:bg-white/5 rounded-lg">
          <RefreshCw className="w-4 h-4 text-white/50" />
        </button>
      </div>
      <div className="card-body space-y-3">
        {loading ? (
          <div className="flex items-center gap-2 text-white/60">
            <Loader2 className="w-4 h-4 animate-spin" />
            Evaluating trade-offs...
          </div>
        ) : error ? (
          <div className="flex items-center gap-2 text-accent-warning">
            <AlertTriangle className="w-4 h-4" /> {error}
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {tradeoffs.slice(0, 4).map((t, i) => (
                <div key={i} className="p-3 rounded-lg bg-white/5 border border-white/10">
                  <div className="flex items-center justify-between">
                    <div className="text-white font-semibold">{t.name || `Scenario ${i + 1}`}</div>
                    {t.impact && <span className="text-accent-primary font-bold">{t.impact}</span>}
                  </div>
                  <div className="text-sm text-white/60 mt-1">
                    {t.description || 'Trade-off between speed, memory, and cost'}
                  </div>
                </div>
              ))}
            </div>
            {recs.length > 0 && (
              <div className="space-y-2">
                <div className="text-xs text-white/50 uppercase">Recommendations</div>
                {recs.slice(0, 3).map((r, i) => (
                  <div key={i} className="p-3 rounded-lg bg-accent-primary/10 border border-accent-primary/20 text-sm text-white">
                    <Sparkles className="w-4 h-4 inline mr-2 text-accent-primary" />
                    {r}
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
