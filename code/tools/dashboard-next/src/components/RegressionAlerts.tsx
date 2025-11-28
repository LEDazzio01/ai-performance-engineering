'use client';

import { useEffect, useState } from 'react';
import { AlertTriangle, CheckCircle, Activity, Loader2, RefreshCw } from 'lucide-react';
import { getHistoryTrends } from '@/lib/api';

export function RegressionAlerts() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [trends, setTrends] = useState<any>(null);

  const load = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getHistoryTrends();
      setTrends(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load trends');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  if (loading) {
    return (
      <div className="p-4 rounded-xl border border-white/10 bg-white/5 flex items-center gap-2 text-sm text-white/60">
        <Loader2 className="w-4 h-4 animate-spin" />
        Checking regressions...
      </div>
    );
  }

  if (error || !trends) {
    return (
      <div className="p-4 rounded-xl border border-accent-warning/20 bg-accent-warning/10 flex items-center justify-between text-sm text-white/80">
        <div className="flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-accent-warning" />
          {error || 'Trend data unavailable'}
        </div>
        <button onClick={load} className="flex items-center gap-1 text-accent-warning">
          <RefreshCw className="w-3 h-3" /> Retry
        </button>
      </div>
    );
  }

  const regressions = trends.regressions || [];
  const improvements = trends.improvements || [];
  const latestRegression = regressions[0];

  return (
    <div
      className={`p-4 rounded-xl border ${
        regressions.length > 0
          ? 'border-accent-danger/30 bg-accent-danger/10'
          : 'border-accent-success/20 bg-accent-success/10'
      } flex items-center justify-between`}
    >
      <div className="flex items-center gap-3">
        {regressions.length > 0 ? (
          <AlertTriangle className="w-5 h-5 text-accent-danger" />
        ) : (
          <CheckCircle className="w-5 h-5 text-accent-success" />
        )}
        <div>
          <div className="text-sm font-semibold text-white">
            {regressions.length > 0
              ? `${regressions.length} regression${regressions.length > 1 ? 's' : ''} detected`
              : 'No regressions detected'}
          </div>
          <div className="text-xs text-white/60">
            {latestRegression
              ? `${latestRegression.date || ''} • ${latestRegression.name || 'Unknown'} • ${latestRegression.delta || ''}`
              : `${improvements.length} improvements tracked`}
          </div>
        </div>
      </div>
      <div className="flex items-center gap-3 text-xs text-white/70">
        <div className="flex items-center gap-1">
          <Activity className="w-3 h-3" />
          {improvements.length} improvements
        </div>
        <button
          onClick={load}
          className="px-3 py-1 rounded-lg bg-white/10 hover:bg-white/20 text-white border border-white/10"
        >
          Refresh
        </button>
      </div>
    </div>
  );
}
