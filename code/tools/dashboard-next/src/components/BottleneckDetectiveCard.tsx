'use client';

import { AlertTriangle, CheckCircle, Gauge, RefreshCw } from 'lucide-react';
import { getProfilerBottlenecks, getOptimizationScore } from '@/lib/api';
import { EmptyState, ErrorState, LoadingState } from '@/components/DataState';
import { getErrorMessage, useApiQuery } from '@/lib/useApi';

type BottleneckData = {
  bottlenecks: any[];
  recommendations: string[];
  score: any;
};

export function BottleneckDetectiveCard() {
  const bottleneckQuery = useApiQuery<BottleneckData>('profiler/bottlenecks', async () => {
    const [combined, scoreRes] = await Promise.allSettled([
      getProfilerBottlenecks(),
      getOptimizationScore(),
    ]);

    const combinedValue = combined.status === 'fulfilled' ? combined.value : null;
    const list =
      (combinedValue as any)?.profile?.bottlenecks ||
      (combinedValue as any)?.bottlenecks ||
      (Array.isArray(combinedValue) ? combinedValue : []);
    const recs = (combinedValue as any)?.recommendations || [];

    return {
      bottlenecks: list || [],
      recommendations: Array.isArray(recs) ? recs : [],
      score: scoreRes.status === 'fulfilled' ? scoreRes.value : null,
    };
  });

  const data = bottleneckQuery.data;

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-accent-warning" />
          <h3 className="font-medium text-white">Bottleneck Detective</h3>
        </div>
        <button
          onClick={() => bottleneckQuery.mutate()}
          className="p-2 hover:bg-white/5 rounded-lg text-white/70"
          aria-label="Refresh bottleneck analysis"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>
      <div className="card-body space-y-3">
        {bottleneckQuery.error ? (
          <ErrorState
            message={getErrorMessage(bottleneckQuery.error, 'Failed to load bottleneck analysis')}
            onRetry={() => bottleneckQuery.mutate()}
          />
        ) : bottleneckQuery.isLoading ? (
          <LoadingState inline message="Analyzing kernels..." />
        ) : (
          <>
            <div className="flex items-center gap-3 p-3 rounded-lg bg-white/5 border border-white/10">
              <Gauge className="w-5 h-5 text-accent-success" />
              <div>
                <div className="text-white font-semibold">
                  Optimization Score: {data?.score?.score ?? '--'}/100 {data?.score?.grade ? `(${data?.score.grade})` : ''}
                </div>
                <div className="text-xs text-white/50">{data?.score?.summary || 'LLM-guided bottleneck scoring'}</div>
              </div>
            </div>

            {data && data.bottlenecks.length > 0 ? (
              <div className="space-y-2">
                {data.bottlenecks.slice(0, 5).map((b, i) => (
                  <div
                    key={i}
                    className="p-3 rounded-lg bg-white/5 border border-white/10"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-white font-medium">{b.name || b.id || `Bottleneck ${i + 1}`}</span>
                      <span
                        className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                          b.severity === 'high'
                            ? 'bg-accent-danger/20 text-accent-danger'
                            : b.severity === 'medium'
                            ? 'bg-accent-warning/20 text-accent-warning'
                            : 'bg-accent-info/20 text-accent-info'
                        }`}
                      >
                        {b.severity || 'info'}
                      </span>
                    </div>
                    <div className="text-sm text-white/60 mt-1">
                      {b.description || b.detail || 'Detected performance bottleneck'}
                    </div>
                    {b.recommendation && (
                      <div className="text-sm text-accent-success mt-1">💡 {b.recommendation}</div>
                    )}
                  </div>
                ))}
                {data.recommendations.length > 0 && (
                  <div className="mt-3 p-3 rounded-lg bg-white/5 border border-white/10">
                    <div className="text-sm font-semibold text-accent-info mb-2">Recommended Next Steps</div>
                    <ul className="space-y-1 list-disc list-inside text-sm text-white/70">
                      {data.recommendations.slice(0, 5).map((rec, idx) => (
                        <li key={idx}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ) : (
              <EmptyState
                title="No bottlenecks reported"
                description="Run a profile to surface kernel-level bottlenecks."
              />
            )}
          </>
        )}
      </div>
    </div>
  );
}
