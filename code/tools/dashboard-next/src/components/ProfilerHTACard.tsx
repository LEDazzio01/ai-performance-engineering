'use client';

import { ListChecks, RefreshCw } from 'lucide-react';
import { getProfilerHTA } from '@/lib/api';
import { EmptyState, ErrorState, LoadingState } from '@/components/DataState';
import { getErrorMessage, useApiQuery } from '@/lib/useApi';

export function ProfilerHTACard() {
  const htaQuery = useApiQuery('profiler/hta', getProfilerHTA);
  const data = htaQuery.data;

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <ListChecks className="w-5 h-5 text-accent-primary" />
          <h3 className="font-medium text-white">HTA Highlights</h3>
        </div>
        <button
          onClick={() => htaQuery.mutate()}
          className="p-2 rounded hover:bg-white/5 text-white/70"
          aria-label="Refresh HTA highlights"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>
      <div className="card-body space-y-2">
        {htaQuery.error ? (
          <ErrorState
            message={getErrorMessage(htaQuery.error, 'Failed to load HTA')}
            onRetry={() => htaQuery.mutate()}
          />
        ) : htaQuery.isLoading ? (
          <LoadingState inline message="Parsing HTA..." />
        ) : data?.top_kernels ? (
          <div className="space-y-2">
            {data.top_kernels.slice(0, 5).map((k: any, i: number) => (
              <div key={i} className="p-3 rounded-lg bg-white/5 border border-white/10 flex items-center justify-between">
                <div>
                  <div className="text-white font-semibold">{k.name}</div>
                  <div className="text-xs text-white/50">{k.pct?.toFixed?.(1) || '?'}% of time</div>
                </div>
                <div className="text-accent-primary font-mono text-sm">
                  {k.time_us ? `${(k.time_us / 1000).toFixed(2)} ms` : ''}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <EmptyState title="No HTA data available" description="Run a profile to generate HTA highlights." />
        )}
      </div>
    </div>
  );
}
