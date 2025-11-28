'use client';

import { useEffect, useState } from 'react';
import { Layers, Route, Loader2, RefreshCw, AlertTriangle } from 'lucide-react';
import {
  getParallelismCalibration,
  getParallelismClusters,
  getParallelismPareto,
  getShardingPlan,
  recommendParallelism,
} from '@/lib/api';

export function ParallelismExtrasCard() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<any>(null);

  const load = async () => {
    try {
      setLoading(true);
      setError(null);
      const [cal, clusters, pareto, sharding, rec] = await Promise.all([
        getParallelismCalibration().catch(() => null),
        getParallelismClusters().catch(() => null),
        getParallelismPareto().catch(() => null),
        getShardingPlan({
          model: 'llama-3.1-70b',
          dp: 8,
          tp: 1,
          pp: 1,
          batch: 8,
          seq: 4096,
        }).catch(() => null),
        recommendParallelism({
          model: 'llama-3.1-70b',
          batch_size: 8,
          seq_length: 4096,
          goal: 'throughput',
          is_training: true,
        }).catch(() => null),
      ]);
      setData({ cal, clusters, pareto, sharding, rec });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load parallelism extras');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const Card = ({ title, value }: { title: string; value: any }) => (
    <div className="p-3 rounded-lg bg-white/5 border border-white/10">
      <div className="text-xs text-white/50 uppercase">{title}</div>
      <div className="text-sm text-white/80 break-words">
        {value === null || value === undefined
          ? 'N/A'
          : typeof value === 'object'
          ? JSON.stringify(value, null, 2)
          : String(value)}
      </div>
    </div>
  );

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <Layers className="w-5 h-5 text-accent-secondary" />
          <h3 className="font-medium text-white">Parallelism Extras</h3>
        </div>
        <button onClick={load} className="p-2 rounded hover:bg-white/5">
          <RefreshCw className="w-4 h-4 text-white/50" />
        </button>
      </div>
      <div className="card-body space-y-3">
        {loading ? (
          <div className="flex items-center gap-2 text-white/60">
            <Loader2 className="w-4 h-4 animate-spin" /> Loading parallelism helpers...
          </div>
        ) : error ? (
          <div className="flex items-center gap-2 text-sm text-accent-warning">
            <AlertTriangle className="w-4 h-4" /> {error}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <Card title="Calibration" value={data?.cal} />
            <Card title="Clusters" value={data?.clusters} />
            <Card title="Pareto" value={data?.pareto} />
            <Card title="Sharding Plan" value={data?.sharding} />
            <Card title="Recommendation" value={data?.rec} />
          </div>
        )}
      </div>
    </div>
  );
}

