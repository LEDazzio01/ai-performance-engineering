'use client';

import { useEffect, useState } from 'react';
import { Server, Activity, Cpu, Database, Loader2, RefreshCw, AlertTriangle } from 'lucide-react';
import {
  getAnalysisFullSystem,
  getAnalysisCpuMemory,
  getAnalysisSystemParams,
  getAnalysisContainerLimits,
  getAnalysisPlaybooks,
} from '@/lib/api';

export function SystemDeepDiveCard() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<any>(null);

  const load = async () => {
    try {
      setLoading(true);
      setError(null);
      const [full, cpuMem, params, limits, playbooks] = await Promise.all([
        getAnalysisFullSystem().catch(() => null),
        getAnalysisCpuMemory().catch(() => null),
        getAnalysisSystemParams().catch(() => null),
        getAnalysisContainerLimits().catch(() => null),
        getAnalysisPlaybooks().catch(() => null),
      ]);
      setData({ full, cpuMem, params, limits, playbooks });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load system analysis');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const Item = ({ label, value }: { label: string; value: any }) => (
    <div className="p-3 rounded-lg bg-white/5 border border-white/10">
      <div className="text-xs text-white/50 uppercase">{label}</div>
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
          <Server className="w-5 h-5 text-accent-primary" />
          <h3 className="font-medium text-white">System Deep Dive</h3>
        </div>
        <button onClick={load} className="p-2 rounded hover:bg-white/5">
          <RefreshCw className="w-4 h-4 text-white/50" />
        </button>
      </div>
      <div className="card-body space-y-3">
        {loading ? (
          <div className="flex items-center gap-2 text-white/60">
            <Loader2 className="w-4 h-4 animate-spin" /> Collecting system signals...
          </div>
        ) : error ? (
          <div className="flex items-center gap-2 text-sm text-accent-warning">
            <AlertTriangle className="w-4 h-4" /> {error}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <Item label="Full System" value={data?.full?.summary || data?.full} />
            <Item label="CPU & Memory" value={data?.cpuMem} />
            <Item label="Parameters" value={data?.params} />
            <Item label="Container Limits" value={data?.limits} />
            <Item label="Playbooks" value={data?.playbooks?.playbooks || data?.playbooks} />
          </div>
        )}
      </div>
    </div>
  );
}

