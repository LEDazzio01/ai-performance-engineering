'use client';

import { useEffect, useState } from 'react';
import { Network, Loader2, AlertTriangle, RefreshCw } from 'lucide-react';
import { getParallelismTroubleshootTopics, getParallelismPresets } from '@/lib/api';

export function NcclTuningCard() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [topics, setTopics] = useState<any[]>([]);
  const [presets, setPresets] = useState<any[]>([]);

  const load = async () => {
    try {
      setLoading(true);
      setError(null);
      const [t, p] = await Promise.all([
        getParallelismTroubleshootTopics().catch(() => null),
        getParallelismPresets().catch(() => []),
      ]);
      const topicsList = (t as any)?.topics || (Array.isArray(t) ? t : []);
      const presetList = (p as any)?.presets || (Array.isArray(p) ? p : []);
      setTopics(topicsList || []);
      setPresets(presetList || []);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load NCCL guidance');
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
          <Network className="w-5 h-5 text-accent-info" />
          <h3 className="font-medium text-white">NCCL / Distributed Tuning</h3>
        </div>
        <button onClick={load} className="p-2 hover:bg-white/5 rounded-lg">
          <RefreshCw className="w-4 h-4 text-white/50" />
        </button>
      </div>
      <div className="card-body space-y-3">
        {loading ? (
          <div className="flex items-center gap-2 text-white/60">
            <Loader2 className="w-4 h-4 animate-spin" /> Loading NCCL guidance...
          </div>
        ) : error ? (
          <div className="flex items-center gap-2 text-accent-warning">
            <AlertTriangle className="w-4 h-4" /> {error}
          </div>
        ) : (
          <>
            {topics.length > 0 && (
              <div className="space-y-2">
                {topics.slice(0, 5).map((t, i) => (
                  <div key={i} className="p-3 rounded-lg bg-white/5 border border-white/10">
                    <div className="text-white font-semibold">{t.title || t.topic}</div>
                    <div className="text-sm text-white/60">{t.tip || t.description}</div>
                  </div>
                ))}
              </div>
            )}
            {presets.length > 0 && (
              <div>
                <div className="text-xs text-white/50 uppercase mb-2">Presets</div>
                <div className="flex flex-wrap gap-2">
                  {presets.slice(0, 4).map((p, i) => (
                    <span
                      key={i}
                      className="px-3 py-1 rounded-full bg-accent-primary/10 text-accent-primary text-sm"
                    >
                      {p.name || p.strategy || 'Preset'}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {topics.length === 0 && presets.length === 0 && (
              <div className="text-sm text-white/50">No NCCL guidance available from backend.</div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
