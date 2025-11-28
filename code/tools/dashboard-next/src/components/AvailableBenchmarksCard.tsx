'use client';

import { useEffect, useState } from 'react';
import { ListChecks, Loader2, RefreshCw } from 'lucide-react';
import { getAvailableBenchmarks } from '@/lib/api';

export function AvailableBenchmarksCard() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<any>(null);

  const load = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await getAvailableBenchmarks();
      setData(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load availability');
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
          <ListChecks className="w-5 h-5 text-accent-success" />
          <h3 className="font-medium text-white">Available Benchmarks</h3>
        </div>
        <button onClick={load} className="p-2 hover:bg-white/5 rounded-lg">
          <RefreshCw className="w-4 h-4 text-white/50" />
        </button>
      </div>
      <div className="card-body">
        {loading ? (
          <div className="flex items-center gap-2 text-white/60">
            <Loader2 className="w-4 h-4 animate-spin" /> Scanning chapters...
          </div>
        ) : error ? (
          <div className="text-sm text-accent-warning">{error}</div>
        ) : (
          <div className="grid grid-cols-3 gap-3 text-sm text-white/80">
            <div>
              <div className="text-white/40 text-xs">Chapters</div>
              <div className="text-xl font-bold text-accent-primary">
                {data?.total_chapters ?? data?.chapters?.length ?? 0}
              </div>
            </div>
            <div>
              <div className="text-white/40 text-xs">Labs</div>
              <div className="text-xl font-bold text-accent-secondary">
                {data?.total_labs ?? data?.labs?.length ?? 0}
              </div>
            </div>
            <div>
              <div className="text-white/40 text-xs">Benchmarks</div>
              <div className="text-xl font-bold text-accent-success">
                {data?.total_benchmarks ?? 0}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

