'use client';

import { useEffect, useState } from 'react';
import { ListChecks, Loader2, RefreshCw, AlertTriangle } from 'lucide-react';
import { scanAllBenchmarks } from '@/lib/api';

export function BenchmarkScannerCard() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<any>(null);

  const load = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await scanAllBenchmarks();
      setData(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to scan benchmarks');
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
          <h3 className="font-medium text-white">Benchmark Scanner</h3>
        </div>
        <button onClick={load} className="p-2 rounded hover:bg-white/5">
          <RefreshCw className="w-4 h-4 text-white/50" />
        </button>
      </div>
      <div className="card-body space-y-2">
        {loading ? (
          <div className="flex items-center gap-2 text-white/60">
            <Loader2 className="w-4 h-4 animate-spin" /> Scanning chapters and labs...
          </div>
        ) : error ? (
          <div className="flex items-center gap-2 text-sm text-accent-warning">
            <AlertTriangle className="w-4 h-4" /> {error}
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-3 text-sm text-white/80">
            <div>
              <div className="text-white/40 text-xs">Chapters</div>
              <div className="text-xl font-bold text-accent-primary">
                {data?.summary?.total_directories || data?.total_chapters || 0}
              </div>
            </div>
            <div>
              <div className="text-white/40 text-xs">Benchmarks</div>
              <div className="text-xl font-bold text-accent-secondary">
                {data?.summary?.total_benchmarks || data?.total_benchmarks || 0}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

