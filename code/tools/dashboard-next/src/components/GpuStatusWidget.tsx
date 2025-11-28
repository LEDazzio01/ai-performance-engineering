'use client';

import { useState, useEffect } from 'react';
import { Loader2, AlertCircle } from 'lucide-react';
import { getGpuInfo } from '@/lib/api';

export function GpuStatusWidget() {
  const [gpu, setGpu] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  async function load() {
    try {
      const data = await getGpuInfo();
      setGpu(data);
      setError(false);
    } catch (e) {
      setError(true);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    const interval = setInterval(load, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center gap-2 px-4 py-2 bg-white/5 rounded-full">
        <Loader2 className="w-4 h-4 animate-spin text-accent-primary" />
        <span className="text-sm text-white/50">Loading...</span>
      </div>
    );
  }

  if (error || !gpu) {
    return (
      <div className="flex items-center gap-2 px-4 py-2 bg-accent-danger/10 border border-accent-danger/20 rounded-full">
        <AlertCircle className="w-4 h-4 text-accent-danger" />
        <span className="text-sm text-accent-danger">GPU info unavailable</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3 px-4 py-2 bg-gradient-to-r from-accent-primary/10 to-accent-secondary/10 border border-accent-primary/20 rounded-full">
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-accent-success animate-pulse" />
        <span className="text-sm font-medium text-accent-primary">{gpu.name}</span>
      </div>
      <div className="h-4 w-px bg-white/20" />
      <div className="flex items-center gap-4 text-xs">
        <span className="text-white/70">
          <span className="text-accent-primary font-bold">{gpu.utilization}%</span> GPU
        </span>
        <span className="text-white/70">
          <span className="text-accent-secondary font-bold">
            {typeof gpu.memory_used === 'number' ? gpu.memory_used.toFixed(1) : gpu.memory_used}GB
          </span> / {gpu.memory_total}GB
        </span>
        <span className="text-white/70">
          <span className="text-accent-warning font-bold">{gpu.temperature}°C</span>
        </span>
      </div>
    </div>
  );
}
