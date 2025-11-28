'use client';

import { useState, useEffect } from 'react';
import { Package, Loader2, AlertCircle } from 'lucide-react';
import { getSoftwareInfo } from '@/lib/api';

export function SoftwareStackWidget() {
  const [info, setInfo] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    async function load() {
      try {
        const data = await getSoftwareInfo();
        setInfo(data);
        setError(false);
      } catch (e) {
        setError(true);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center gap-2 px-4 py-2 bg-white/5 rounded-full">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span className="text-sm text-white/50">Loading versions...</span>
      </div>
    );
  }

  if (error || !info) {
    return (
      <div className="flex items-center gap-2 px-4 py-2 bg-accent-danger/10 border border-accent-danger/20 rounded-full">
        <AlertCircle className="w-4 h-4 text-accent-danger" />
        <span className="text-sm text-accent-danger">Software info unavailable</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3 px-4 py-2 bg-white/5 border border-white/10 rounded-full">
      <Package className="w-4 h-4 text-accent-info" />
      <div className="flex items-center gap-3 text-xs">
        <span className="text-white/70">
          Python <span className="text-accent-info font-bold">{info.python_version}</span>
        </span>
        <span className="text-white/70">
          PyTorch <span className="text-accent-warning font-bold">{info.pytorch_version}</span>
        </span>
        <span className="text-white/70">
          CUDA <span className="text-accent-success font-bold">{info.cuda_version}</span>
        </span>
        {info.triton_version && (
          <span className="text-white/70">
            Triton <span className="text-accent-secondary font-bold">{info.triton_version}</span>
          </span>
        )}
      </div>
    </div>
  );
}
