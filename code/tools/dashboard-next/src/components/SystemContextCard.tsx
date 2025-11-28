'use client';

import { useEffect, useState } from 'react';
import { Clipboard, ClipboardCheck, DownloadCloud, Laptop2, Loader2, AlertTriangle } from 'lucide-react';
import { getSystemContext } from '@/lib/api';

export function SystemContextCard() {
  const [context, setContext] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getSystemContext();
      setContext(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load system context');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const gpu = context?.gpu;
  const gpuName =
    typeof gpu === 'string'
      ? gpu
      : gpu?.name || context?.accelerator || context?.gpu_name || 'Unknown';
  const gpuArch =
    typeof gpu === 'object' && gpu
      ? gpu.architecture || gpu.compute_capability
      : context?.architecture || context?.compute_capability;
  const gpuDisplay = [gpuName, gpuArch].filter(Boolean).join(' • ');
  const gpuMemoryTotal = typeof gpu === 'object' ? gpu?.memory?.total_gb : undefined;
  const gpuMemoryUsed = typeof gpu === 'object' ? gpu?.memory?.used_gb : undefined;
  const gpuUtil = typeof gpu === 'object' ? gpu?.current_state?.utilization_pct : undefined;

  const system = context?.system;
  const host = context?.hostname || context?.host || system?.hostname || 'Unknown';
  const cpu = system?.cpu?.model || context?.cpu_model || context?.cpu || 'Unknown';
  const os = context?.os || context?.platform || system?.os || 'Unknown';

  const software = context?.software || {};
  const cuda = context?.cuda_version || context?.cuda || software.cuda_runtime || software.cuda_driver || 'Unknown';
  const pytorch = context?.pytorch_version || software.pytorch || 'Unknown';

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(context, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setError('Clipboard copy failed');
    }
  };

  const handleExport = () => {
    try {
      const blob = new Blob([JSON.stringify(context, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `system-context-${new Date().toISOString().split('T')[0]}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      setError('Export failed');
    }
  };

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <Laptop2 className="w-5 h-5 text-accent-primary" />
          <h3 className="font-medium text-white">System Context</h3>
        </div>
        <button onClick={load} className="p-2 hover:bg-white/5 rounded-lg">
          <Loader2 className={`w-4 h-4 ${loading ? 'animate-spin text-accent-primary' : 'text-white/50'}`} />
        </button>
      </div>
      <div className="card-body space-y-3">
        {error && (
          <div className="text-sm text-accent-warning flex items-center gap-2">
            <AlertTriangle className="w-4 h-4" /> {error}
          </div>
        )}
        {loading ? (
          <div className="flex items-center gap-2 text-white/60">
            <Loader2 className="w-4 h-4 animate-spin" />
            Gathering hardware/software details...
          </div>
        ) : context ? (
          <>
            <div className="grid grid-cols-2 gap-3 text-sm text-white/80">
              <div>
                <div className="text-white/40 text-xs">Host</div>
                <div className="font-semibold">{host}</div>
              </div>
              <div>
                <div className="text-white/40 text-xs">GPU</div>
                <div className="font-semibold">{gpuDisplay || 'Unknown'}</div>
              </div>
              <div>
                <div className="text-white/40 text-xs">OS</div>
                <div className="font-semibold">{os}</div>
              </div>
              <div>
                <div className="text-white/40 text-xs">CUDA</div>
                <div className="font-semibold">{cuda}</div>
              </div>
              <div>
                <div className="text-white/40 text-xs">CPU</div>
                <div className="font-semibold">{cpu}</div>
              </div>
              <div>
                <div className="text-white/40 text-xs">PyTorch</div>
                <div className="font-semibold">{pytorch}</div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs text-white/60">
              <div className="rounded-lg border border-white/5 bg-white/5 p-3 space-y-1">
                <div className="text-white/40 uppercase tracking-wide text-[10px]">GPU State</div>
                <div>
                  Memory:{' '}
                  {typeof gpuMemoryUsed === 'number' && typeof gpuMemoryTotal === 'number'
                    ? `${gpuMemoryUsed.toFixed(2)} / ${gpuMemoryTotal.toFixed(1)} GB`
                    : 'Unknown'}
                </div>
                <div>Utilization: {typeof gpuUtil === 'number' ? `${gpuUtil}%` : 'Unknown'}</div>
              </div>
              <div className="rounded-lg border border-white/5 bg-white/5 p-3 space-y-1">
                <div className="text-white/40 uppercase tracking-wide text-[10px]">Toolkit</div>
                <div>CUDA: {cuda}</div>
                <div>PyTorch: {pytorch}</div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleCopy}
                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-sm text-white"
              >
                {copied ? <ClipboardCheck className="w-4 h-4 text-accent-success" /> : <Clipboard className="w-4 h-4" />}
                {copied ? 'Copied' : 'Copy JSON'}
              </button>
              <button
                onClick={handleExport}
                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-sm text-white"
              >
                <DownloadCloud className="w-4 h-4 text-accent-primary" />
                Export
              </button>
            </div>
          </>
        ) : (
          <div className="text-sm text-white/50">No context available.</div>
        )}
      </div>
    </div>
  );
}
