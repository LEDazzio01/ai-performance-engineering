'use client';

import { useState, useEffect } from 'react';
import { Network, Cpu, ArrowRightLeft, Loader2, AlertTriangle, RefreshCw } from 'lucide-react';
import { formatBytes, cn } from '@/lib/utils';
import { getGpuTopology, getGpuNvlink } from '@/lib/api';

export function MultiGpuTab() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [topology, setTopology] = useState<any>(null);
  const [nvlinks, setNvlinks] = useState<any>(null);

  async function loadData() {
    try {
      setLoading(true);
      setError(null);
      const [topoData, nvlinkData] = await Promise.all([
        getGpuTopology(),
        getGpuNvlink().catch(() => null),
      ]);
      setTopology(topoData);
      setNvlinks(nvlinkData);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load GPU topology');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="card">
        <div className="card-body flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 animate-spin text-accent-success" />
          <span className="ml-3 text-white/50">Loading GPU topology...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="card-body text-center py-16">
          <AlertTriangle className="w-12 h-12 text-accent-danger mx-auto mb-4" />
          <p className="text-white/70 mb-4">{error}</p>
          <button
            onClick={loadData}
            className="flex items-center gap-2 px-4 py-2 bg-accent-primary/20 text-accent-primary rounded-lg hover:bg-accent-primary/30 mx-auto"
          >
            <RefreshCw className="w-4 h-4" />
            Retry
          </button>
        </div>
      </div>
    );
  }

  const gpus = topology?.gpus || topology?.devices || [];
  const links = nvlinks?.links || nvlinks?.connections || [];

  const totalMemory = gpus.reduce((sum: number, g: any) => sum + (g.memory_total || 0), 0);
  const usedMemory = gpus.reduce((sum: number, g: any) => sum + (g.memory_used || 0), 0);
  const avgUtilization = gpus.length > 0 
    ? gpus.reduce((sum: number, g: any) => sum + (g.utilization || 0), 0) / gpus.length 
    : 0;
  const totalPower = gpus.reduce((sum: number, g: any) => sum + (g.power_draw || g.power || 0), 0);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Network className="w-5 h-5 text-accent-success" />
            <h2 className="text-lg font-semibold text-white">Multi-GPU Topology</h2>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-white/50">
              <span className="text-accent-success font-bold">{gpus.length}</span> GPUs detected
            </span>
            <button onClick={loadData} className="p-2 hover:bg-white/5 rounded-lg">
              <RefreshCw className="w-4 h-4 text-white/50" />
            </button>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card p-5">
          <div className="text-sm text-white/50 mb-2">Total VRAM</div>
          <div className="text-2xl font-bold text-accent-primary">
            {formatBytes(totalMemory * 1e9)}
          </div>
          <div className="text-sm text-white/40">
            {formatBytes(usedMemory * 1e9)} used
          </div>
        </div>
        <div className="card p-5">
          <div className="text-sm text-white/50 mb-2">Avg Utilization</div>
          <div className="text-2xl font-bold text-accent-secondary">
            {avgUtilization.toFixed(0)}%
          </div>
        </div>
        <div className="card p-5">
          <div className="text-sm text-white/50 mb-2">Total Power</div>
          <div className="text-2xl font-bold text-accent-warning">{totalPower}W</div>
        </div>
        <div className="card p-5">
          <div className="text-sm text-white/50 mb-2">NVLink Status</div>
          <div className="text-2xl font-bold text-accent-info">
            {links.length > 0 ? 'Active' : 'N/A'}
          </div>
        </div>
      </div>

      {/* GPU grid */}
      {gpus.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">GPU Devices</h3>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {gpus.map((gpu: any, i: number) => (
                <div
                  key={i}
                  className="p-6 bg-white/5 border border-white/10 rounded-xl"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className="w-12 h-12 bg-gradient-to-br from-accent-success to-accent-primary rounded-lg flex items-center justify-center">
                        <Cpu className="w-6 h-6 text-white" />
                      </div>
                      <div>
                        <div className="text-sm font-bold text-white">GPU {gpu.id || i}</div>
                        <div className="text-xs text-white/50">{gpu.name}</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-accent-primary">
                        {gpu.utilization}%
                      </div>
                      <div className="text-xs text-white/40">Utilization</div>
                    </div>
                  </div>

                  {/* Memory bar */}
                  <div className="mb-3">
                    <div className="flex justify-between text-xs text-white/50 mb-1">
                      <span>Memory</span>
                      <span>
                        {gpu.memory_used?.toFixed?.(1) || gpu.memory_used}GB / {gpu.memory_total}GB
                      </span>
                    </div>
                    <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-accent-primary to-accent-secondary rounded-full"
                        style={{ width: `${(gpu.memory_used / gpu.memory_total) * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* Stats */}
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-white/40">Temp:</span>{' '}
                      <span
                        className={cn(
                          'font-bold',
                          (gpu.temperature || 0) > 75 ? 'text-accent-danger' : 'text-accent-success'
                        )}
                      >
                        {gpu.temperature}°C
                      </span>
                    </div>
                    <div>
                      <span className="text-white/40">Power:</span>{' '}
                      <span className="font-bold text-accent-warning">{gpu.power_draw || gpu.power}W</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* NVLink connections */}
      {links.length > 0 && (
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <ArrowRightLeft className="w-5 h-5 text-accent-info" />
              <h3 className="font-medium text-white">NVLink Connections</h3>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/5">
                  <th className="px-5 py-3 text-left text-xs font-medium text-white/50 uppercase">
                    From
                  </th>
                  <th className="px-5 py-3 text-left text-xs font-medium text-white/50 uppercase">
                    To
                  </th>
                  <th className="px-5 py-3 text-left text-xs font-medium text-white/50 uppercase">
                    Version
                  </th>
                  <th className="px-5 py-3 text-right text-xs font-medium text-white/50 uppercase">
                    Bandwidth
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {links.map((link: any, i: number) => (
                  <tr key={i} className="hover:bg-white/[0.02]">
                    <td className="px-5 py-4">
                      <span className="font-bold text-accent-primary">GPU {link.from || link.src}</span>
                    </td>
                    <td className="px-5 py-4">
                      <span className="font-bold text-accent-secondary">GPU {link.to || link.dst}</span>
                    </td>
                    <td className="px-5 py-4 text-white/70">{link.version || 'NVLink'}</td>
                    <td className="px-5 py-4 text-right font-bold text-accent-success">
                      {link.bandwidth} GB/s
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
