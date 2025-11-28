'use client';

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { Cpu, RefreshCw } from 'lucide-react';
import { getProfilerRoofline, getRooflineInteractive, getHardwareCapabilities } from '@/lib/api';
import { useApiQuery, getErrorMessage } from '@/lib/useApi';
import { EmptyState, ErrorState, LoadingState } from '@/components/DataState';

export function RooflineTab() {
  const rooflineQuery = useApiQuery('profiler/roofline', async () => {
    const [rooflineData, hwData] = await Promise.allSettled([
      getProfilerRoofline().catch(() => getRooflineInteractive()),
      getHardwareCapabilities(),
    ]);

    const mainData = rooflineData.status === 'fulfilled' ? rooflineData.value : null;

    return {
      data: mainData,
      hardware: hwData.status === 'fulfilled' ? hwData.value : null,
    };
  });

  if (rooflineQuery.error) {
    return (
      <div className="card">
        <div className="card-body">
          <ErrorState
            message={getErrorMessage(rooflineQuery.error, 'Failed to load roofline data')}
            onRetry={() => rooflineQuery.mutate()}
          />
        </div>
      </div>
    );
  }

  if (rooflineQuery.isLoading) {
    return (
      <div className="card">
        <div className="card-body">
          <LoadingState message="Loading roofline data..." />
        </div>
      </div>
    );
  }

  const peakFlops =
    rooflineQuery.data?.hardware?.peak_flops || rooflineQuery.data?.data?.peak_flops || 312;
  const memoryBandwidth =
    rooflineQuery.data?.hardware?.memory_bandwidth ||
    rooflineQuery.data?.data?.memory_bandwidth ||
    3350;
  const ridgePoint = (peakFlops * 1000) / memoryBandwidth;
  const kernels = rooflineQuery.data?.data?.kernels || rooflineQuery.data?.data?.points || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Cpu className="w-5 h-5 text-accent-primary" />
            <h2 className="text-lg font-semibold text-white">Roofline Model Analysis</h2>
          </div>
          <button
            onClick={() => rooflineQuery.mutate()}
            className="p-2 hover:bg-white/5 rounded-lg transition-colors text-white/70 flex items-center gap-2"
            aria-label="Refresh roofline data"
          >
            <RefreshCw className="w-4 h-4" />
            {rooflineQuery.isValidating && <span className="text-xs">Refreshing…</span>}
          </button>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="p-4 bg-white/5 rounded-lg">
              <div className="text-sm text-white/50 mb-1">Peak Performance</div>
              <div className="text-2xl font-bold text-accent-primary">{peakFlops} TFLOPS</div>
            </div>
            <div className="p-4 bg-white/5 rounded-lg">
              <div className="text-sm text-white/50 mb-1">Memory Bandwidth</div>
              <div className="text-2xl font-bold text-accent-secondary">{memoryBandwidth} GB/s</div>
            </div>
            <div className="p-4 bg-white/5 rounded-lg">
              <div className="text-sm text-white/50 mb-1">Ridge Point</div>
              <div className="text-2xl font-bold text-accent-warning">{ridgePoint.toFixed(1)} FLOP/B</div>
            </div>
            <div className="p-4 bg-white/5 rounded-lg">
              <div className="text-sm text-white/50 mb-1">Kernels Analyzed</div>
              <div className="text-2xl font-bold text-white">{kernels.length}</div>
            </div>
          </div>

          {/* Roofline Chart */}
          {kernels.length > 0 ? (
            <div className="h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis
                    type="number"
                    dataKey="arithmetic_intensity"
                    name="Arithmetic Intensity"
                    unit=" FLOP/B"
                    scale="log"
                    domain={[0.1, 256]}
                    tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
                    axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                  />
                  <YAxis
                    type="number"
                    dataKey="performance"
                    name="Performance"
                    unit=" GFLOPS"
                    scale="log"
                    domain={[1, peakFlops * 1000]}
                    tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
                    axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(16, 16, 24, 0.95)',
                      border: '1px solid rgba(255,255,255,0.1)',
                      borderRadius: '8px',
                    }}
                    formatter={(value: any, name: string) => [
                      typeof value === 'number' ? value.toFixed(2) : value,
                      name,
                    ]}
                  />
                  <ReferenceLine
                    x={ridgePoint}
                    stroke="rgba(255,196,61,0.5)"
                    strokeDasharray="5 5"
                  />
                  <Scatter name="Kernels" data={kernels} fill="#9d4edd" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyState
              title="No kernels analyzed"
              description="Run a profile to populate roofline points."
              actionLabel="Refresh data"
              onAction={() => rooflineQuery.mutate()}
            />
          )}
        </div>
      </div>

      {/* Kernel details */}
      {kernels.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">Kernel Efficiency Analysis</h3>
          </div>
          <div className="card-body">
            <div className="space-y-3">
              {kernels.slice(0, 10).map((kernel: any, i: number) => (
                <div
                  key={i}
                  className="flex items-center gap-4 p-4 bg-white/5 rounded-lg"
                >
                  <div className="flex-1">
                    <div className="font-medium text-white">{kernel.name}</div>
                    <div className="text-sm text-white/50">
                      AI: {kernel.arithmetic_intensity?.toFixed?.(1) || 'N/A'} FLOP/B
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-lg font-bold text-accent-primary">
                      {kernel.performance?.toFixed?.(0) || 'N/A'} GFLOPS
                    </div>
                    <div className="text-sm text-white/50">
                      {((kernel.efficiency || 0) * 100).toFixed(0)}% efficiency
                    </div>
                  </div>
                  <div className="w-32">
                    <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-accent-primary to-accent-success rounded-full"
                        style={{ width: `${(kernel.efficiency || 0) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
