'use client';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { HardDrive, TrendingUp, AlertTriangle, RefreshCw } from 'lucide-react';
import { formatBytes } from '@/lib/utils';
import { getProfilerMemory } from '@/lib/api';
import { useApiQuery, getErrorMessage } from '@/lib/useApi';
import { EmptyState, ErrorState, LoadingState } from '@/components/DataState';

export function MemoryTab() {
  const memoryQuery = useApiQuery('profiler/memory', getProfilerMemory);
  const data = memoryQuery.data;

  const categoryColors: Record<string, string> = {
    Parameters: '#00f5d4',
    Optimizer: '#9d4edd',
    Activations: '#f72585',
    Gradients: '#ffc43d',
    Buffers: '#4cc9f0',
    System: '#868e96',
  };

  if (memoryQuery.error) {
    return (
      <div className="card">
        <div className="card-body">
          <ErrorState
            message={getErrorMessage(memoryQuery.error, 'Failed to load memory data')}
            onRetry={() => memoryQuery.mutate()}
          />
        </div>
      </div>
    );
  }

  if (memoryQuery.isLoading) {
    return (
      <div className="card">
        <div className="card-body">
          <LoadingState message="Loading memory data..." />
        </div>
      </div>
    );
  }

  const timeline = data?.timeline || [];
  const allocations = data?.allocations || data?.breakdown || [];
  const peakMemory = data?.peak_memory || data?.peak || 0;
  const totalMemory = data?.total_memory || data?.total || 80 * 1e9;
  const memoryUsagePercent = totalMemory > 0 ? (peakMemory / totalMemory) * 100 : 0;

  const pieData = allocations.map((a: any) => ({
    name: a.name,
    value: a.size || a.bytes,
    color: categoryColors[a.category] || '#868e96',
  }));

  const refreshButton = (
    <button
      onClick={() => memoryQuery.mutate()}
      className="p-2 hover:bg-white/5 rounded-lg"
      aria-label="Refresh memory data"
    >
      <RefreshCw className="w-4 h-4 text-white/50" />
    </button>
  );

  return (
    <div className="space-y-6">
      {/* Header stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card p-5">
          <div className="flex items-center gap-2 text-sm text-white/50 mb-2">
            <HardDrive className="w-4 h-4" />
            Peak Memory
          </div>
          <div className="text-2xl font-bold text-accent-primary">
            {formatBytes(peakMemory)}
          </div>
          <div className="text-sm text-white/40">
            {memoryUsagePercent.toFixed(1)}% of {formatBytes(totalMemory)}
          </div>
        </div>
        <div className="card p-5">
          <div className="flex items-center gap-2 text-sm text-white/50 mb-2">
            <TrendingUp className="w-4 h-4" />
            Current Allocated
          </div>
          <div className="text-2xl font-bold text-accent-secondary">
            {formatBytes(data?.current_allocated || timeline[timeline.length - 1]?.allocated || 0)}
          </div>
        </div>
        <div className="card p-5">
          <div className="flex items-center gap-2 text-sm text-white/50 mb-2">
            <HardDrive className="w-4 h-4" />
            Reserved
          </div>
          <div className="text-2xl font-bold text-accent-warning">
            {formatBytes(data?.reserved || timeline[timeline.length - 1]?.reserved || 0)}
          </div>
        </div>
        <div className="card p-5">
          {memoryUsagePercent > 90 ? (
            <>
              <div className="flex items-center gap-2 text-sm text-accent-danger mb-2">
                <AlertTriangle className="w-4 h-4" />
                Warning
              </div>
              <div className="text-lg font-bold text-accent-danger">Memory Pressure</div>
            </>
          ) : (
            <>
              <div className="flex items-center gap-2 text-sm text-accent-success mb-2">
                <HardDrive className="w-4 h-4" />
                Status
              </div>
              <div className="text-lg font-bold text-accent-success">Healthy</div>
            </>
          )}
        </div>
      </div>

      {/* Memory timeline */}
      {timeline.length > 0 ? (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">Memory Timeline</h3>
            {refreshButton}
          </div>
          <div className="card-body">
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={timeline} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis
                    dataKey="timestamp"
                    tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
                    axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                    tickFormatter={(v) => `${v}ms`}
                  />
                  <YAxis
                    tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
                    axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                    tickFormatter={(v) => formatBytes(v)}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(16, 16, 24, 0.95)',
                      border: '1px solid rgba(255,255,255,0.1)',
                      borderRadius: '8px',
                    }}
                    formatter={(value: number) => [formatBytes(value), '']}
                  />
                  <Area
                    type="monotone"
                    dataKey="reserved"
                    stroke="#ffc43d"
                    fill="rgba(255, 196, 61, 0.2)"
                    name="Reserved"
                  />
                  <Area
                    type="monotone"
                    dataKey="allocated"
                    stroke="#00f5d4"
                    fill="rgba(0, 245, 212, 0.3)"
                    name="Allocated"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      ) : (
        <EmptyState
          title="No timeline data"
          description="Run a profile to collect memory allocation timeline."
          actionLabel="Refresh"
          onAction={() => memoryQuery.mutate()}
        />
      )}

      {/* Memory breakdown */}
      {allocations.length > 0 ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="card">
            <div className="card-header">
              <h3 className="font-medium text-white">Memory Breakdown</h3>
            </div>
            <div className="card-body">
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={2}
                      dataKey="value"
                    >
                      {pieData.map((entry: any, index: number) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'rgba(16, 16, 24, 0.95)',
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: '8px',
                      }}
                      formatter={(value: number) => [formatBytes(value), '']}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <h3 className="font-medium text-white">Top Allocations</h3>
            </div>
            <div className="card-body space-y-3">
              {allocations.map((alloc: any, i: number) => (
                <div key={i} className="flex items-center gap-4">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: categoryColors[alloc.category] || '#868e96' }}
                  />
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm text-white">{alloc.name}</span>
                      <span className="text-sm font-mono text-white/70">
                        {formatBytes(alloc.size || alloc.bytes)}
                      </span>
                    </div>
                    <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${((alloc.size || alloc.bytes) / peakMemory) * 100}%`,
                          backgroundColor: categoryColors[alloc.category] || '#868e96',
                        }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <EmptyState
          title="No allocation breakdown"
          description="Profiling data did not include allocation breakdowns."
          actionLabel="Refresh"
          onAction={() => memoryQuery.mutate()}
        />
      )}
    </div>
  );
}
