'use client';

import { Flame, Zap, Clock, Activity, TrendingDown, Info } from 'lucide-react';

interface ApiBar {
  name: string;
  time_pct: number;
  total_time_ns: number;
  num_calls: number;
  type: 'sync' | 'malloc' | 'launch' | 'memcpy' | 'wait' | 'other';
}

interface KernelBar {
  name: string;
  time_pct: number;
  total_time_ns: number;
  instances: number;
}

interface ProfileSide {
  file: string;
  total_time_ms: number;
  api_bars: ApiBar[];
  kernel_bars: KernelBar[];
}

interface FlameGraphData {
  baseline: ProfileSide;
  optimized: ProfileSide;
  speedup: number;
  metrics: {
    baseline_sync_calls: number;
    optimized_sync_calls: number;
    sync_reduction_pct: number;
    baseline_device_sync: number;
    optimized_device_sync: number;
    device_sync_reduction_pct: number;
    optimized_wait_events: number;
  };
  insight: string;
  chapter?: string;
  error?: string;
}

interface FlameGraphComparisonProps {
  data: FlameGraphData | null;
  isLoading?: boolean;
}

const barTypeColors: Record<string, { bg: string; text: string }> = {
  sync: { bg: 'bg-gradient-to-r from-red-500 to-red-600', text: 'text-white' },
  malloc: { bg: 'bg-gradient-to-r from-purple-500 to-purple-600', text: 'text-white' },
  launch: { bg: 'bg-gradient-to-r from-violet-400 to-violet-500', text: 'text-gray-900' },
  memcpy: { bg: 'bg-gradient-to-r from-yellow-400 to-yellow-500', text: 'text-gray-900' },
  wait: { bg: 'bg-gradient-to-r from-emerald-400 to-emerald-500', text: 'text-gray-900' },
  other: { bg: 'bg-gradient-to-r from-gray-500 to-gray-600', text: 'text-white' },
};

const kernelColor = 'bg-gradient-to-r from-cyan-400 to-teal-500';

function FlameBar({ bar, isKernel = false }: { bar: ApiBar | KernelBar; isKernel?: boolean }) {
  const colors = isKernel
    ? { bg: kernelColor, text: 'text-gray-900' }
    : barTypeColors[(bar as ApiBar).type] || barTypeColors.other;
  
  const width = Math.max(bar.time_pct, 3); // Min 3% for visibility
  
  return (
    <div
      className={`h-7 flex items-center px-2 text-xs font-medium rounded transition-all hover:brightness-110 hover:shadow-lg cursor-pointer ${colors.bg} ${colors.text}`}
      style={{ width: `${width}%`, minWidth: '40px' }}
      title={`${bar.name}: ${bar.time_pct.toFixed(1)}% (${(bar.total_time_ns / 1_000_000).toFixed(2)}ms)`}
    >
      <span className="truncate">
        {bar.time_pct > 8 ? bar.name.slice(0, 20) : bar.time_pct.toFixed(0) + '%'}
      </span>
    </div>
  );
}

function FlameRow({ bars, isKernel = false }: { bars: (ApiBar | KernelBar)[]; isKernel?: boolean }) {
  return (
    <div className="flex gap-0.5 h-7 overflow-hidden rounded">
      {bars.map((bar, i) => (
        <FlameBar key={i} bar={bar} isKernel={isKernel} />
      ))}
    </div>
  );
}

function ProfileSection({
  title,
  icon: Icon,
  profile,
  variant,
}: {
  title: string;
  icon: typeof Flame;
  profile: ProfileSide;
  variant: 'baseline' | 'optimized';
}) {
  const borderColor = variant === 'baseline' ? 'border-red-500/30' : 'border-emerald-500/30';
  const titleColor = variant === 'baseline' ? 'text-red-400' : 'text-emerald-400';
  const badgeBg = variant === 'baseline' ? 'bg-red-500/20' : 'bg-emerald-500/20';
  
  return (
    <div className={`p-4 bg-white/5 rounded-xl border ${borderColor}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Icon className={`w-5 h-5 ${titleColor}`} />
          <h4 className={`font-semibold ${titleColor}`}>{title}</h4>
        </div>
        <span className={`px-3 py-1 rounded-full text-sm font-mono ${badgeBg} ${titleColor}`}>
          {profile.total_time_ms.toFixed(2)} ms
        </span>
      </div>
      
      <div className="space-y-3">
        <div>
          <div className="text-xs text-white/50 mb-1.5 uppercase tracking-wide">CUDA API Distribution</div>
          <FlameRow bars={profile.api_bars} />
        </div>
        
        <div>
          <div className="text-xs text-white/50 mb-1.5 uppercase tracking-wide">Kernel Distribution</div>
          <FlameRow bars={profile.kernel_bars} isKernel />
        </div>
      </div>
    </div>
  );
}

function MetricCard({
  label,
  baseline,
  optimized,
  reduction,
  icon: Icon,
}: {
  label: string;
  baseline: number | string;
  optimized: number | string;
  reduction?: number;
  icon: typeof Clock;
}) {
  return (
    <div className="p-3 bg-white/5 rounded-lg border border-white/10">
      <div className="flex items-center gap-2 text-xs text-white/50 mb-2">
        <Icon className="w-3.5 h-3.5" />
        {label}
      </div>
      <div className="flex items-center justify-between">
        <div className="text-sm">
          <span className="text-red-400">{baseline}</span>
          <span className="text-white/30 mx-2">→</span>
          <span className="text-emerald-400">{optimized}</span>
        </div>
        {reduction !== undefined && reduction > 0 && (
          <span className="text-xs text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded-full">
            -{reduction.toFixed(0)}%
          </span>
        )}
      </div>
    </div>
  );
}

export function FlameGraphComparison({ data, isLoading }: FlameGraphComparisonProps) {
  if (isLoading) {
    return (
      <div className="card">
        <div className="card-body flex items-center justify-center py-12">
          <div className="animate-pulse text-white/50">Loading flame graph comparison...</div>
        </div>
      </div>
    );
  }
  
  if (!data) {
    return null;
  }
  
  if (data.error) {
    return (
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Flame className="w-5 h-5 text-accent-warning" />
            <h3 className="font-medium text-white">Flame Graph Comparison</h3>
          </div>
        </div>
        <div className="card-body">
          <div className="text-center py-8">
            <Info className="w-8 h-8 text-white/30 mx-auto mb-3" />
            <p className="text-white/70 mb-2">{data.error}</p>
            {(data as any).hint && (
              <p className="text-sm text-white/50 font-mono">{(data as any).hint}</p>
            )}
          </div>
        </div>
      </div>
    );
  }
  
  const { baseline, optimized, speedup, metrics, insight } = data;
  
  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <Flame className="w-5 h-5 text-orange-400" />
          <h3 className="font-medium text-white">Flame Graph Comparison</h3>
        </div>
        <div className="flex items-center gap-3">
          <span className="px-4 py-1.5 rounded-full bg-gradient-to-r from-emerald-500 to-cyan-500 text-gray-900 font-bold text-sm flex items-center gap-1.5 shadow-lg shadow-emerald-500/20">
            <Zap className="w-4 h-4" />
            {speedup}x Speedup
          </span>
        </div>
      </div>
      
      <div className="card-body space-y-6">
        {/* Legend */}
        <div className="flex flex-wrap gap-3 text-xs">
          {Object.entries(barTypeColors).map(([type, colors]) => (
            <div key={type} className="flex items-center gap-1.5">
              <div className={`w-4 h-3 rounded ${colors.bg}`} />
              <span className="text-white/60 capitalize">{type}</span>
            </div>
          ))}
          <div className="flex items-center gap-1.5">
            <div className={`w-4 h-3 rounded ${kernelColor}`} />
            <span className="text-white/60">Kernel</span>
          </div>
        </div>
        
        {/* Side-by-side profiles */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <ProfileSection
            title="Baseline (Sequential)"
            icon={Clock}
            profile={baseline}
            variant="baseline"
          />
          <ProfileSection
            title="Optimized (Pipelined)"
            icon={Zap}
            profile={optimized}
            variant="optimized"
          />
        </div>
        
        {/* Metrics comparison */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricCard
            label="Stream Syncs"
            baseline={metrics.baseline_sync_calls}
            optimized={metrics.optimized_sync_calls}
            reduction={metrics.sync_reduction_pct}
            icon={Activity}
          />
          <MetricCard
            label="Device Syncs"
            baseline={metrics.baseline_device_sync}
            optimized={metrics.optimized_device_sync}
            reduction={metrics.device_sync_reduction_pct}
            icon={TrendingDown}
          />
          <MetricCard
            label="Wait Events"
            baseline="0"
            optimized={metrics.optimized_wait_events}
            icon={Clock}
          />
          <MetricCard
            label="Total Time"
            baseline={`${baseline.total_time_ms.toFixed(1)}ms`}
            optimized={`${optimized.total_time_ms.toFixed(1)}ms`}
            reduction={Math.round((1 - optimized.total_time_ms / baseline.total_time_ms) * 100)}
            icon={Flame}
          />
        </div>
        
        {/* Insight */}
        {insight && (
          <div className="p-4 bg-gradient-to-r from-emerald-500/10 to-cyan-500/10 rounded-lg border border-emerald-500/20">
            <div className="flex items-start gap-3">
              <Info className="w-5 h-5 text-emerald-400 mt-0.5 flex-shrink-0" />
              <div>
                <h4 className="font-medium text-emerald-400 mb-1">Optimization Insight</h4>
                <p className="text-sm text-white/80">{insight}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}







