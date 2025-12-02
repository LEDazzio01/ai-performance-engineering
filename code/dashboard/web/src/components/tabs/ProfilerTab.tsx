'use client';

import { useEffect, useMemo, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { Flame, Layers, Clock, Cpu, RefreshCw, Sparkles, Rocket } from 'lucide-react';
import {
  getProfilerKernels,
  getProfilerFlame,
  getProfilerBottlenecks,
  getOptimizationScore,
  getProfilerTimeline,
  analyzeKernel,
  askProfiler,
  generatePatch,
  startNsightSystemsCapture,
  startNsightComputeCapture,
  startTorchProfilerCapture,
  startHTACapture,
  fetchJobStatus,
  fetchNsightJobStatus,
  fetchMcpJobStatus,
  getProfilerTorch,
} from '@/lib/api';
import { BottleneckDetectiveCard } from '@/components/BottleneckDetectiveCard';
import { ProfilerHTACard } from '@/components/ProfilerHTACard';
import { ProfilesListCard } from '@/components/ProfilesListCard';
import { useApiMutation, useApiQuery, getErrorMessage } from '@/lib/useApi';
import { EmptyState, ErrorState, LoadingState } from '@/components/DataState';

type KernelRow = {
  name: string;
  duration_ms?: number;
  duration?: number;
  calls?: number;
  category?: string;
};

type ProfilerDataset = {
  kernels: KernelRow[];
  bottlenecks: any;
  flame: any;
  score: any;
  timeline: Record<string, any[]> | null;
};

const categoryColors: Record<string, string> = {
  GEMM: '#00f5d4',
  Normalization: '#9d4edd',
  Activation: '#f72585',
  Elementwise: '#ffc43d',
  Reduction: '#4cc9f0',
  Memory: '#00f5a0',
  Regularization: '#ff6b6b',
  Embedding: '#845ef7',
  Other: '#868e96',
};

export function ProfilerTab() {
  const [kernelCode, setKernelCode] = useState('');
  const [kernelGoal, setKernelGoal] = useState('Optimize occupancy and memory traffic');
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [patchResult, setPatchResult] = useState<any>(null);
  const [askQuestion, setAskQuestion] = useState('');
  const [askResult, setAskResult] = useState<string | null>(null);
  const [toolError, setToolError] = useState<string | null>(null);
  // Single async toggle for all profiler operations
  const [runAsync, setRunAsync] = useState(false);

  const [nsysCommand, setNsysCommand] = useState('python -c "print(123)"');
  const [nsysPreset, setNsysPreset] = useState<'light' | 'full'>('full');
  const [nsysFullTimeline, setNsysFullTimeline] = useState(false);
  const [nsysTimeout, setNsysTimeout] = useState<number | ''>('');
  const [nsysForceLineinfo, setNsysForceLineinfo] = useState(true);
  const [nsysResult, setNsysResult] = useState<any>(null);
  const [nsysLoading, setNsysLoading] = useState(false);

  const [ncuCommand, setNcuCommand] = useState('python -c "print(456)"');
  const [ncuWorkload, setNcuWorkload] = useState('memory_bound');
  const [ncuTimeout, setNcuTimeout] = useState<number | ''>('');
  const [ncuForceLineinfo, setNcuForceLineinfo] = useState(true);
  const [ncuResult, setNcuResult] = useState<any>(null);
  const [ncuLoading, setNcuLoading] = useState(false);

  const [torchScript, setTorchScript] = useState('ch01/baseline.py');
  const [torchArgs, setTorchArgs] = useState('');
  const [torchMode, setTorchMode] = useState('full');
  const [torchForceLineinfo, setTorchForceLineinfo] = useState(true);
  const [torchUseNvtx, setTorchUseNvtx] = useState(true);
  const [torchNvtxLabel, setTorchNvtxLabel] = useState('aisp_torch_profile');
  const [torchTimeout, setTorchTimeout] = useState<number | ''>('');
  const [torchResult, setTorchResult] = useState<any>(null);
  const [torchLoading, setTorchLoading] = useState(false);

  const [htaCommand, setHtaCommand] = useState('python -c "print(789)"');
  const [htaPreset, setHtaPreset] = useState<'light' | 'full'>('full');
  const [htaForceLineinfo, setHtaForceLineinfo] = useState(true);
  const [htaTimeout, setHtaTimeout] = useState<number | ''>('');
  const [htaResult, setHtaResult] = useState<any>(null);
  const [htaLoading, setHtaLoading] = useState(false);

  const [jobId, setJobId] = useState('');
  const [jobStatus, setJobStatus] = useState<any>(null);
  const [isPolling, setIsPolling] = useState(false);

  // Auto-poll for job status when a job is running (uses UNIFIED endpoint)
  useEffect(() => {
    if (!jobId || !isPolling) return;
    
    const pollInterval = setInterval(async () => {
      try {
        const status = await fetchJobStatus(jobId); // Uses unified endpoint
        setJobStatus(status);
        
        // Stop polling if job completed or errored
        if (status.status === 'completed' || status.status === 'error' || status.status === 'not_found') {
          setIsPolling(false);
        }
      } catch (e: any) {
        setJobStatus({ error: e.message });
        setIsPolling(false);
      }
    }, 3000); // Poll every 3 seconds
    
    return () => clearInterval(pollInterval);
  }, [jobId, isPolling]);

  const profilerQuery = useApiQuery<ProfilerDataset>('profiler/summary', async () => {
    const [kernelsRes, flameRes, bottlenecksRes, scoreRes, timelineRes] = await Promise.allSettled([
      getProfilerKernels(),
      getProfilerFlame(),
      getProfilerBottlenecks(),
      getOptimizationScore(),
      getProfilerTimeline(),
    ]);

    if (kernelsRes.status === 'rejected') {
      throw kernelsRes.reason;
    }

    const normalizedKernels =
      Array.isArray(kernelsRes.value) ? kernelsRes.value : (kernelsRes.value as any)?.kernels || [];

    return {
      kernels: normalizedKernels,
      flame: flameRes.status === 'fulfilled' ? flameRes.value : null,
      bottlenecks: bottlenecksRes.status === 'fulfilled' ? bottlenecksRes.value : null,
      score: scoreRes.status === 'fulfilled' ? scoreRes.value : null,
      timeline: timelineRes.status === 'fulfilled' ? (timelineRes.value as any) : null,
    };
  });

  const torchProfilerQuery = useApiQuery('profiler/torch', getProfilerTorch);
  const torchProfilerData = torchProfilerQuery.data as any;

  const analyzeMutation = useApiMutation('profiler/analyze', analyzeKernel);
  const patchMutation = useApiMutation('profiler/patch', generatePatch);
  const askMutation = useApiMutation('profiler/ask', askProfiler);

  const toolBusy: 'analyze' | 'patch' | 'ask' | null = analyzeMutation.isMutating
    ? 'analyze'
    : patchMutation.isMutating
    ? 'patch'
    : askMutation.isMutating
    ? 'ask'
    : null;

  const kernels = useMemo(() => profilerQuery.data?.kernels || [], [profilerQuery.data?.kernels]);
  const flame = profilerQuery.data?.flame;
  const bottlenecks = profilerQuery.data?.bottlenecks;
  const score = profilerQuery.data?.score;
  const timeline = profilerQuery.data?.timeline;

  const chartData = useMemo(
    () =>
      kernels.map((k) => ({
        ...k,
        duration_ms: k.duration_ms ?? k.duration ?? 0,
      })),
    [kernels]
  );

  const totalTime = useMemo(
    () => kernels.reduce((sum, k) => sum + (k.duration_ms || k.duration || 0), 0),
    [kernels]
  );

  const handleRefresh = () => {
    void profilerQuery.mutate();
    void torchProfilerQuery.mutate();
  };

  const runKernelAnalysis = async () => {
    if (!kernelCode.trim()) return;
    setToolError(null);
    try {
      const res = await analyzeMutation.trigger({ code: kernelCode, goal: kernelGoal });
      setAnalysisResult(res);
    } catch (err) {
      setToolError(getErrorMessage(err, 'Failed to analyze kernel'));
    }
  };

  const runPatch = async () => {
    if (!kernelCode.trim()) return;
    setToolError(null);
    try {
      const res = await patchMutation.trigger({ code: kernelCode, goal: kernelGoal });
      setPatchResult(res);
    } catch (err) {
      setToolError(getErrorMessage(err, 'Failed to generate patch'));
    }
  };

  const runAsk = async () => {
    if (!askQuestion.trim()) return;
    setToolError(null);
    try {
      const res = await askMutation.trigger({ question: askQuestion, code: kernelCode || undefined });
      const payload = res as any;
      setAskResult(payload?.answer || payload?.response || JSON.stringify(payload));
    } catch (err) {
      setToolError(getErrorMessage(err, 'Failed to ask profiler AI'));
    }
  };

  if (profilerQuery.isLoading) {
    return (
      <div className="card">
        <div className="card-body">
          <LoadingState message="Loading profiler data..." />
        </div>
      </div>
    );
  }

  if (profilerQuery.error) {
    return (
      <div className="card">
        <div className="card-body">
          <ErrorState
            message={getErrorMessage(profilerQuery.error, 'Failed to load profiler data')}
            onRetry={handleRefresh}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Flame className="w-5 h-5 text-accent-tertiary" />
            <h2 className="text-lg font-semibold text-white">GPU Kernel Profiler</h2>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleRefresh}
              className="flex items-center gap-2 rounded-lg p-2 hover:bg-white/5 text-white/70"
              aria-label="Refresh profiler data"
            >
              <RefreshCw className="w-4 h-4" />
              {profilerQuery.isValidating && <span className="text-xs">Refreshing…</span>}
            </button>
          </div>
        </div>
        <div className="card-body">
          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="p-4 bg-white/5 rounded-lg">
              <div className="flex items-center gap-2 text-sm text-white/50 mb-1">
                <Clock className="w-4 h-4" />
                Total GPU Time
              </div>
              <div className="text-2xl font-bold text-accent-primary">
                {totalTime.toFixed(1)}ms
              </div>
            </div>
            <div className="p-4 bg-white/5 rounded-lg">
              <div className="flex items-center gap-2 text-sm text-white/50 mb-1">
                <Layers className="w-4 h-4" />
                Unique Kernels
              </div>
              <div className="text-2xl font-bold text-accent-secondary">{kernels.length}</div>
            </div>
            <div className="p-4 bg-white/5 rounded-lg">
              <div className="flex items-center gap-2 text-sm text-white/50 mb-1">
                <Cpu className="w-4 h-4" />
                Optimization Score
              </div>
              <div className="text-2xl font-bold text-accent-success">
                {score?.score || 'N/A'}
              </div>
            </div>
            <div className="p-4 bg-white/5 rounded-lg">
              <div className="flex items-center gap-2 text-sm text-white/50 mb-1">
                <Flame className="w-4 h-4" />
                Bottlenecks
              </div>
              <div className="text-2xl font-bold text-accent-warning">
                {bottlenecks?.profile?.bottlenecks?.length || bottlenecks?.bottlenecks?.length || bottlenecks?.count || 0}
              </div>
            </div>
          </div>

          {/* Chart */}
          {chartData.length > 0 && (
            <div className="h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={chartData.slice(0, 15)}
                  layout="vertical"
                  margin={{ top: 10, right: 30, left: 150, bottom: 10 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis
                    type="number"
                    tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
                    axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                    unit="ms"
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 10 }}
                    axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                    width={150}
                    tickFormatter={(v) => (v.length > 25 ? v.slice(0, 25) + '...' : v)}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(16, 16, 24, 0.95)',
                      border: '1px solid rgba(255,255,255,0.1)',
                      borderRadius: '8px',
                    }}
                  />
                  <Bar dataKey="duration_ms" radius={[0, 4, 4, 0]}>
                    {chartData.slice(0, 15).map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={categoryColors[entry.category] || '#00f5d4'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>

      {flame && (
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <Flame className="w-5 h-5 text-accent-primary" />
              <h3 className="font-medium text-white">Flame Graph Summary</h3>
            </div>
          </div>
          <div className="card-body space-y-2">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm text-white/80">
              <div>
                <div className="text-white/40 text-xs">Nodes</div>
                <div className="text-lg font-bold text-accent-primary">
                  {flame.nodes?.length || flame.length || Object.keys(flame || {}).length}
                </div>
              </div>
              <div>
                <div className="text-white/40 text-xs">Total Samples</div>
                <div className="text-lg font-bold text-accent-secondary">
                  {flame.total_samples || flame.total || '—'}
                </div>
              </div>
              <div>
                <div className="text-white/40 text-xs">Root</div>
                <div className="font-semibold">
                  {flame.name || flame.root || 'main'}
                </div>
              </div>
            </div>
            <div className="space-y-1">
              {(flame.nodes || []).slice(0, 5).map((n: any, i: number) => (
                <div key={i} className="flex items-center justify-between p-2 bg-white/5 rounded-lg">
                  <span className="text-sm text-white/80 truncate">{n.name || n.id || `Node ${i + 1}`}</span>
                  <div className="text-right">
                    {n.file && (
                      <div className="text-[11px] text-white/50">
                        {n.file}:{n.line || '?'}
                      </div>
                    )}
                    <span className="text-xs text-accent-primary font-mono">
                      {n.value || n.samples || n.self || 0}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <ProfilerHTACard />
        <ProfilesListCard />
        <div className="card">
          <div className="card-header flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Flame className="w-4 h-4 text-accent-secondary" />
              <h3 className="font-medium text-white">Torch Profiler</h3>
            </div>
            <button
              onClick={() => torchProfilerQuery.mutate()}
              className="p-2 rounded hover:bg-white/5 text-white/70"
              aria-label="Refresh torch profiler"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
          <div className="card-body space-y-3">
            {torchProfilerQuery.error ? (
              <ErrorState message={getErrorMessage(torchProfilerQuery.error, 'Failed to load torch profiler')} onRetry={() => torchProfilerQuery.mutate()} />
            ) : torchProfilerQuery.isLoading ? (
              <LoadingState inline message="Loading torch profiler..." />
            ) : torchProfilerData?.top_ops?.length ? (
              <div className="space-y-2">
                {torchProfilerData.top_ops.slice(0, 4).map((op: any, idx: number) => (
                  <div key={idx} className="p-3 rounded-lg bg-white/5 border border-white/10 flex items-center justify-between">
                    <div>
                      <div className="text-white font-semibold truncate max-w-[180px]">{op.name || 'op'}</div>
                      <div className="text-xs text-white/50">{op.count} calls</div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-accent-primary font-mono">{(op.cuda_time_total_us / 1000).toFixed(2)} ms</div>
                      <div className="text-[11px] text-white/40">cuda total</div>
                    </div>
                  </div>
                ))}
                <div className="text-xs text-white/50">
                  Trace: {torchProfilerData.trace_path || 'n/a'}
                </div>
              </div>
            ) : (
              <EmptyState title="No torch.profiler data" description="Run a torch capture below to populate this card." />
            )}
          </div>
        </div>
      </div>

      {timeline && (
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <Clock className="w-5 h-5 text-accent-info" />
              <h3 className="font-medium text-white">CPU/GPU Timeline (with source)</h3>
            </div>
          </div>
          <div className="card-body grid grid-cols-1 lg:grid-cols-2 gap-3">
            {['cpu', 'gpu'].map((key) => {
              const list = (timeline as any)?.[key] || [];
              return (
                <div key={key} className="space-y-2">
                  <div className="text-sm text-white/60 uppercase">{key}</div>
                  {list.slice(0, 6).map((ev: any, idx: number) => (
                    <div key={idx} className="p-3 rounded-lg bg-white/5 border border-white/10">
                      <div className="flex items-center justify-between">
                        <span className="text-white font-semibold truncate">{ev.name}</span>
                        <span className="text-xs text-accent-primary font-mono">
                          {ev.duration_ms?.toFixed?.(2) || ev.duration_ms} ms
                        </span>
                      </div>
                      <div className="text-xs text-white/50">
                        {ev.start_ms?.toFixed?.(2)} ms · {ev.cat || key}
                      </div>
                      {(ev.file || ev.line) && (
                        <div className="text-xs text-accent-secondary mt-1">
                          {ev.file || 'unknown'}{ev.line ? `:${ev.line}` : ''}
                        </div>
                      )}
                    </div>
                  ))}
                  {list.length === 0 && <div className="text-sm text-white/50">No timeline data.</div>}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Nsight capture quick controls */}
      <div className="card">
        <div className="card-header flex items-center gap-2">
          <Rocket className="w-4 h-4 text-accent-primary" />
          <h3 className="font-medium text-white">Nsight Capture (nsys/ncu)</h3>
          <label className="flex items-center gap-2 text-sm">
            <input 
              type="checkbox" 
              className="accent-accent-info" 
              checked={runAsync} 
              onChange={(e) => setRunAsync(e.target.checked)} 
            />
            <span className="text-accent-info">Run Async</span>
            <span className="text-white/40 text-xs">(background, poll for status)</span>
          </label>
        </div>
        <div className="card-body space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-xs text-white/60">Nsight Systems command</label>
              <input
                className="w-full rounded-lg bg-white/10 border border-white/10 px-3 py-2 text-sm text-white"
                value={nsysCommand}
                onChange={(e) => setNsysCommand(e.target.value)}
              />
              <div className="flex flex-wrap gap-3 text-xs text-white/70 items-center">
                <label className="flex items-center gap-2">
                  Preset
                  <select
                    className="bg-white/10 border border-white/10 rounded-lg px-2 py-1 text-white"
                    value={nsysPreset}
                    onChange={(e) => setNsysPreset(e.target.value as 'light' | 'full')}
                  >
                    <option value="full">full (default)</option>
                    <option value="light">light</option>
                  </select>
                </label>
                <label className="flex items-center gap-2">
                  <input type="checkbox" className="accent-accent-primary" checked={nsysFullTimeline} onChange={(e) => setNsysFullTimeline(e.target.checked)} />
                  Full timeline
                </label>
                <label className="flex items-center gap-2">
                  <input type="checkbox" className="accent-accent-primary" checked={nsysForceLineinfo} onChange={(e) => setNsysForceLineinfo(e.target.checked)} />
                  Force lineinfo
                </label>
                <label className="flex items-center gap-2">
                  Timeout (s)
                  <input
                    type="number"
                    className="bg-white/10 border border-white/10 rounded-lg px-2 py-1 w-20 text-white"
                    value={nsysTimeout}
                    onChange={(e) => setNsysTimeout(e.target.value === '' ? '' : Number(e.target.value))}
                  />
                </label>
              </div>
              <button
                className="rounded-lg bg-accent-primary/20 px-3 py-2 text-sm text-accent-primary hover:bg-accent-primary/30 transition-colors"
                onClick={async () => {
                  if (!nsysCommand.trim()) {
                    setNsysResult({ error: 'Command is required' });
                    return;
                  }
                  setNsysLoading(true);
                  setNsysResult(null);
                  try {
                    const res = await startNsightSystemsCapture({
                      command: nsysCommand,
                      preset: nsysPreset,
                      full_timeline: nsysFullTimeline,
                      async: runAsync,
                      force_lineinfo: nsysForceLineinfo,
                      timeout_seconds: nsysTimeout === '' ? undefined : Number(nsysTimeout),
                    });
                    setNsysResult(res);
                    if (res.job_id) {
                      setJobId(res.job_id);
                      setJobStatus({ status: 'running', job_id: res.job_id });
                      if (runAsync) setIsPolling(true); // Start auto-polling
                    }
                  } catch (e: any) {
                    setNsysResult({ error: e.message });
                  } finally {
                    setNsysLoading(false);
                  }
                }}
              >
                {nsysLoading ? 'Running...' : 'Run Nsight Systems'}
              </button>
              <ResultPanel title="Nsight Systems Result" content={nsysResult} />
            </div>

            <div className="space-y-2">
              <label className="text-xs text-white/60">Nsight Compute command</label>
              <input
                className="w-full rounded-lg bg-white/10 border border-white/10 px-3 py-2 text-sm text-white"
                value={ncuCommand}
                onChange={(e) => setNcuCommand(e.target.value)}
              />
              <div className="flex flex-wrap gap-3 text-xs text-white/70 items-center">
                <label className="flex items-center gap-2">
                  Workload
                  <select
                    className="bg-white/10 border border-white/10 rounded-lg px-2 py-1 text-white"
                    value={ncuWorkload}
                    onChange={(e) => setNcuWorkload(e.target.value)}
                  >
                    <option value="memory_bound">memory_bound</option>
                    <option value="compute_bound">compute_bound</option>
                    <option value="tensor_core">tensor_core</option>
                    <option value="communication">communication</option>
                    <option value="occupancy">occupancy</option>
                  </select>
                </label>
                <label className="flex items-center gap-2">
                  <input type="checkbox" className="accent-accent-primary" checked={ncuForceLineinfo} onChange={(e) => setNcuForceLineinfo(e.target.checked)} />
                  Force lineinfo
                </label>
                <label className="flex items-center gap-2">
                  Timeout (s)
                  <input
                    type="number"
                    className="bg-white/10 border border-white/10 rounded-lg px-2 py-1 w-20 text-white"
                    value={ncuTimeout}
                    onChange={(e) => setNcuTimeout(e.target.value === '' ? '' : Number(e.target.value))}
                  />
                </label>
              </div>
              <button
                className="rounded-lg bg-accent-warning/20 px-3 py-2 text-sm text-accent-warning hover:bg-accent-warning/30 transition-colors"
                onClick={async () => {
                  if (!ncuCommand.trim()) {
                    setNcuResult({ error: 'Command is required' });
                    return;
                  }
                  setNcuLoading(true);
                  setNcuResult(null);
                  try {
                    const res = await startNsightComputeCapture({
                      command: ncuCommand,
                      workload_type: ncuWorkload,
                      async: runAsync,
                      force_lineinfo: ncuForceLineinfo,
                      timeout_seconds: ncuTimeout === '' ? undefined : Number(ncuTimeout),
                    });
                    setNcuResult(res);
                    if (res.job_id) {
                      setJobId(res.job_id);
                      setJobStatus({ status: 'running', job_id: res.job_id });
                      if (runAsync) setIsPolling(true); // Start auto-polling
                    }
                  } catch (e: any) {
                    setNcuResult({ error: e.message });
                  } finally {
                    setNcuLoading(false);
                  }
                }}
              >
                {ncuLoading ? 'Running...' : 'Run Nsight Compute'}
              </button>
              <ResultPanel title="Nsight Compute Result" content={ncuResult} />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-xs text-white/60">torch.profiler script</label>
              <input
                className="w-full rounded-lg bg-white/10 border border-white/10 px-3 py-2 text-sm text-white"
                value={torchScript}
                onChange={(e) => setTorchScript(e.target.value)}
              />
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-xs text-white/60">Args (optional)</label>
                  <input
                    className="w-full rounded-lg bg-white/10 border border-white/10 px-3 py-2 text-sm text-white"
                    value={torchArgs}
                    onChange={(e) => setTorchArgs(e.target.value)}
                    placeholder="--batch 8 --steps 5"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-white/60">NVTX label</label>
                  <input
                    className="w-full rounded-lg bg-white/10 border border-white/10 px-3 py-2 text-sm text-white"
                    value={torchNvtxLabel}
                    onChange={(e) => setTorchNvtxLabel(e.target.value)}
                  />
                </div>
              </div>
              <div className="flex flex-wrap gap-3 text-xs text-white/70 items-center">
                <label className="flex items-center gap-2">
                  Mode
                  <select
                    className="bg-white/10 border border-white/10 rounded-lg px-2 py-1 text-white"
                    value={torchMode}
                    onChange={(e) => setTorchMode(e.target.value)}
                  >
                    <option value="full">full</option>
                    <option value="memory">memory</option>
                    <option value="flops">flops</option>
                    <option value="modules">modules</option>
                    <option value="blackwell">blackwell</option>
                  </select>
                </label>
                <label className="flex items-center gap-2">
                  <input type="checkbox" className="accent-accent-primary" checked={torchForceLineinfo} onChange={(e) => setTorchForceLineinfo(e.target.checked)} />
                  Force lineinfo
                </label>
                <label className="flex items-center gap-2">
                  <input type="checkbox" className="accent-accent-primary" checked={torchUseNvtx} onChange={(e) => setTorchUseNvtx(e.target.checked)} />
                  NVTX range
                </label>
                <label className="flex items-center gap-2">
                  Timeout (s)
                  <input
                    type="number"
                    className="bg-white/10 border border-white/10 rounded-lg px-2 py-1 w-20 text-white"
                    value={torchTimeout}
                    onChange={(e) => setTorchTimeout(e.target.value === '' ? '' : Number(e.target.value))}
                  />
                </label>
              </div>
              <button
                className="rounded-lg bg-accent-secondary/20 px-3 py-2 text-sm text-accent-secondary hover:bg-accent-secondary/30 transition-colors"
                onClick={async () => {
                  if (!torchScript.trim()) {
                    setTorchResult({ error: 'Script is required' });
                    return;
                  }
                  setTorchLoading(true);
                  setTorchResult(null);
                  try {
                    const res = await startTorchProfilerCapture({
                      script: torchScript,
                      mode: torchMode,
                      async: runAsync,
                      nvtx_label: torchNvtxLabel,
                      force_lineinfo: torchForceLineinfo,
                      use_nvtx: torchUseNvtx,
                      timeout_seconds: torchTimeout === '' ? undefined : Number(torchTimeout),
                      script_args: torchArgs.trim() ? torchArgs.trim().match(/\S+/g) || [] : undefined,
                    });
                    setTorchResult(res);
                    if ((res as any)?.job_id) setJobId((res as any).job_id);
                  } catch (e: any) {
                    setTorchResult({ error: e.message });
                  } finally {
                    setTorchLoading(false);
                  }
                }}
              >
                {torchLoading ? 'Running...' : 'Run torch.profiler'}
              </button>
              <ResultPanel title="torch.profiler Result" content={torchResult} />
            </div>

            <div className="space-y-2">
              <label className="text-xs text-white/60">HTA (nsys + HTAAnalyzer) command</label>
              <input
                className="w-full rounded-lg bg-white/10 border border-white/10 px-3 py-2 text-sm text-white"
                value={htaCommand}
                onChange={(e) => setHtaCommand(e.target.value)}
              />
              <div className="flex flex-wrap gap-3 text-xs text-white/70 items-center">
                <label className="flex items-center gap-2">
                  Preset
                  <select
                    className="bg-white/10 border border-white/10 rounded-lg px-2 py-1 text-white"
                    value={htaPreset}
                    onChange={(e) => setHtaPreset(e.target.value as 'full' | 'light')}
                  >
                    <option value="full">full</option>
                    <option value="light">light</option>
                  </select>
                </label>
                <label className="flex items-center gap-2">
                  <input type="checkbox" className="accent-accent-primary" checked={htaForceLineinfo} onChange={(e) => setHtaForceLineinfo(e.target.checked)} />
                  Force lineinfo
                </label>
                <label className="flex items-center gap-2">
                  Timeout (s)
                  <input
                    type="number"
                    className="bg-white/10 border border-white/10 rounded-lg px-2 py-1 w-20 text-white"
                    value={htaTimeout}
                    onChange={(e) => setHtaTimeout(e.target.value === '' ? '' : Number(e.target.value))}
                  />
                </label>
              </div>
              <button
                className="rounded-lg bg-accent-info/20 px-3 py-2 text-sm text-accent-info hover:bg-accent-info/30 transition-colors"
                onClick={async () => {
                  if (!htaCommand.trim()) {
                    setHtaResult({ error: 'Command is required' });
                    return;
                  }
                  setHtaLoading(true);
                  setHtaResult(null);
                  try {
                    const res = await startHTACapture({
                      command: htaCommand,
                      preset: htaPreset,
                      async: runAsync,
                      force_lineinfo: htaForceLineinfo,
                      timeout_seconds: htaTimeout === '' ? undefined : Number(htaTimeout),
                    });
                    setHtaResult(res);
                    if ((res as any)?.job_id) setJobId((res as any).job_id);
                  } catch (e: any) {
                    setHtaResult({ error: e.message });
                  } finally {
                    setHtaLoading(false);
                  }
                }}
              >
                {htaLoading ? 'Running...' : 'Run HTA capture'}
              </button>
              <ResultPanel title="HTA Capture Result" content={htaResult} />
            </div>
          </div>

          <div className="flex flex-wrap gap-3 items-center">
            <input
              className="rounded-lg bg-white/10 border border-white/10 px-3 py-2 text-sm text-white w-72"
              value={jobId}
              onChange={(e) => setJobId(e.target.value)}
              placeholder="Job ID (queue runs)"
            />
            <button
              className="rounded-lg bg-accent-info/20 px-3 py-2 text-sm text-accent-info hover:bg-accent-info/30 transition-colors"
              onClick={async () => {
                if (!jobId) return;
                try {
                  const res = await fetchNsightJobStatus(jobId);
                  setJobStatus(res);
                } catch (e: any) {
                  setJobStatus({ error: e.message });
                }
              }}
            >
              Check Job Status
            </button>
            <button
              className="rounded-lg bg-accent-secondary/20 px-3 py-2 text-sm text-accent-secondary hover:bg-accent-secondary/30 transition-colors"
              onClick={async () => {
                if (!jobId) return;
                try {
                  const res = await fetchMcpJobStatus(jobId);
                  setJobStatus(res);
                } catch (e: any) {
                  setJobStatus({ error: e.message });
                }
              }}
            >
              Check MCP Job
            </button>
            {isPolling && (
              <span className="text-xs text-accent-info animate-pulse">
                ⏳ Auto-polling every 3s...
              </span>
            )}
          </div>
          <ResultPanel 
            title={`Job Status${isPolling ? ' (polling...)' : ''}`} 
            content={jobStatus}
          />
          {jobStatus?.status === 'completed' && (
            <div className="mt-2 p-3 bg-accent-success/10 border border-accent-success/30 rounded-lg">
              <div className="text-sm text-accent-success font-medium">✅ Job completed!</div>
              <div className="text-xs text-white/60 mt-1">
                Use the result or run <code className="bg-white/10 px-1 rounded">aisp bench triage</code> for recommendations.
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-accent-secondary" />
            <h3 className="font-medium text-white">AI Kernel Assistant</h3>
          </div>
          {toolError && (
            <div className="text-sm text-accent-warning">{toolError}</div>
          )}
        </div>
        <div className="card-body space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="space-y-2">
              <label className="text-sm text-white/60" htmlFor="kernelCode">
                Kernel or log snippet
              </label>
              <textarea
                id="kernelCode"
                value={kernelCode}
                onChange={(e) => setKernelCode(e.target.value)}
                className="w-full h-32 bg-white/5 border border-white/10 rounded-lg p-3 text-sm text-white"
                placeholder="Paste PTX/kernel code or profile excerpt..."
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm text-white/60" htmlFor="kernelGoal">
                Goal
              </label>
              <textarea
                id="kernelGoal"
                value={kernelGoal}
                onChange={(e) => setKernelGoal(e.target.value)}
                className="w-full h-32 bg-white/5 border border-white/10 rounded-lg p-3 text-sm text-white"
              />
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={runKernelAnalysis}
              disabled={toolBusy === 'analyze'}
              className="px-3 py-2 rounded-lg bg-accent-primary/20 text-accent-primary text-sm disabled:opacity-50"
            >
              {toolBusy === 'analyze' ? 'Analyzing…' : 'Analyze Kernel'}
            </button>
            <button
              onClick={runPatch}
              disabled={toolBusy === 'patch'}
              className="px-3 py-2 rounded-lg bg-accent-secondary/20 text-accent-secondary text-sm disabled:opacity-50"
            >
              {toolBusy === 'patch' ? 'Drafting…' : 'Generate Patch'}
            </button>
            <div className="flex items-center gap-2">
              <label className="sr-only" htmlFor="profilerQuestion">
                Ask a profiler question
              </label>
              <input
                id="profilerQuestion"
                value={askQuestion}
                onChange={(e) => setAskQuestion(e.target.value)}
                className="px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white"
                placeholder="Ask a profiler question"
              />
              <button
                onClick={runAsk}
                disabled={toolBusy === 'ask'}
                className="px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-sm text-white disabled:opacity-50"
              >
                {toolBusy === 'ask' ? 'Asking…' : 'Ask'}
              </button>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <ResultPanel title="Analysis" content={analysisResult} />
            <ResultPanel title="Patch" content={patchResult} />
            <ResultPanel title="AI Answer" content={askResult} />
          </div>
        </div>
      </div>

      <BottleneckDetectiveCard />

      {/* Kernel table */}
      <div className="card">
        <div className="card-header">
          <h3 className="font-medium text-white">Kernel Details</h3>
        </div>
        {kernels.length === 0 ? (
          <div className="card-body">
            <EmptyState
              title="No kernel samples yet"
              description="Run a profile to see kernels, bottlenecks, and timelines."
              actionLabel="Refresh data"
              onAction={handleRefresh}
            />
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/5">
                  <th className="px-5 py-3 text-left text-xs font-medium text-white/50 uppercase">
                    Kernel
                  </th>
                  <th className="px-5 py-3 text-right text-xs font-medium text-white/50 uppercase">
                    Duration
                  </th>
                  <th className="px-5 py-3 text-right text-xs font-medium text-white/50 uppercase">
                    Calls
                  </th>
                  <th className="px-5 py-3 text-right text-xs font-medium text-white/50 uppercase">
                    % Time
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {kernels.slice(0, 20).map((k, i) => {
                  const duration = k.duration_ms || k.duration || 0;
                  const percentage = totalTime > 0 ? (duration / totalTime) * 100 : 0;
                  return (
                    <tr key={i} className="hover:bg-white/[0.02]">
                      <td className="px-5 py-4 font-mono text-sm text-white">{k.name}</td>
                      <td className="px-5 py-4 text-right font-mono text-sm text-white">
                        {duration.toFixed(2)}ms
                      </td>
                      <td className="px-5 py-4 text-right font-mono text-sm text-white/70">
                        {k.calls || k.count || 1}
                      </td>
                      <td className="px-5 py-4 text-right">
                        <span className="font-bold text-accent-primary">{percentage.toFixed(1)}%</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function ResultPanel({ title, content }: { title: string; content: any }) {
  return (
    <div className="p-3 rounded-lg bg-white/5 border border-white/10 text-sm text-white/80 min-h-[140px]">
      <div className="text-white/60 text-xs mb-2 uppercase tracking-wide">{title}</div>
      {content ? (
        typeof content === 'string' ? (
          <div className="whitespace-pre-wrap break-words">{content}</div>
        ) : (
          <pre className="whitespace-pre-wrap break-words text-xs">
            {JSON.stringify(content, null, 2)}
          </pre>
        )
      ) : (
        <div className="text-white/40">No output yet.</div>
      )}
    </div>
  );
}
