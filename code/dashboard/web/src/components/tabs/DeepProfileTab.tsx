'use client';

import { useEffect, useState } from 'react';
import {
  Microscope,
  Loader2,
  RefreshCw,
  Upload,
  FileText,
  BarChart3,
  Activity,
  Gauge,
  Code,
  ToggleRight,
  ToggleLeft,
  Rocket,
} from 'lucide-react';
import {
  getDeepProfileList,
  getDeepProfileRecommendations,
  getNcuDeepDive,
  getDeepProfileCompare,
  getFlameGraphComparison,
  startNsightSystemsCapture,
  startNsightComputeCapture,
  fetchNsightJobStatus,
  fetchMcpJobStatus,
} from '@/lib/api';
import { EmptyState, ErrorState, LoadingState } from '@/components/DataState';
import { getErrorMessage, useApiQuery } from '@/lib/useApi';
import { FlameGraphComparison } from '@/components/FlameGraphComparison';

type DeepProfileDataset = {
  profiles: any[];
  recommendations: any;
  ncuData: any;
};

export function DeepProfileTab() {
  const [fullTimelineToggle, setFullTimelineToggle] = useState(false);
  const [preset, setPreset] = useState<'light' | 'full'>('full');
  const [selectedChapter, setSelectedChapter] = useState<string | null>(null);
  const [nsysCommand, setNsysCommand] = useState('python -c "print(123)"');
  const [runAsync, setRunAsync] = useState(false);
  const [nsysTimeout, setNsysTimeout] = useState<number | ''>('');
  const [nsysResult, setNsysResult] = useState<any>(null);
  const [nsysLoading, setNsysLoading] = useState(false);

  const [ncuCommand, setNcuCommand] = useState('python -c "print(456)"');
  const [ncuWorkload, setNcuWorkload] = useState('memory_bound');
  const [ncuTimeout, setNcuTimeout] = useState<number | ''>('');
  const [forceLineinfo, setForceLineinfo] = useState(true);
  const [ncuResult, setNcuResult] = useState<any>(null);
  const [ncuLoading, setNcuLoading] = useState(false);

  const [jobId, setJobId] = useState('');
  const [jobStatus, setJobStatus] = useState<any>(null);

  const baseQuery = useApiQuery<DeepProfileDataset>('deep-profile/base', async () => {
    const [profilesData, recsData, ncuDataRes] = await Promise.allSettled([
      getDeepProfileList(),
      getDeepProfileRecommendations(),
      getNcuDeepDive(),
    ]);

    if (profilesData.status === 'rejected') {
      throw profilesData.reason;
    }

    const profList =
      (profilesData.value as any)?.pairs ||
      (profilesData.value as any)?.profiles ||
      profilesData.value ||
      [];

    return {
      profiles: profList,
      recommendations: recsData.status === 'fulfilled' ? recsData.value : null,
      ncuData: ncuDataRes.status === 'fulfilled' ? ncuDataRes.value : null,
    };
  });

  useEffect(() => {
    if (!selectedChapter && baseQuery.data?.profiles?.length) {
      const first = baseQuery.data.profiles[0];
      setSelectedChapter(first?.chapter || first?.name || null);
    }
  }, [baseQuery.data?.profiles, selectedChapter]);

  const compareQuery = useApiQuery(
    selectedChapter ? ['deep-profile/compare', selectedChapter] : null,
    () => getDeepProfileCompare(selectedChapter as string),
    { keepPreviousData: true }
  );

  const flameGraphQuery = useApiQuery(
    selectedChapter ? ['deep-profile/flamegraph', selectedChapter] : null,
    () => getFlameGraphComparison(selectedChapter as string),
    { keepPreviousData: true }
  );

  const profiles = baseQuery.data?.profiles || [];
  const recommendations = baseQuery.data?.recommendations;
  const ncuData = baseQuery.data?.ncuData;
  const compareData = compareQuery.data;
  const compareLoading = compareQuery.isLoading;
  const compareError = compareQuery.error;
  const flameGraphData = flameGraphQuery.data as any;
  const flameGraphLoading = flameGraphQuery.isLoading;

  if (baseQuery.isLoading) {
    return (
      <div className="card">
        <div className="card-body">
          <LoadingState message="Loading deep profile data..." />
        </div>
      </div>
    );
  }

  if (baseQuery.error) {
    return (
      <div className="card">
        <div className="card-body">
          <ErrorState
            message={getErrorMessage(baseQuery.error, 'Failed to load deep profile data')}
            onRetry={() => baseQuery.mutate()}
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
            <Microscope className="w-5 h-5 text-accent-info" />
            <h2 className="text-lg font-semibold text-white">Deep Profile Comparison</h2>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => baseQuery.mutate()}
              className="flex items-center gap-2 rounded-lg p-2 text-white/70 hover:bg-white/5"
              aria-label="Refresh deep profile data"
            >
              <RefreshCw className="w-4 h-4" />
              {baseQuery.isValidating && <span className="text-xs">Refreshing…</span>}
            </button>
            <button
              className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 text-xs text-white"
              onClick={() => setFullTimelineToggle((v) => !v)}
              title="Use this when capturing new traces for richer source attribution."
              aria-pressed={fullTimelineToggle}
            >
              {fullTimelineToggle ? (
                <ToggleRight className="w-4 h-4 text-accent-success" />
              ) : (
                <ToggleLeft className="w-4 h-4 text-white/40" />
              )}
              Full timeline capture (nsys)
            </button>
            <label className="sr-only" htmlFor="nsysPreset">
              NSYS preset
            </label>
            <select
              id="nsysPreset"
              value={preset}
              onChange={(e) => setPreset(e.target.value as 'light' | 'full')}
              className="px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-xs text-white"
              title="NSYS preset for capture tooling"
            >
              <option value="full">Full (default: cuda-hw + cuBLAS/cuSOLVER/cuSPARSE/cuDNN + forks)</option>
              <option value="light">Light (smaller/faster)</option>
            </select>
            {preset === 'full' && (
              <span className="text-[11px] text-accent-warning">
                Full mode runs slower and produces larger traces.
              </span>
            )}
            <span className="text-[11px] text-white/60">
              Active preset: <span className="text-accent-primary">{preset}</span>
            </span>
            <span className="text-[11px] text-white/60">
              Tip: set <code className="text-accent-primary">TMPDIR</code> to a path with 200MB+ free before NSYS capture.
            </span>
          </div>
        </div>
        <div className="card-body">
          <p className="text-white/60 mb-4">
            Compare nsys/ncu profiles to identify optimization opportunities
          </p>
          <div className="flex gap-4 flex-wrap">
            <button className="flex items-center gap-2 px-4 py-2 bg-white/5 border border-white/10 rounded-lg hover:bg-white/10">
              <Upload className="w-4 h-4" />
              Upload Baseline Profile
            </button>
            <button className="flex items-center gap-2 px-4 py-2 bg-white/5 border border-white/10 rounded-lg hover:bg-white/10">
              <Upload className="w-4 h-4" />
              Upload Optimized Profile
            </button>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header flex items-center gap-2">
          <Rocket className="w-4 h-4 text-accent-primary" />
          <h3 className="font-medium text-white">Capture New Profiles (nsys/ncu)</h3>
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
                  <input type="checkbox" className="accent-accent-secondary" checked={runAsync} onChange={(e) => setRunAsync(e.target.checked)} />
                  Async
                </label>
                <label className="flex items-center gap-2">
                  <input type="checkbox" className="accent-accent-primary" checked={forceLineinfo} onChange={(e) => setForceLineinfo(e.target.checked)} />
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
                      preset: preset,
                      full_timeline: fullTimelineToggle || preset === 'full',
                      async: runAsync,
                      force_lineinfo: forceLineinfo,
                      timeout_seconds: nsysTimeout === '' ? undefined : Number(nsysTimeout),
                    });
                    setNsysResult(res);
                    if (res.job_id) setJobId(res.job_id);
                  } catch (e: any) {
                    setNsysResult({ error: e.message });
                  } finally {
                    setNsysLoading(false);
                  }
                }}
              >
                {nsysLoading ? 'Running...' : 'Run Nsight Systems'}
              </button>
              <div className="text-xs text-white/50">Uses preset {preset} (full timeline toggle respected).</div>
              <pre className="bg-white/5 border border-white/10 rounded-lg p-2 text-xs text-white/80 whitespace-pre-wrap">
                {nsysResult ? JSON.stringify(nsysResult, null, 2) : 'No result yet.'}
              </pre>
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
                  <input type="checkbox" className="accent-accent-secondary" checked={runAsync} onChange={(e) => setRunAsync(e.target.checked)} />
                  Async
                </label>
                <label className="flex items-center gap-2">
                  <input type="checkbox" className="accent-accent-primary" checked={forceLineinfo} onChange={(e) => setForceLineinfo(e.target.checked)} />
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
                      force_lineinfo: forceLineinfo,
                      timeout_seconds: ncuTimeout === '' ? undefined : Number(ncuTimeout),
                    });
                    setNcuResult(res);
                    if (res.job_id) setJobId(res.job_id);
                  } catch (e: any) {
                    setNcuResult({ error: e.message });
                  } finally {
                    setNcuLoading(false);
                  }
                }}
              >
                {ncuLoading ? 'Running...' : 'Run Nsight Compute'}
              </button>
              <pre className="bg-white/5 border border-white/10 rounded-lg p-2 text-xs text-white/80 whitespace-pre-wrap">
                {ncuResult ? JSON.stringify(ncuResult, null, 2) : 'No result yet.'}
              </pre>
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
          </div>
          <pre className="bg-white/5 border border-white/10 rounded-lg p-2 text-xs text-white/80 whitespace-pre-wrap">
            {jobStatus ? JSON.stringify(jobStatus, null, 2) : 'No job queried yet.'}
          </pre>
        </div>
      </div>

      {/* Profiles list */}
      {profiles.length > 0 ? (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">Available Profiles</h3>
          </div>
          <div className="card-body">
            <div className="space-y-3">
              {profiles.map((profile, i) => {
                const active = selectedChapter === (profile?.chapter || profile?.name);
                return (
                  <button
                    key={i}
                    className={`flex w-full items-center justify-between rounded-lg p-4 text-left transition-colors ${
                      active ? 'bg-accent-info/10 border border-accent-info/40' : 'bg-white/5 border border-white/10 hover:bg-white/10'
                    }`}
                    type="button"
                    onClick={() => setSelectedChapter(profile?.chapter || profile?.name || null)}
                  >
                    <div className="flex items-center gap-3">
                      <FileText className="w-5 h-5 text-accent-info" />
                      <div>
                        <div className="font-medium text-white">
                          {profile.chapter || profile.name || profile.filename}
                        </div>
                        <div className="text-sm text-white/50">
                          {profile.path || profile.date || profile.timestamp}
                        </div>
                      </div>
                    </div>
                    <span className="text-sm text-white/40">{profile.type || 'nsys'}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      ) : (
        <EmptyState
          title="No profiles found"
          description="Run profiling with nsys/ncu to populate this list."
          actionLabel="Refresh list"
          onAction={() => baseQuery.mutate()}
        />
      )}

      {/* Flame Graph Comparison - NEW! */}
      {selectedChapter && (
        <FlameGraphComparison data={flameGraphData} isLoading={flameGraphLoading} />
      )}

      {/* Recommendations */}
      {recommendations && (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">Profile Recommendations</h3>
          </div>
          <div className="card-body space-y-3">
            {(recommendations.recommendations || recommendations.items || []).map((rec: any, i: number) => (
              <div
                key={i}
                className="p-4 bg-gradient-to-r from-accent-info/10 to-transparent rounded-lg border-l-2 border-accent-info"
              >
                <h4 className="font-medium text-white mb-1">{rec.title || rec.name}</h4>
                <p className="text-sm text-white/70">{rec.description || rec.message}</p>
                {rec.impact && (
                  <div className="mt-2 text-sm text-accent-success">
                    Estimated impact: {rec.impact}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* NCU Deep Dive */}
      {ncuData && (
        <div className="space-y-4">
          <div className="card">
            <div className="card-header">
              <div className="flex items-center gap-2">
                <Gauge className="w-5 h-5 text-accent-primary" />
                <h3 className="font-medium text-white">NCU Deep Dive</h3>
              </div>
            </div>
            <div className="card-body">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {Object.entries(ncuData.metrics || ncuData || {}).slice(0, 6).map(([key, value]: [string, any], i) => (
                  <div key={i} className="p-4 bg-white/5 rounded-lg">
                    <div className="text-sm text-white/50 mb-1">{key}</div>
                    <div className="text-xl font-bold text-accent-primary">
                      {typeof value === 'number' ? value.toFixed(2) : value}
                    </div>
                  </div>
                ))}
                {Object.keys(ncuData.metrics || ncuData || {}).length === 0 && (
                  <EmptyState
                    title="No deep dive metrics"
                    description="Run Nsight Compute with metrics enabled to populate this section."
                    className="md:col-span-3"
                  />
                )}
              </div>
            </div>
          </div>

          {ncuData.memory_analysis && (
            <div className="card">
              <div className="card-header">
                <div className="flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-accent-warning" />
                  <h3 className="font-medium text-white">Memory Analysis</h3>
                </div>
              </div>
              <div className="card-body grid grid-cols-1 md:grid-cols-3 gap-3 text-sm text-white/80">
                {['hbm_achieved_gbs', 'hbm_peak_gbs', 'hbm_utilization_pct', 'l2_hit_rate_pct', 'l1_hit_rate_pct', 'shared_mem_utilization_pct'].map((k) => (
                  <div key={k} className="p-3 rounded-lg bg-white/5 border border-white/10">
                    <div className="text-white/50 text-xs uppercase">{k.replace(/_/g, ' ')}</div>
                    <div className="text-lg font-bold text-accent-primary">{ncuData.memory_analysis[k]}</div>
                  </div>
                ))}
                {ncuData.memory_analysis.recommendations && (
                  <div className="md:col-span-3 space-y-1">
                    {ncuData.memory_analysis.recommendations.map((r: string, i: number) => (
                      <div key={i} className="text-xs text-white/70">• {r}</div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {ncuData.occupancy_analysis && (
            <div className="card">
              <div className="card-header">
                <div className="flex items-center gap-2">
                  <Activity className="w-5 h-5 text-accent-success" />
                  <h3 className="font-medium text-white">Occupancy</h3>
                </div>
              </div>
              <div className="card-body grid grid-cols-1 md:grid-cols-3 gap-3 text-sm text-white/80">
                <div className="p-3 rounded-lg bg-white/5 border border-white/10">
                  <div className="text-white/50 text-xs uppercase">Theoretical</div>
                  <div className="text-lg font-bold text-accent-success">{ncuData.occupancy_analysis.theoretical_max}%</div>
                </div>
                <div className="p-3 rounded-lg bg-white/5 border border-white/10">
                  <div className="text-white/50 text-xs uppercase">Achieved</div>
                  <div className="text-lg font-bold text-accent-primary">{ncuData.occupancy_analysis.achieved_avg}%</div>
                </div>
                <div className="p-3 rounded-lg bg-white/5 border border-white/10">
                  <div className="text-white/50 text-xs uppercase">Limiter</div>
                  <div className="text-lg font-bold text-accent-warning">{ncuData.occupancy_analysis.limiting_factor}</div>
                </div>
                {ncuData.occupancy_analysis.recommendations && (
                  <div className="md:col-span-3 space-y-1">
                    {ncuData.occupancy_analysis.recommendations.map((r: string, i: number) => (
                      <div key={i} className="text-xs text-white/70">• {r}</div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {ncuData.warp_stalls && (
            <div className="card">
              <div className="card-header">
                <div className="flex items-center gap-2">
                  <Microscope className="w-5 h-5 text-accent-info" />
                  <h3 className="font-medium text-white">Warp Stalls</h3>
                </div>
              </div>
              <div className="card-body space-y-2">
                {(ncuData.warp_stalls.categories || []).map((c: any, i: number) => (
                  <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-white/5 border border-white/10">
                    <div>
                      <div className="text-white font-semibold">{c.name}</div>
                      <div className="text-xs text-white/50">{c.description}</div>
                    </div>
                    <div className="text-accent-primary font-mono">{c.pct}%</div>
                  </div>
                ))}
                {ncuData.warp_stalls.recommendations && (
                  <div className="space-y-1 text-xs text-white/70">
                    {ncuData.warp_stalls.recommendations.map((r: string, i: number) => (
                      <div key={i}>• {r}</div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Source Attribution (NCU/NSYS) */}
      {(compareData?.ncu_comparison?.baseline_sources?.length ||
        compareData?.ncu_comparison?.optimized_sources?.length ||
        compareData?.nsys_comparison?.baseline_sources?.length ||
        compareData?.nsys_comparison?.optimized_sources?.length ||
        ncuData?.source_samples?.length ||
        compareData?.ncu_comparison?.baseline_disassembly?.length ||
        compareData?.ncu_comparison?.optimized_disassembly?.length ||
        ncuData?.disassembly?.length) && (
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <Code className="w-5 h-5 text-accent-secondary" />
              <h3 className="font-medium text-white">
                Source Attribution (lineinfo)
              </h3>
            </div>
            {compareLoading && (
              <span className="text-xs text-white/50 flex items-center gap-2">
                <Loader2 className="w-3 h-3 animate-spin" /> Loading comparison…
              </span>
            )}
            {compareError && !compareLoading && (
              <span className="text-xs text-accent-warning">
                {getErrorMessage(compareError, 'Comparison failed')}
              </span>
            )}
          </div>
          <div className="card-body grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-white/80">
            <SourceList
              title="NCU Baseline"
              entries={compareData?.ncu_comparison?.baseline_sources || ncuData?.source_samples || []}
            />
            <SourceList
              title="NCU Optimized"
              entries={compareData?.ncu_comparison?.optimized_sources || []}
            />
            <SourceList
              title="NSYS Baseline"
              entries={compareData?.nsys_comparison?.baseline_sources || []}
            />
            <SourceList
              title="NSYS Optimized"
              entries={compareData?.nsys_comparison?.optimized_sources || []}
            />
            <DisassemblyList
              title="NCU Baseline SASS"
              lines={compareData?.ncu_comparison?.baseline_disassembly || ncuData?.disassembly || []}
            />
            <DisassemblyList
              title="NCU Optimized SASS"
              lines={compareData?.ncu_comparison?.optimized_disassembly || []}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function SourceList({ title, entries }: { title: string; entries: any[] }) {
  if (!entries || entries.length === 0) {
    return (
      <div className="p-3 rounded-lg bg-white/5 border border-white/10">
        <div className="text-white/60 text-xs uppercase">{title}</div>
        <div className="text-white/40 text-xs">No source records found.</div>
      </div>
    );
  }
  return (
    <div className="p-3 rounded-lg bg-white/5 border border-white/10 space-y-2">
      <div className="text-white/60 text-xs uppercase">{title}</div>
      {entries.slice(0, 6).map((e, idx) => (
        <div key={idx} className="text-xs text-white/80">
          <div className="font-semibold text-white truncate">{e.kernel || e.name || e.cat || 'event'}</div>
          <div className="text-white/60 truncate">
            {e.file || e.source || 'unknown'}{e.line ? `:${e.line}` : ''}
          </div>
          {e.address && <div className="text-white/40">addr {e.address}</div>}
          {e.stall && <div className="text-accent-warning">stall {e.stall}</div>}
        </div>
      ))}
    </div>
  );
}

function DisassemblyList({ title, lines }: { title: string; lines: string[] }) {
  if (!lines || lines.length === 0) {
    return (
      <div className="p-3 rounded-lg bg-white/5 border border-white/10">
        <div className="text-white/60 text-xs uppercase">{title}</div>
        <div className="text-white/40 text-xs">No disassembly available.</div>
      </div>
    );
  }
  return (
    <div className="p-3 rounded-lg bg-white/5 border border-white/10 space-y-1">
      <div className="text-white/60 text-xs uppercase">{title}</div>
      <pre className="text-[11px] leading-4 text-white/80 whitespace-pre-wrap break-words max-h-64 overflow-auto">
        {lines.slice(0, 60).join('\n')}
      </pre>
    </div>
  );
}
