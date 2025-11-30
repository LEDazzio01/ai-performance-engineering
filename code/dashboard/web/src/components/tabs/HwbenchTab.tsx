'use client';

import { useState } from 'react';
import useSWRMutation from 'swr/mutation';
import { Timer, Cpu, HardDrive, Network, Zap, RefreshCw, Download, Rocket } from 'lucide-react';
import { startNsightComputeCapture, startNsightSystemsCapture, fetchNsightJobStatus, fetchMcpJobStatus } from '@/lib/api';

type Result = Record<string, any> | null;

async function fetchJson(_: string, { arg }: { arg: { url: string } }): Promise<any> {
  const res = await fetch(arg.url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return res.json();
}

function ResultCard({ title, result, loading, error }: { title: string; result: Result; loading: boolean; error?: string | null }) {
  return (
    <div className="rounded-xl border border-white/10 bg-white/5 backdrop-blur-sm p-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-white">{title}</h3>
        {loading && <span className="text-xs text-accent-primary animate-pulse">Loading…</span>}
      </div>
      {error ? (
        <pre className="text-xs text-accent-danger whitespace-pre-wrap break-all">{error}</pre>
      ) : result ? (
        <pre className="text-xs text-white/80 whitespace-pre-wrap break-all max-h-48 overflow-auto">{JSON.stringify(result, null, 2)}</pre>
      ) : (
        <p className="text-xs text-white/40">No data yet.</p>
      )}
    </div>
  );
}

function MetricsTable({ result }: { result: any }) {
  if (!result || typeof result !== 'object') return null;
  const metrics = Array.isArray(result.metrics) ? result.metrics : Array.isArray(result) ? result : null;
  if (!metrics) return null;
  return (
    <div className="overflow-auto rounded-lg border border-white/10">
      <table className="min-w-full text-xs text-left text-white/80">
        <thead className="bg-white/5 text-white/60">
          <tr>
            <th className="px-3 py-2">Name</th>
            <th className="px-3 py-2">Baseline</th>
            <th className="px-3 py-2">Optimized</th>
            <th className="px-3 py-2">Delta</th>
          </tr>
        </thead>
        <tbody>
          {metrics.map((m: any, idx: number) => (
            <tr key={idx} className="border-t border-white/5 hover:bg-white/[0.02]">
              <td className="px-3 py-2 font-medium text-white">{m.name || m.metric || m.section || 'metric'}</td>
              <td className="px-3 py-2">{m.baseline ?? m.base ?? ''}</td>
              <td className="px-3 py-2">{m.optimized ?? m.opt ?? ''}</td>
              <td className="px-3 py-2 text-accent-primary">{m.delta ?? ''}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function HwbenchTab() {
  const disk = useSWRMutation('disk', fetchJson);
  const [diskSize, setDiskSize] = useState(256);
  const [diskBlock, setDiskBlock] = useState(1024);

  const pcie = useSWRMutation('pcie', fetchJson);
  const [pcieSize, setPcieSize] = useState(256);
  const [pcieIters, setPcieIters] = useState(10);

  const mem = useSWRMutation('mem', fetchJson);
  const [memSize, setMemSize] = useState(256);
  const [memStride, setMemStride] = useState(128);

  const tensor = useSWRMutation('tensor', fetchJson);
  const [tensorSize, setTensorSize] = useState(4096);
  const [tensorPrecision, setTensorPrecision] = useState('fp16');

  const sfu = useSWRMutation('sfu', fetchJson);
  const [sfuElems, setSfuElems] = useState(64 * 1024 * 1024);

  const loop = useSWRMutation('loop', fetchJson);
  const [loopSize, setLoopSize] = useState(64);
  const [loopPort, setLoopPort] = useState(50007);

  const exportCsv = useSWRMutation('exportCsv', fetchJson);
  const exportHtml = useSWRMutation('exportHtml', fetchJson);
  const [pdfStatus, setPdfStatus] = useState<string | null>(null);

  const nsightAvail = useSWRMutation('nsightAvail', fetchJson);
  const compareNsys = useSWRMutation('compareNsys', fetchJson);
  const compareNcu = useSWRMutation('compareNcu', fetchJson);
  const [compareDir, setCompareDir] = useState<string>('artifacts/mcp-profiles');
  const [detailedCsv, setDetailedCsv] = useState<boolean>(false);
  const [precheckOnly, setPrecheckOnly] = useState(false);
  const [dryRun, setDryRun] = useState(false);
  const [timeoutSeconds, setTimeoutSeconds] = useState<number | ''>('');

  const [nsysCommand, setNsysCommand] = useState('python -c "print(123)"');
  const [nsysPreset, setNsysPreset] = useState<'light' | 'full'>('full');
  const [nsysQueue, setNsysQueue] = useState(false);
  const [nsysFullTimeline, setNsysFullTimeline] = useState(false);
  const [nsysResult, setNsysResult] = useState<Result>(null);
  const [nsysLoading, setNsysLoading] = useState(false);
  const [nsysForceLineinfo, setNsysForceLineinfo] = useState(true);

  const [ncuCommand, setNcuCommand] = useState('python -c "print(456)"');
  const [ncuWorkload, setNcuWorkload] = useState('memory_bound');
  const [ncuQueue, setNcuQueue] = useState(false);
  const [ncuResult, setNcuResult] = useState<Result>(null);
  const [ncuLoading, setNcuLoading] = useState(false);
  const [ncuForceLineinfo, setNcuForceLineinfo] = useState(true);

  const [jobId, setJobId] = useState('');
  const [jobStatus, setJobStatus] = useState<Result>(null);

  const flagQuery = `&precheck_only=${precheckOnly}&dry_run=${dryRun}${timeoutSeconds !== '' ? `&timeout_seconds=${timeoutSeconds}` : ''}`;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Timer className="w-5 h-5 text-accent-warning" />
            <h2 className="text-lg font-semibold text-white">Hardware Benchmarks</h2>
          </div>
        </div>
        <div className="card-body">
          <p className="text-sm text-white/60">Quick, lightweight diagnostics for disk, PCIe, memory, tensor cores, SFU, and network loopback.</p>
          <p className="text-xs text-white/60">
            Use precheck to validate paths/availability without running; dry run to see planned args; set a timeout to avoid runaway tests.
          </p>
        </div>
      </div>

      <div className="card">
        <div className="card-body flex flex-wrap gap-4 items-center">
          <label className="flex items-center gap-2 text-sm text-white/70">
            <input type="checkbox" className="accent-accent-primary" checked={precheckOnly} onChange={(e) => setPrecheckOnly(e.target.checked)} />
            Precheck only
          </label>
          <label className="flex items-center gap-2 text-sm text-white/70">
            <input type="checkbox" className="accent-accent-secondary" checked={dryRun} onChange={(e) => setDryRun(e.target.checked)} />
            Dry run (describe only)
          </label>
          <label className="flex items-center gap-2 text-sm text-white/70">
            Timeout (s)
            <input
              type="number"
              min={0}
              placeholder="120"
              className="bg-white/10 border border-white/10 px-2 py-1 rounded-lg w-24 text-white"
              value={timeoutSeconds}
              onChange={(e) => setTimeoutSeconds(e.target.value === '' ? '' : Number(e.target.value))}
            />
          </label>
          <span className="text-xs text-white/50">Applies to all hardware benchmark calls below.</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <HardDrive className="w-4 h-4 text-accent-info" />
              <h3 className="font-medium text-white">Disk I/O</h3>
            </div>
          </div>
          <div className="card-body space-y-3">
            <div className="flex flex-wrap gap-2 items-center">
              <button
                className="rounded-lg bg-accent-info/20 px-3 py-2 text-sm text-accent-info hover:bg-accent-info/30 transition-colors"
                onClick={() => disk.trigger({ url: `/api/hwbench/disk?file_size_mb=${diskSize}&block_size_kb=${diskBlock}${flagQuery}` })}
              >
                Run Disk I/O
              </button>
              <label className="text-xs text-white/60 flex items-center gap-1">
                Size MB <input className="bg-white/10 border border-white/10 px-2 py-1 rounded-lg w-16 text-white" type="number" value={diskSize} onChange={(e) => setDiskSize(Number(e.target.value))} />
              </label>
              <label className="text-xs text-white/60 flex items-center gap-1">
                Block KB <input className="bg-white/10 border border-white/10 px-2 py-1 rounded-lg w-16 text-white" type="number" value={diskBlock} onChange={(e) => setDiskBlock(Number(e.target.value))} />
              </label>
            </div>
            <ResultCard title="Disk I/O" result={(disk.data as Result) || null} loading={disk.isMutating} error={(disk.error as any)?.message || null} />
            <div className="flex items-center gap-2 pt-2 border-t border-white/5">
              <button
                className="rounded-lg bg-accent-secondary/20 px-3 py-2 text-sm text-accent-secondary hover:bg-accent-secondary/30 transition-colors flex items-center gap-2"
                onClick={async () => {
                  try {
                    setPdfStatus('Downloading...');
                    const res = await fetch('/api/export/pdf');
                    if (!res.ok) throw new Error(`HTTP ${res.status}`);
                    const blob = await res.blob();
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'benchmark_report.pdf';
                    a.click();
                    URL.revokeObjectURL(url);
                    setPdfStatus('Downloaded benchmark_report.pdf');
                  } catch (e: any) {
                    setPdfStatus(`Error: ${e.message}`);
                  }
                }}
              >
                <Download className="w-4 h-4" />
                Export PDF
              </button>
              <span className="text-xs text-white/40">{pdfStatus || 'PDF downloads directly.'}</span>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-accent-primary" />
              <h3 className="font-medium text-white">PCIe Bandwidth</h3>
            </div>
          </div>
          <div className="card-body space-y-3">
            <div className="flex gap-2 items-center">
              <button
                className="rounded-lg bg-accent-primary/20 px-3 py-2 text-sm text-accent-primary hover:bg-accent-primary/30 transition-colors"
                onClick={() => pcie.trigger({ url: `/api/hwbench/pcie?size_mb=${pcieSize}&iters=${pcieIters}${flagQuery}` })}
              >
                Run PCIe H2D/D2H
              </button>
              <label className="text-xs text-white/60 flex items-center gap-1">
                Size MB <input className="bg-white/10 border border-white/10 px-2 py-1 rounded-lg w-16 text-white" type="number" value={pcieSize} onChange={(e) => setPcieSize(Number(e.target.value))} />
              </label>
              <label className="text-xs text-white/60 flex items-center gap-1">
                Iters <input className="bg-white/10 border border-white/10 px-2 py-1 rounded-lg w-14 text-white" type="number" value={pcieIters} onChange={(e) => setPcieIters(Number(e.target.value))} />
              </label>
            </div>
            <ResultCard title="PCIe" result={(pcie.data as Result) || null} loading={pcie.isMutating} error={(pcie.error as any)?.message || null} />
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <HardDrive className="w-4 h-4 text-accent-success" />
              <h3 className="font-medium text-white">Memory Hierarchy</h3>
            </div>
          </div>
          <div className="card-body space-y-3">
            <div className="flex gap-2 items-center">
              <button
                className="rounded-lg bg-accent-success/20 px-3 py-2 text-sm text-accent-success hover:bg-accent-success/30 transition-colors"
                onClick={() => mem.trigger({ url: `/api/hwbench/mem?size_mb=${memSize}&stride=${memStride}${flagQuery}` })}
              >
                Run Memory Stride
              </button>
              <label className="text-xs text-white/60 flex items-center gap-1">
                Size MB <input className="bg-white/10 border border-white/10 px-2 py-1 rounded-lg w-16 text-white" type="number" value={memSize} onChange={(e) => setMemSize(Number(e.target.value))} />
              </label>
              <label className="text-xs text-white/60 flex items-center gap-1">
                Stride <input className="bg-white/10 border border-white/10 px-2 py-1 rounded-lg w-16 text-white" type="number" value={memStride} onChange={(e) => setMemStride(Number(e.target.value))} />
              </label>
            </div>
            <ResultCard title="Memory Hierarchy" result={(mem.data as Result) || null} loading={mem.isMutating} error={(mem.error as any)?.message || null} />
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-accent-warning" />
              <h3 className="font-medium text-white">Tensor Core</h3>
            </div>
          </div>
          <div className="card-body space-y-3">
            <div className="flex gap-2 items-center">
              <button
                className="rounded-lg bg-accent-warning/20 px-3 py-2 text-sm text-accent-warning hover:bg-accent-warning/30 transition-colors"
                onClick={() => tensor.trigger({ url: `/api/hwbench/tensor-cores?size=${tensorSize}&precision=${encodeURIComponent(tensorPrecision)}${flagQuery}` })}
              >
                Run Tensor Core
              </button>
              <label className="text-xs text-white/60 flex items-center gap-1">
                Size <input className="bg-white/10 border border-white/10 px-2 py-1 rounded-lg w-16 text-white" type="number" value={tensorSize} onChange={(e) => setTensorSize(Number(e.target.value))} />
              </label>
              <label className="text-xs text-white/60 flex items-center gap-1">
                Precision <input className="bg-white/10 border border-white/10 px-2 py-1 rounded-lg w-20 text-white" type="text" value={tensorPrecision} onChange={(e) => setTensorPrecision(e.target.value)} />
              </label>
            </div>
            <ResultCard title="Tensor Core" result={(tensor.data as Result) || null} loading={tensor.isMutating} error={(tensor.error as any)?.message || null} />
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-accent-tertiary" />
              <h3 className="font-medium text-white">SFU (Special Functions)</h3>
            </div>
          </div>
          <div className="card-body space-y-3">
            <div className="flex gap-2 items-center">
              <button
                className="rounded-lg bg-accent-tertiary/20 px-3 py-2 text-sm text-accent-tertiary hover:bg-accent-tertiary/30 transition-colors"
                onClick={() => sfu.trigger({ url: `/api/hwbench/sfu?elements=${sfuElems}${flagQuery}` })}
              >
                Run SFU
              </button>
              <label className="text-xs text-white/60 flex items-center gap-1">
                Elements <input className="bg-white/10 border border-white/10 px-2 py-1 rounded-lg w-24 text-white" type="number" value={sfuElems} onChange={(e) => setSfuElems(Number(e.target.value))} />
              </label>
            </div>
            <ResultCard title="SFU" result={(sfu.data as Result) || null} loading={sfu.isMutating} error={(sfu.error as any)?.message || null} />
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <Network className="w-4 h-4 text-accent-info" />
              <h3 className="font-medium text-white">Network Loopback</h3>
            </div>
          </div>
          <div className="card-body space-y-3">
            <div className="flex gap-2 items-center">
              <button
                className="rounded-lg bg-accent-info/20 px-3 py-2 text-sm text-accent-info hover:bg-accent-info/30 transition-colors"
                onClick={() => loop.trigger({ url: `/api/hwbench/loopback?size_mb=${loopSize}&port=${loopPort}${flagQuery}` })}
              >
                Run Loopback
              </button>
              <label className="text-xs text-white/60 flex items-center gap-1">
                Size MB <input className="bg-white/10 border border-white/10 px-2 py-1 rounded-lg w-16 text-white" type="number" value={loopSize} onChange={(e) => setLoopSize(Number(e.target.value))} />
              </label>
              <label className="text-xs text-white/60 flex items-center gap-1">
                Port <input className="bg-white/10 border border-white/10 px-2 py-1 rounded-lg w-16 text-white" type="number" value={loopPort} onChange={(e) => setLoopPort(Number(e.target.value))} />
              </label>
            </div>
            <ResultCard title="Loopback" result={(loop.data as Result) || null} loading={loop.isMutating} error={(loop.error as any)?.message || null} />
          </div>
        </div>
      </div>

      {/* Nsight Utilities */}
      <div className="card">
        <div className="card-header">
          <h3 className="font-medium text-white">Nsight Profiling Utilities</h3>
        </div>
        <div className="card-body space-y-4">
          <div className="flex flex-wrap gap-2">
            <button
              className="rounded-lg bg-accent-primary/20 px-3 py-2 text-sm text-accent-primary hover:bg-accent-primary/30 transition-colors"
              onClick={() => nsightAvail.trigger({ url: '/api/nsight/availability' })}
            >
              Check Nsight Availability
            </button>
            <input
              className="rounded-lg bg-white/10 border border-white/10 px-3 py-2 text-sm text-white w-64"
              value={compareDir}
              onChange={(e) => setCompareDir(e.target.value)}
              placeholder="Profiles directory"
            />
            <button
              className="rounded-lg bg-accent-secondary/20 px-3 py-2 text-sm text-accent-secondary hover:bg-accent-secondary/30 transition-colors"
              onClick={() => compareNsys.trigger({ url: `/api/nsight/compare/nsys?dir=${encodeURIComponent(compareDir)}` })}
            >
              Compare Nsight Systems
            </button>
            <button
              className="rounded-lg bg-accent-warning/20 px-3 py-2 text-sm text-accent-warning hover:bg-accent-warning/30 transition-colors"
              onClick={() => compareNcu.trigger({ url: `/api/nsight/compare/ncu?dir=${encodeURIComponent(compareDir)}` })}
            >
              Compare Nsight Compute
            </button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <ResultCard title="Nsight Availability" result={(nsightAvail.data as Result) || null} loading={nsightAvail.isMutating} error={(nsightAvail.error as any)?.message || null} />
            <div className="space-y-2">
              <ResultCard title="Nsight Systems Compare" result={(compareNsys.data as Result) || null} loading={compareNsys.isMutating} error={(compareNsys.error as any)?.message || null} />
              <MetricsTable result={compareNsys.data} />
            </div>
            <div className="space-y-2">
              <ResultCard title="Nsight Compute Compare" result={(compareNcu.data as Result) || null} loading={compareNcu.isMutating} error={(compareNcu.error as any)?.message || null} />
              <MetricsTable result={compareNcu.data} />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="card bg-white/5 border-white/10">
              <div className="card-header flex items-center gap-2">
                <Rocket className="w-4 h-4 text-accent-primary" />
                <span className="text-sm font-semibold text-white">Start Nsight Systems</span>
              </div>
              <div className="card-body space-y-2">
                <input
                  className="w-full rounded-lg bg-white/10 border border-white/10 px-3 py-2 text-sm text-white"
                  value={nsysCommand}
                  onChange={(e) => setNsysCommand(e.target.value)}
                  placeholder="Command (e.g., python train.py)"
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
                   <input type="checkbox" className="accent-accent-secondary" checked={nsysQueue} onChange={(e) => setNsysQueue(e.target.checked)} />
                   Queue only
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
                      value={timeoutSeconds}
                      onChange={(e) => setTimeoutSeconds(e.target.value === '' ? '' : Number(e.target.value))}
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
                      const json = await startNsightSystemsCapture({
                        command: nsysCommand,
                      preset: nsysPreset,
                      full_timeline: nsysFullTimeline,
                      queue_only: nsysQueue,
                      force_lineinfo: nsysForceLineinfo,
                      timeout_seconds: timeoutSeconds === '' ? undefined : Number(timeoutSeconds),
                    });
                      setNsysResult(json);
                      if (json.job_id) setJobId(json.job_id);
                    } catch (e: any) {
                      setNsysResult({ error: e.message });
                    } finally {
                      setNsysLoading(false);
                    }
                  }}
                >
                  {nsysLoading ? 'Running...' : 'Run Nsight Systems'}
                </button>
                <ResultCard title="Nsight Systems Result" result={nsysResult} loading={nsysLoading} />
              </div>
            </div>

            <div className="card bg-white/5 border-white/10">
              <div className="card-header flex items-center gap-2">
                <Rocket className="w-4 h-4 text-accent-warning" />
                <span className="text-sm font-semibold text-white">Start Nsight Compute</span>
              </div>
              <div className="card-body space-y-2">
                <input
                  className="w-full rounded-lg bg-white/10 border border-white/10 px-3 py-2 text-sm text-white"
                  value={ncuCommand}
                  onChange={(e) => setNcuCommand(e.target.value)}
                  placeholder="Command (e.g., python train.py)"
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
                   <input type="checkbox" className="accent-accent-secondary" checked={ncuQueue} onChange={(e) => setNcuQueue(e.target.checked)} />
                   Queue only
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
                      value={timeoutSeconds}
                      onChange={(e) => setTimeoutSeconds(e.target.value === '' ? '' : Number(e.target.value))}
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
                      const json = await startNsightComputeCapture({
                        command: ncuCommand,
                      workload_type: ncuWorkload,
                      queue_only: ncuQueue,
                      force_lineinfo: ncuForceLineinfo,
                      timeout_seconds: timeoutSeconds === '' ? undefined : Number(timeoutSeconds),
                    });
                      setNcuResult(json);
                      if (json.job_id) setJobId(json.job_id);
                    } catch (e: any) {
                      setNcuResult({ error: e.message });
                    } finally {
                      setNcuLoading(false);
                    }
                  }}
                >
                  {ncuLoading ? 'Running...' : 'Run Nsight Compute'}
                </button>
                <ResultCard title="Nsight Compute Result" result={ncuResult} loading={ncuLoading} />
              </div>
            </div>
          </div>

          <div className="card bg-white/5 border-white/10">
            <div className="card-body flex flex-wrap items-center gap-3">
              <input
                className="rounded-lg bg-white/10 border border-white/10 px-3 py-2 text-sm text-white w-64"
                value={jobId}
                onChange={(e) => setJobId(e.target.value)}
                placeholder="Job ID (queue runs)"
              />
              <button
                className="rounded-lg bg-accent-info/20 px-3 py-2 text-sm text-accent-info hover:bg-accent-info/30 transition-colors"
                onClick={async () => {
                  if (!jobId) return;
                  try {
                    const json = await fetchNsightJobStatus(jobId);
                    setJobStatus(json);
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
                    const json = await fetchMcpJobStatus(jobId);
                    setJobStatus(json);
                  } catch (e: any) {
                    setJobStatus({ error: e.message });
                  }
                }}
              >
                Check MCP Job
              </button>
              <ResultCard title="Job Status" result={jobStatus} loading={false} />
            </div>
          </div>
        </div>
      </div>

      {/* Exports */}
      <div className="card">
        <div className="card-header">
          <h3 className="font-medium text-white">Data Exports</h3>
        </div>
        <div className="card-body space-y-4">
          <div className="flex flex-wrap gap-2 items-center">
            <button
              className="rounded-lg bg-accent-success/20 px-3 py-2 text-sm text-accent-success hover:bg-accent-success/30 transition-colors flex items-center gap-2"
              onClick={() => exportCsv.trigger({ url: `/api/export/csv?detailed=${detailedCsv ? 1 : 0}` })}
            >
              <Download className="w-4 h-4" />
              Export CSV{detailedCsv ? ' (Detailed)' : ''}
            </button>
            <label className="flex items-center gap-2 text-sm text-white/80">
              <input type="checkbox" checked={detailedCsv} onChange={(e) => setDetailedCsv(e.target.checked)} className="accent-accent-primary" />
              Detailed
            </label>
            <button
              className="rounded-lg bg-accent-info/20 px-3 py-2 text-sm text-accent-info hover:bg-accent-info/30 transition-colors flex items-center gap-2"
              onClick={() => exportHtml.trigger({ url: '/api/export/html' })}
            >
              <Download className="w-4 h-4" />
              Export HTML
            </button>
            <span className="text-xs text-white/40">PDF export downloads directly from server (not previewed).</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <ResultCard title="CSV Export" result={(exportCsv.data as Result) || null} loading={exportCsv.isMutating} error={(exportCsv.error as any)?.message || null} />
            <ResultCard title="HTML Export" result={(exportHtml.data as Result) || null} loading={exportHtml.isMutating} error={(exportHtml.error as any)?.message || null} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default HwbenchTab;
