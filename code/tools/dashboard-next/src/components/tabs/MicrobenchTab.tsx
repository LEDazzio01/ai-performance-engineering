import { useState } from 'react';
import useSWRMutation from 'swr/mutation';

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
    <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-slate-100">{title}</h3>
        {loading && <span className="text-xs text-slate-400">Loading…</span>}
      </div>
      {error ? (
        <pre className="text-xs text-red-300 whitespace-pre-wrap break-all">{error}</pre>
      ) : result ? (
        <pre className="text-xs text-slate-200 whitespace-pre-wrap break-all">{JSON.stringify(result, null, 2)}</pre>
      ) : (
        <p className="text-xs text-slate-500">No data yet.</p>
      )}
    </div>
  );
}

function MetricsTable({ result }: { result: any }) {
  if (!result || typeof result !== 'object') return null;
  const metrics = Array.isArray(result.metrics) ? result.metrics : Array.isArray(result) ? result : null;
  if (!metrics) return null;
  return (
    <div className="overflow-auto rounded border border-slate-800">
      <table className="min-w-full text-xs text-left text-slate-200">
        <thead className="bg-slate-800 text-slate-300">
          <tr>
            <th className="px-2 py-1">Name</th>
            <th className="px-2 py-1">Baseline</th>
            <th className="px-2 py-1">Optimized</th>
            <th className="px-2 py-1">Delta</th>
          </tr>
        </thead>
        <tbody>
          {metrics.map((m: any, idx: number) => (
            <tr key={idx} className="border-t border-slate-800">
              <td className="px-2 py-1">{m.name || m.metric || m.section || 'metric'}</td>
              <td className="px-2 py-1">{m.baseline ?? m.base ?? ''}</td>
              <td className="px-2 py-1">{m.optimized ?? m.opt ?? ''}</td>
              <td className="px-2 py-1">{m.delta ?? ''}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function MicrobenchTab() {
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

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-slate-100">Microbenchmarks</h2>
        <p className="text-sm text-slate-400">Quick, lightweight diagnostics (disk, PCIe, memory, tensor cores, SFU, loopback).</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <div className="flex flex-wrap gap-2 items-center">
            <button
              className="rounded bg-slate-800 px-3 py-2 text-sm text-slate-100 hover:bg-slate-700"
              onClick={() => disk.trigger({ url: `/api/microbench/disk?file_size_mb=${diskSize}&block_size_kb=${diskBlock}` })}
            >
              Run Disk I/O
            </button>
            <label className="text-xs text-slate-300 flex items-center gap-1">
              Size MB <input className="bg-slate-800 px-2 py-1 rounded w-16" type="number" value={diskSize} onChange={(e) => setDiskSize(Number(e.target.value))} />
            </label>
            <label className="text-xs text-slate-300 flex items-center gap-1">
              Block KB <input className="bg-slate-800 px-2 py-1 rounded w-16" type="number" value={diskBlock} onChange={(e) => setDiskBlock(Number(e.target.value))} />
            </label>
          </div>
          <ResultCard title="Disk I/O" result={(disk.data as Result) || null} loading={disk.isMutating} error={(disk.error as any)?.message || null} />
          <div className="flex items-center gap-2">
            <button
              className="rounded bg-slate-800 px-3 py-2 text-sm text-slate-100 hover:bg-slate-700"
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
              Export PDF
            </button>
            <span className="text-xs text-slate-400">{pdfStatus || 'PDF downloads directly.'}</span>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex gap-2 items-center">
            <button
              className="rounded bg-slate-800 px-3 py-2 text-sm text-slate-100 hover:bg-slate-700"
              onClick={() => pcie.trigger({ url: `/api/microbench/pcie?size_mb=${pcieSize}&iters=${pcieIters}` })}
            >
              Run PCIe H2D/D2H
            </button>
            <label className="text-xs text-slate-300 flex items-center gap-1">
              Size MB <input className="bg-slate-800 px-2 py-1 rounded w-16" type="number" value={pcieSize} onChange={(e) => setPcieSize(Number(e.target.value))} />
            </label>
            <label className="text-xs text-slate-300 flex items-center gap-1">
              Iters <input className="bg-slate-800 px-2 py-1 rounded w-14" type="number" value={pcieIters} onChange={(e) => setPcieIters(Number(e.target.value))} />
            </label>
          </div>
          <ResultCard title="PCIe" result={(pcie.data as Result) || null} loading={pcie.isMutating} error={(pcie.error as any)?.message || null} />
        </div>

        <div className="space-y-2">
          <div className="flex gap-2 items-center">
            <button
              className="rounded bg-slate-800 px-3 py-2 text-sm text-slate-100 hover:bg-slate-700"
              onClick={() => mem.trigger({ url: `/api/microbench/mem?size_mb=${memSize}&stride=${memStride}` })}
            >
              Run Memory Stride
            </button>
            <label className="text-xs text-slate-300 flex items-center gap-1">
              Size MB <input className="bg-slate-800 px-2 py-1 rounded w-16" type="number" value={memSize} onChange={(e) => setMemSize(Number(e.target.value))} />
            </label>
            <label className="text-xs text-slate-300 flex items-center gap-1">
              Stride <input className="bg-slate-800 px-2 py-1 rounded w-16" type="number" value={memStride} onChange={(e) => setMemStride(Number(e.target.value))} />
            </label>
          </div>
          <ResultCard title="Memory Hierarchy" result={(mem.data as Result) || null} loading={mem.isMutating} error={(mem.error as any)?.message || null} />
        </div>

        <div className="space-y-2">
          <div className="flex gap-2 items-center">
            <button
              className="rounded bg-slate-800 px-3 py-2 text-sm text-slate-100 hover:bg-slate-700"
              onClick={() => tensor.trigger({ url: `/api/microbench/tensor?size=${tensorSize}&precision=${encodeURIComponent(tensorPrecision)}` })}
            >
              Run Tensor Core
            </button>
            <label className="text-xs text-slate-300 flex items-center gap-1">
              Size <input className="bg-slate-800 px-2 py-1 rounded w-16" type="number" value={tensorSize} onChange={(e) => setTensorSize(Number(e.target.value))} />
            </label>
            <label className="text-xs text-slate-300 flex items-center gap-1">
              Precision <input className="bg-slate-800 px-2 py-1 rounded w-20" type="text" value={tensorPrecision} onChange={(e) => setTensorPrecision(e.target.value)} />
            </label>
          </div>
          <ResultCard title="Tensor Core" result={(tensor.data as Result) || null} loading={tensor.isMutating} error={(tensor.error as any)?.message || null} />
        </div>

        <div className="space-y-2">
          <div className="flex gap-2 items-center">
            <button
              className="rounded bg-slate-800 px-3 py-2 text-sm text-slate-100 hover:bg-slate-700"
              onClick={() => sfu.trigger({ url: `/api/microbench/sfu?elements=${sfuElems}` })}
            >
              Run SFU
            </button>
            <label className="text-xs text-slate-300 flex items-center gap-1">
              Elements <input className="bg-slate-800 px-2 py-1 rounded w-24" type="number" value={sfuElems} onChange={(e) => setSfuElems(Number(e.target.value))} />
            </label>
          </div>
          <ResultCard title="SFU" result={(sfu.data as Result) || null} loading={sfu.isMutating} error={(sfu.error as any)?.message || null} />
        </div>

        <div className="space-y-2">
          <div className="flex gap-2 items-center">
            <button
              className="rounded bg-slate-800 px-3 py-2 text-sm text-slate-100 hover:bg-slate-700"
              onClick={() => loop.trigger({ url: `/api/microbench/loopback?size_mb=${loopSize}&port=${loopPort}` })}
            >
              Run Loopback
            </button>
            <label className="text-xs text-slate-300 flex items-center gap-1">
              Size MB <input className="bg-slate-800 px-2 py-1 rounded w-16" type="number" value={loopSize} onChange={(e) => setLoopSize(Number(e.target.value))} />
            </label>
            <label className="text-xs text-slate-300 flex items-center gap-1">
              Port <input className="bg-slate-800 px-2 py-1 rounded w-16" type="number" value={loopPort} onChange={(e) => setLoopPort(Number(e.target.value))} />
            </label>
          </div>
          <ResultCard title="Loopback" result={(loop.data as Result) || null} loading={loop.isMutating} error={(loop.error as any)?.message || null} />
        </div>
      </div>

      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-slate-100">Nsight Utilities</h3>
        <div className="flex flex-wrap gap-2">
          <button
            className="rounded bg-slate-800 px-3 py-2 text-sm text-slate-100 hover:bg-slate-700"
            onClick={() => nsightAvail.trigger({ url: '/api/nsight/availability' })}
          >
            Check Nsight Availability
          </button>
          <input
            className="rounded bg-slate-800 px-2 py-2 text-sm text-slate-100 w-64"
            value={compareDir}
            onChange={(e) => setCompareDir(e.target.value)}
            placeholder="Profiles dir"
          />
          <button
            className="rounded bg-slate-800 px-3 py-2 text-sm text-slate-100 hover:bg-slate-700"
            onClick={() => compareNsys.trigger({ url: `/api/nsight/compare/nsys?dir=${encodeURIComponent(compareDir)}` })}
          >
            Compare Nsight Systems
          </button>
          <button
            className="rounded bg-slate-800 px-3 py-2 text-sm text-slate-100 hover:bg-slate-700"
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
      </div>

      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-slate-100">Exports</h3>
        <div className="flex flex-wrap gap-2 items-center">
          <button
            className="rounded bg-slate-800 px-3 py-2 text-sm text-slate-100 hover:bg-slate-700"
            onClick={() => exportCsv.trigger({ url: `/api/export/csv?detailed=${detailedCsv ? 1 : 0}` })}
          >
            Export CSV{detailedCsv ? ' (Detailed)' : ''}
          </button>
          <label className="flex items-center gap-2 text-sm text-slate-200">
            <input type="checkbox" checked={detailedCsv} onChange={(e) => setDetailedCsv(e.target.checked)} />
            Detailed
          </label>
          <button
            className="rounded bg-slate-800 px-3 py-2 text-sm text-slate-100 hover:bg-slate-700"
            onClick={() => exportHtml.trigger({ url: '/api/export/html' })}
          >
            Export HTML
          </button>
          <span className="text-xs text-slate-400">PDF export downloads directly from server (not previewed).</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <ResultCard title="CSV Export" result={(exportCsv.data as Result) || null} loading={exportCsv.isMutating} error={(exportCsv.error as any)?.message || null} />
          <ResultCard title="HTML Export" result={(exportHtml.data as Result) || null} loading={exportHtml.isMutating} error={(exportHtml.error as any)?.message || null} />
        </div>
      </div>
    </div>
  );
}
