'use client';

import { useState, useEffect } from 'react';
import { Package, Loader2, AlertTriangle, RefreshCw, Play, CheckCircle, XCircle, Clock } from 'lucide-react';
import { getTargets, batchOptimize, getOptimizeJobs, calculateBatch, getQuantizationComparison } from '@/lib/api';

export function BatchOptTab() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [targets, setTargets] = useState<string[]>([]);
  const [selectedTargets, setSelectedTargets] = useState<Set<string>>(new Set());
  const [jobs, setJobs] = useState<any[]>([]);
  const [running, setRunning] = useState(false);
  const [calcParams, setCalcParams] = useState({
    model: 'llama-3.1-70b',
    seq_length: 4096,
    gpus: 8,
    memory_gb: 80,
  });
  const [calcResult, setCalcResult] = useState<any>(null);
  const [quantParams, setQuantParams] = useState({
    model: 'llama-3.1-70b',
    precision: 'fp16',
    batch: 32,
  });
  const [quantResult, setQuantResult] = useState<any>(null);
  const [toolsRunning, setToolsRunning] = useState<'calc' | 'quant' | null>(null);

  async function loadData() {
    try {
      setLoading(true);
      setError(null);
      const [targetsData, jobsData] = await Promise.all([
        getTargets(),
        getOptimizeJobs().catch(() => ({ jobs: [] })),
      ]);
      setTargets(targetsData || []);
      const jobList = (jobsData as any)?.jobs || jobsData || [];
      setJobs(jobList);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load batch optimization data');
    } finally {
      setLoading(false);
    }
  }

  async function runBatch() {
    if (selectedTargets.size === 0) return;

    setRunning(true);
    try {
      await batchOptimize({ targets: Array.from(selectedTargets) });
      // Refresh jobs list
      const jobsData = await getOptimizeJobs();
      const jobList = (jobsData as any)?.jobs || jobsData || [];
      setJobs(jobList);
      setSelectedTargets(new Set());
    } catch (e) {
      console.error('Batch optimization failed:', e);
    } finally {
      setRunning(false);
    }
  }

  async function runCalc() {
    setToolsRunning('calc');
    try {
      const res = await calculateBatch({
        model: calcParams.model,
        seq_length: Number(calcParams.seq_length),
        gpus: Number(calcParams.gpus),
        memory_gb: Number(calcParams.memory_gb),
      });
      setCalcResult(res);
    } catch (e) {
      setCalcResult({ error: e instanceof Error ? e.message : 'Failed to calculate batch size' });
    } finally {
      setToolsRunning(null);
    }
  }

  async function runQuant() {
    setToolsRunning('quant');
    try {
      const res = await getQuantizationComparison({
        model: quantParams.model,
        precision: quantParams.precision,
        batch: Number(quantParams.batch),
      });
      setQuantResult(res);
    } catch (e) {
      setQuantResult({ error: e instanceof Error ? e.message : 'Failed to evaluate quantization' });
    } finally {
      setToolsRunning(null);
    }
  }

  function toggleTarget(target: string) {
    const newSelected = new Set(selectedTargets);
    if (newSelected.has(target)) {
      newSelected.delete(target);
    } else {
      newSelected.add(target);
    }
    setSelectedTargets(newSelected);
  }

  function selectAll() {
    setSelectedTargets(new Set(targets));
  }

  function selectNone() {
    setSelectedTargets(new Set());
  }

  useEffect(() => {
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="card">
        <div className="card-body flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 animate-spin text-accent-warning" />
          <span className="ml-3 text-white/50">Loading batch optimization...</span>
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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Package className="w-5 h-5 text-accent-warning" />
            <h2 className="text-lg font-semibold text-white">Batch Optimization</h2>
          </div>
          <button onClick={loadData} className="p-2 hover:bg-white/5 rounded-lg">
            <RefreshCw className="w-4 h-4 text-white/50" />
          </button>
        </div>
      </div>

      {/* Target selection */}
      <div className="card">
        <div className="card-header">
          <h3 className="font-medium text-white">Select Targets</h3>
          <div className="flex items-center gap-2">
            <button
              onClick={selectAll}
              className="text-sm text-accent-primary hover:underline"
            >
              Select All
            </button>
            <span className="text-white/20">|</span>
            <button
              onClick={selectNone}
              className="text-sm text-white/50 hover:text-white"
            >
              Clear
            </button>
          </div>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 max-h-[300px] overflow-y-auto">
            {targets.map((target, i) => (
              <button
                key={i}
                onClick={() => toggleTarget(target)}
                className={`p-3 rounded-lg text-left transition-all ${
                  selectedTargets.has(target)
                    ? 'bg-accent-primary/20 border border-accent-primary/30'
                    : 'bg-white/5 border border-white/10 hover:bg-white/10'
                }`}
              >
                <div className="flex items-center gap-2">
                  <div
                    className={`w-4 h-4 rounded border ${
                      selectedTargets.has(target)
                        ? 'bg-accent-primary border-accent-primary'
                        : 'border-white/30'
                    }`}
                  >
                    {selectedTargets.has(target) && (
                      <CheckCircle className="w-4 h-4 text-black" />
                    )}
                  </div>
                  <span className="text-sm text-white truncate">{target}</span>
                </div>
              </button>
            ))}
          </div>

          <div className="mt-4 pt-4 border-t border-white/5 flex items-center justify-between">
            <span className="text-sm text-white/50">
              {selectedTargets.size} of {targets.length} selected
            </span>
            <button
              onClick={runBatch}
              disabled={selectedTargets.size === 0 || running}
              className="flex items-center gap-2 px-6 py-2 bg-gradient-to-r from-accent-warning to-accent-tertiary text-black rounded-lg font-medium disabled:opacity-50"
            >
              {running ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              {running ? 'Running...' : 'Run Batch'}
            </button>
          </div>
        </div>
      </div>

      {/* Job history */}
      {jobs.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">Job History</h3>
          </div>
          <div className="card-body">
            <div className="space-y-2">
              {jobs.slice(0, 10).map((job, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between p-4 bg-white/5 rounded-lg"
                >
                  <div className="flex items-center gap-3">
                    {job.status === 'completed' ? (
                      <CheckCircle className="w-5 h-5 text-accent-success" />
                    ) : job.status === 'failed' ? (
                      <XCircle className="w-5 h-5 text-accent-danger" />
                    ) : job.status === 'running' ? (
                      <Loader2 className="w-5 h-5 animate-spin text-accent-primary" />
                    ) : (
                      <Clock className="w-5 h-5 text-white/40" />
                    )}
                    <div>
                      <div className="font-medium text-white">
                        {job.targets?.length || 1} targets
                      </div>
                      <div className="text-sm text-white/50">
                        {job.timestamp || job.created_at || 'N/A'}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-white/50">{job.status}</div>
                    {job.avg_speedup && (
                      <div className="text-accent-success font-bold">
                        {job.avg_speedup.toFixed(2)}x avg
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">Batch Size Calculator</h3>
          </div>
          <div className="card-body space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-white/50">Model</label>
                <input
                  value={calcParams.model}
                  onChange={(e) => setCalcParams({ ...calcParams, model: e.target.value })}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white"
                />
              </div>
              <div>
                <label className="text-xs text-white/50">Seq Length</label>
                <input
                  type="number"
                  value={calcParams.seq_length}
                  onChange={(e) => setCalcParams({ ...calcParams, seq_length: Number(e.target.value) })}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white"
                />
              </div>
              <div>
                <label className="text-xs text-white/50">GPUs</label>
                <input
                  type="number"
                  value={calcParams.gpus}
                  onChange={(e) => setCalcParams({ ...calcParams, gpus: Number(e.target.value) })}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white"
                />
              </div>
              <div>
                <label className="text-xs text-white/50">Memory (GB)</label>
                <input
                  type="number"
                  value={calcParams.memory_gb}
                  onChange={(e) => setCalcParams({ ...calcParams, memory_gb: Number(e.target.value) })}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white"
                />
              </div>
            </div>
            <button
              onClick={runCalc}
              disabled={toolsRunning === 'calc'}
              className="px-4 py-2 rounded-lg bg-accent-primary/20 text-accent-primary text-sm disabled:opacity-50"
            >
              {toolsRunning === 'calc' ? 'Calculating…' : 'Calculate'}
            </button>
            <div className="p-3 rounded-lg bg-white/5 border border-white/10 text-xs text-white/80 min-h-[120px]">
              {calcResult ? (
                <pre className="whitespace-pre-wrap break-words">{JSON.stringify(calcResult, null, 2)}</pre>
              ) : (
                <span className="text-white/40">Results will appear here.</span>
              )}
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">Quantization Planner</h3>
          </div>
          <div className="card-body space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-white/50">Model</label>
                <input
                  value={quantParams.model}
                  onChange={(e) => setQuantParams({ ...quantParams, model: e.target.value })}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white"
                />
              </div>
              <div>
                <label className="text-xs text-white/50">Precision</label>
                <select
                  value={quantParams.precision}
                  onChange={(e) => setQuantParams({ ...quantParams, precision: e.target.value })}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white"
                >
                  {['fp16', 'bf16', 'int8', 'int4'].map((p) => (
                    <option key={p} value={p} className="bg-brand-bg">
                      {p.toUpperCase()}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-xs text-white/50">Batch</label>
                <input
                  type="number"
                  value={quantParams.batch}
                  onChange={(e) => setQuantParams({ ...quantParams, batch: Number(e.target.value) })}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white"
                />
              </div>
            </div>
            <button
              onClick={runQuant}
              disabled={toolsRunning === 'quant'}
              className="px-4 py-2 rounded-lg bg-accent-secondary/20 text-accent-secondary text-sm disabled:opacity-50"
            >
              {toolsRunning === 'quant' ? 'Simulating…' : 'Compare Quantization'}
            </button>
            <div className="p-3 rounded-lg bg-white/5 border border-white/10 text-xs text-white/80 min-h-[120px]">
              {quantResult ? (
                <pre className="whitespace-pre-wrap break-words">{JSON.stringify(quantResult, null, 2)}</pre>
              ) : (
                <span className="text-white/40">Quantization trade-offs will appear here.</span>
              )}
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}
