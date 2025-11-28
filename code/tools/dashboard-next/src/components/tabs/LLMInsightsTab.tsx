'use client';

import { useState, useEffect } from 'react';
import {
  getLLMAnalysis,
  getLLMStatus,
  analyzeLLMBottlenecks,
  getLLMAdvisor,
  getLLMCustomQuery,
  getLLMDistributed,
  getLLMInference,
  getLLMRLHF,
} from '@/lib/api';
import { Brain, AlertTriangle, Lightbulb, Loader2, RefreshCw, Sparkles, Send } from 'lucide-react';
import { AISuggestionsCard } from '@/components/AISuggestionsCard';

export function LLMInsightsTab() {
  const [analysis, setAnalysis] = useState<any>(null);
  const [status, setStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [advisor, setAdvisor] = useState<any>(null);
  const [custom, setCustom] = useState<any>(null);
  const [distPlan, setDistPlan] = useState<any>(null);
  const [inferPlan, setInferPlan] = useState<any>(null);
  const [rlhfPlan, setRlhfPlan] = useState<any>(null);
  const [quickLoading, setQuickLoading] = useState<string | null>(null);
  const [distParams, setDistParams] = useState({ nodes: 2, gpus: 8, params: 70, interconnect: 'infiniband' });
  const [inferParams, setInferParams] = useState({ model: 'llama-3.1-70b', batch: 32, seq: 4096 });
  const [rlhfParams, setRlhfParams] = useState({ policy: 7, reward: 7, gpus: 8 });
  const distPresets = [
    { label: '2x8 NVLink 70B', nodes: 2, gpus: 8, params: 70, interconnect: 'nvlink' },
    { label: '4x8 IB 405B', nodes: 4, gpus: 8, params: 405, interconnect: 'infiniband' },
  ];
  const inferPresets = [
    { label: '70B fast', model: 'llama-3.1-70b', batch: 32, seq: 4096 },
    { label: '8B LLM', model: 'llama-3.1-8b', batch: 64, seq: 2048 },
  ];
  const rlhfPresets = [
    { label: 'Policy/Reward 7B', policy: 7, reward: 7, gpus: 8 },
    { label: 'Policy 13B Reward 7B', policy: 13, reward: 7, gpus: 16 },
  ];

  async function loadAnalysis() {
    try {
      setError(null);
      const [analysisData, statusData, advisorData, customData] = await Promise.all([
        getLLMAnalysis(),
        getLLMStatus(),
        getLLMAdvisor({ model: 'llama-3.1-70b', goal: 'throughput', gpus: 8 }).catch(() => null),
        getLLMCustomQuery('Outline a memory-optimized attention stack').catch(() => null),
      ]);
      setAnalysis(analysisData);
      setStatus(statusData);
      setAdvisor(advisorData);
      setCustom(customData);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load LLM analysis');
    } finally {
      setLoading(false);
    }
  }

  async function runQuick(kind: 'dist' | 'infer' | 'rlhf') {
    try {
      setQuickLoading(kind);
      if (kind === 'dist') {
        const res = await getLLMDistributed({
          nodes: distParams.nodes,
          gpus: distParams.gpus,
          params: distParams.params,
          interconnect: distParams.interconnect,
        });
        setDistPlan(res);
      } else if (kind === 'infer') {
        const res = await getLLMInference({
          model: inferParams.model,
          batch: inferParams.batch,
          seq: inferParams.seq,
          latency: 0,
          throughput: 0,
        });
        setInferPlan(res);
      } else {
        const res = await getLLMRLHF({
          policy: rlhfParams.policy,
          reward: rlhfParams.reward,
          gpus: rlhfParams.gpus,
        });
        setRlhfPlan(res);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'LLM quick plan failed');
    } finally {
      setQuickLoading(null);
    }
  }

  async function runAnalysis() {
    setAnalyzing(true);
    try {
      const result = await analyzeLLMBottlenecks({});
      setAnalysis(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to run analysis');
    } finally {
      setAnalyzing(false);
    }
  }

  useEffect(() => {
    loadAnalysis();
  }, []);

  if (loading) {
    return (
      <div className="card">
        <div className="card-body flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 animate-spin text-accent-secondary" />
          <span className="ml-3 text-white/50">Loading LLM analysis...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="card-body text-center py-16">
          <AlertTriangle className="w-12 h-12 text-accent-danger mx-auto mb-4" />
          <p className="text-white/70 mb-2">{error}</p>
          <button
            onClick={loadAnalysis}
            className="px-4 py-2 bg-accent-primary/20 text-accent-primary rounded-lg hover:bg-accent-primary/30"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Status */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-accent-secondary" />
            <h2 className="text-lg font-semibold text-white">LLM Analysis Engine</h2>
          </div>
          <div className="flex items-center gap-3">
            {status && (
              <span className={`text-sm ${status.available ? 'text-accent-success' : 'text-accent-warning'}`}>
                {status.available ? '✓ LLM Connected' : '⚠ LLM Unavailable'}
              </span>
            )}
            <button
              onClick={runAnalysis}
              disabled={analyzing}
              className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-accent-secondary to-accent-tertiary text-white rounded-lg font-medium hover:opacity-90 disabled:opacity-50"
            >
              {analyzing ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Sparkles className="w-4 h-4" />
              )}
              {analyzing ? 'Analyzing...' : 'Run Analysis'}
            </button>
          </div>
        </div>
      </div>

      {/* Summary */}
      {analysis?.summary && (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">AI Analysis Summary</h3>
          </div>
          <div className="card-body">
            <p className="text-white/80 leading-relaxed">{analysis.summary}</p>
          </div>
        </div>
      )}

      {/* Key Findings */}
      {analysis?.key_findings && analysis.key_findings.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">🔍 Key Findings</h3>
          </div>
          <div className="card-body space-y-3">
            {analysis.key_findings.map((finding: string, index: number) => (
              <div
                key={index}
                className="flex items-start gap-3 p-3 bg-white/5 rounded-lg"
              >
                <span className="flex-shrink-0 w-6 h-6 bg-accent-primary/20 text-accent-primary rounded-full flex items-center justify-center text-sm font-bold">
                  {index + 1}
                </span>
                <p className="text-white/80">{finding}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendations */}
      {analysis?.recommendations && analysis.recommendations.length > 0 && (
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <Lightbulb className="w-5 h-5 text-accent-warning" />
              <h3 className="font-medium text-white">Recommendations</h3>
            </div>
          </div>
          <div className="card-body space-y-3">
            {analysis.recommendations.map((rec: string, index: number) => (
              <div
                key={index}
                className="p-4 bg-gradient-to-r from-accent-warning/10 to-transparent rounded-lg border-l-2 border-accent-warning"
              >
                <p className="text-white/80">{rec}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Bottlenecks */}
      {analysis?.bottlenecks && analysis.bottlenecks.length > 0 && (
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-accent-danger" />
              <h3 className="font-medium text-white">Detected Bottlenecks</h3>
            </div>
          </div>
          <div className="card-body space-y-4">
            {analysis.bottlenecks.map((bottleneck: any, index: number) => (
              <div
                key={index}
                className={`p-4 rounded-lg border-l-4 ${
                  bottleneck.severity === 'high'
                    ? 'bg-accent-danger/10 border-accent-danger'
                    : bottleneck.severity === 'medium'
                    ? 'bg-accent-warning/10 border-accent-warning'
                    : 'bg-accent-info/10 border-accent-info'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-white">{bottleneck.name}</h4>
                  <span
                    className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                      bottleneck.severity === 'high'
                        ? 'bg-accent-danger/20 text-accent-danger'
                        : bottleneck.severity === 'medium'
                        ? 'bg-accent-warning/20 text-accent-warning'
                        : 'bg-accent-info/20 text-accent-info'
                    }`}
                  >
                    {bottleneck.severity}
                  </span>
                </div>
                <p className="text-sm text-white/70 mb-2">{bottleneck.description}</p>
                {bottleneck.recommendation && (
                  <p className="text-sm text-accent-success">
                    💡 {bottleneck.recommendation}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      <AISuggestionsCard />

      {(advisor || custom) && (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">LLM Advisor</h3>
          </div>
          <div className="card-body space-y-3">
            {advisor && (
              <div className="p-3 rounded-lg bg-white/5 border border-white/10 text-sm text-white/80">
                {JSON.stringify(advisor, null, 2)}
              </div>
            )}
            {custom && (
              <div className="p-3 rounded-lg bg-white/5 border border-white/10 text-sm text-white/80">
                {custom.answer || JSON.stringify(custom, null, 2)}
              </div>
            )}
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Send className="w-5 h-5 text-accent-info" />
            <h3 className="font-medium text-white">LLM Quick Plans</h3>
          </div>
        </div>
        <div className="card-body space-y-3">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="p-3 rounded-lg bg-white/5 border border-white/10 space-y-2">
              <div className="text-sm text-white font-semibold">Distributed</div>
              <select
                value=""
                onChange={(e) => {
                  const preset = distPresets.find((p) => p.label === e.target.value);
                  if (preset) setDistParams(preset);
                }}
                className="w-full px-2 py-1 bg-white/10 border border-white/20 rounded text-xs text-white"
              >
                <option value="">Custom</option>
                {distPresets.map((p) => (
                  <option key={p.label} value={p.label} className="bg-brand-bg text-white">
                    {p.label}
                  </option>
                ))}
              </select>
              <div className="grid grid-cols-2 gap-2 text-xs text-white/80">
                <label className="flex flex-col gap-1">
                  <span className="text-white/50">Nodes</span>
                  <input
                    type="number"
                    value={distParams.nodes}
                    onChange={(e) => setDistParams({ ...distParams, nodes: Number(e.target.value) })}
                    className="px-2 py-1 rounded bg-white/10 border border-white/20"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-white/50">GPUs</span>
                  <input
                    type="number"
                    value={distParams.gpus}
                    onChange={(e) => setDistParams({ ...distParams, gpus: Number(e.target.value) })}
                    className="px-2 py-1 rounded bg-white/10 border border-white/20"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-white/50">Params (B)</span>
                  <input
                    type="number"
                    value={distParams.params}
                    onChange={(e) => setDistParams({ ...distParams, params: Number(e.target.value) })}
                    className="px-2 py-1 rounded bg-white/10 border border-white/20"
                  />
                </label>
                <label className="flex flex-col gap-1 col-span-2">
                  <span className="text-white/50">Interconnect</span>
                  <input
                    value={distParams.interconnect}
                    onChange={(e) => setDistParams({ ...distParams, interconnect: e.target.value })}
                    className="px-2 py-1 rounded bg-white/10 border border-white/20"
                  />
                </label>
              </div>
              <button
                onClick={() => runQuick('dist')}
                disabled={quickLoading === 'dist'}
                className="w-full px-3 py-2 rounded bg-accent-primary/20 text-accent-primary text-sm font-medium disabled:opacity-50"
              >
                {quickLoading === 'dist' ? 'Running...' : 'Get Plan'}
              </button>
            </div>

            <div className="p-3 rounded-lg bg-white/5 border border-white/10 space-y-2">
              <div className="text-sm text-white font-semibold">Inference</div>
              <select
                value=""
                onChange={(e) => {
                  const preset = inferPresets.find((p) => p.label === e.target.value);
                  if (preset) setInferParams(preset);
                }}
                className="w-full px-2 py-1 bg-white/10 border border-white/20 rounded text-xs text-white"
              >
                <option value="">Custom</option>
                {inferPresets.map((p) => (
                  <option key={p.label} value={p.label} className="bg-brand-bg text-white">
                    {p.label}
                  </option>
                ))}
              </select>
              <div className="grid grid-cols-2 gap-2 text-xs text-white/80">
                <label className="flex flex-col gap-1 col-span-2">
                  <span className="text-white/50">Model</span>
                  <input
                    value={inferParams.model}
                    onChange={(e) => setInferParams({ ...inferParams, model: e.target.value })}
                    className="px-2 py-1 rounded bg-white/10 border border-white/20"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-white/50">Batch</span>
                  <input
                    type="number"
                    value={inferParams.batch}
                    onChange={(e) => setInferParams({ ...inferParams, batch: Number(e.target.value) })}
                    className="px-2 py-1 rounded bg-white/10 border border-white/20"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-white/50">Seq</span>
                  <input
                    type="number"
                    value={inferParams.seq}
                    onChange={(e) => setInferParams({ ...inferParams, seq: Number(e.target.value) })}
                    className="px-2 py-1 rounded bg-white/10 border border-white/20"
                  />
                </label>
              </div>
              <button
                onClick={() => runQuick('infer')}
                disabled={quickLoading === 'infer'}
                className="w-full px-3 py-2 rounded bg-accent-secondary/20 text-accent-secondary text-sm font-medium disabled:opacity-50"
              >
                {quickLoading === 'infer' ? 'Running...' : 'Get Plan'}
              </button>
            </div>

            <div className="p-3 rounded-lg bg-white/5 border border-white/10 space-y-2">
              <div className="text-sm text-white font-semibold">RLHF</div>
              <select
                value=""
                onChange={(e) => {
                  const preset = rlhfPresets.find((p) => p.label === e.target.value);
                  if (preset) setRlhfParams(preset);
                }}
                className="w-full px-2 py-1 bg-white/10 border border-white/20 rounded text-xs text-white"
              >
                <option value="">Custom</option>
                {rlhfPresets.map((p) => (
                  <option key={p.label} value={p.label} className="bg-brand-bg text-white">
                    {p.label}
                  </option>
                ))}
              </select>
              <div className="grid grid-cols-2 gap-2 text-xs text-white/80">
                <label className="flex flex-col gap-1">
                  <span className="text-white/50">Policy (B)</span>
                  <input
                    type="number"
                    value={rlhfParams.policy}
                    onChange={(e) => setRlhfParams({ ...rlhfParams, policy: Number(e.target.value) })}
                    className="px-2 py-1 rounded bg-white/10 border border-white/20"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-white/50">Reward (B)</span>
                  <input
                    type="number"
                    value={rlhfParams.reward}
                    onChange={(e) => setRlhfParams({ ...rlhfParams, reward: Number(e.target.value) })}
                    className="px-2 py-1 rounded bg-white/10 border border-white/20"
                  />
                </label>
                <label className="flex flex-col gap-1 col-span-2">
                  <span className="text-white/50">GPUs</span>
                  <input
                    type="number"
                    value={rlhfParams.gpus}
                    onChange={(e) => setRlhfParams({ ...rlhfParams, gpus: Number(e.target.value) })}
                    className="px-2 py-1 rounded bg-white/10 border border-white/20"
                  />
                </label>
              </div>
              <button
                onClick={() => runQuick('rlhf')}
                disabled={quickLoading === 'rlhf'}
                className="w-full px-3 py-2 rounded bg-accent-info/20 text-accent-info text-sm font-medium disabled:opacity-50"
              >
                {quickLoading === 'rlhf' ? 'Running...' : 'Get Plan'}
              </button>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <ResultPanel title="Distributed" value={distPlan} loading={quickLoading === 'dist'} />
            <ResultPanel title="Inference" value={inferPlan} loading={quickLoading === 'infer'} />
            <ResultPanel title="RLHF" value={rlhfPlan} loading={quickLoading === 'rlhf'} />
          </div>
        </div>
      </div>
    </div>
  );
}

function ResultPanel({ title, value, loading }: { title: string; value: any; loading?: boolean }) {
  return (
    <div className="p-3 rounded-lg bg-white/5 border border-white/10 text-xs text-white/80 min-h-[120px]">
      <div className="text-white/60 text-xs mb-2 uppercase tracking-wide">{title}</div>
      {loading ? (
        <div className="flex items-center gap-2 text-white/60">
          <Loader2 className="w-3 h-3 animate-spin" /> Running...
        </div>
      ) : value ? (
        <pre className="whitespace-pre-wrap break-words">{JSON.stringify(value, null, 2)}</pre>
      ) : (
        <div className="text-white/40">No output yet.</div>
      )}
    </div>
  );
}
