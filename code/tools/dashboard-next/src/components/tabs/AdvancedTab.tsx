'use client';

import { useState, useEffect, useCallback } from 'react';
import { Sparkles, DollarSign, Zap, TrendingUp, Calculator, Loader2, AlertTriangle, RefreshCw } from 'lucide-react';
import { getCostCalculator, getCostROI, simulateWhatIf, getAnalysisScaling, getAnalysisPower, getAnalysisCost } from '@/lib/api';
import { WhatIfSolverCard } from '@/components/WhatIfSolverCard';
import { WarmupAuditCard } from '@/components/WarmupAuditCard';
import { CiCdIntegrationCard } from '@/components/CiCdIntegrationCard';
import { SystemDeepDiveCard, BenchmarkScannerCard } from '@/components';

export function AdvancedTab() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [costData, setCostData] = useState<any>(null);
  const [roiData, setRoiData] = useState<any>(null);
  const [scalingData, setScalingData] = useState<any>(null);
  const [powerData, setPowerData] = useState<any>(null);
  const [analysisCost, setAnalysisCost] = useState<any>(null);
  const [costGpu, setCostGpu] = useState('H100');
  const [costLoading, setCostLoading] = useState(false);

  // What-if form state
  const [whatifVram, setWhatifVram] = useState(24);
  const [whatifLatency, setWhatifLatency] = useState(50);
  const [whatifResult, setWhatifResult] = useState<any>(null);
  const [simulating, setSimulating] = useState(false);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setCostLoading(true);
      setError(null);
      const [cost, roi, scaling, power, costBreakdown] = await Promise.all([
        getCostCalculator().catch(() => null),
        getCostROI().catch(() => null),
        getAnalysisScaling().catch(() => null),
        getAnalysisPower().catch(() => null),
        getAnalysisCost({ gpu: costGpu }).catch(() => null),
      ]);
      setCostData(cost);
      setRoiData(roi);
      setScalingData(scaling);
      setPowerData(power);
      setAnalysisCost(costBreakdown);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load advanced data');
    } finally {
      setLoading(false);
      setCostLoading(false);
    }
  }, [costGpu]);

  async function runWhatIf() {
    setSimulating(true);
    try {
      const result = await simulateWhatIf({
        max_vram: whatifVram,
        max_latency: whatifLatency,
      });
      setWhatifResult(result);
    } catch (e) {
      console.error('What-if simulation failed:', e);
    } finally {
      setSimulating(false);
    }
  }

  const refreshCostOnly = async () => {
    try {
      setCostLoading(true);
      const res = await getAnalysisCost({ gpu: costGpu });
      setAnalysisCost(res);
    } catch {
      setAnalysisCost(null);
    } finally {
      setCostLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [loadData]);

  if (loading) {
    return (
      <div className="card">
        <div className="card-body flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 animate-spin text-accent-secondary" />
          <span className="ml-3 text-white/50">Loading advanced tools...</span>
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
            <Sparkles className="w-5 h-5 text-accent-secondary" />
            <h2 className="text-lg font-semibold text-white">Advanced Tools</h2>
          </div>
          <button onClick={loadData} className="p-2 hover:bg-white/5 rounded-lg">
            <RefreshCw className="w-4 h-4 text-white/50" />
          </button>
        </div>
      </div>

      {/* Cost Calculator */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <DollarSign className="w-5 h-5 text-accent-success" />
            <h3 className="font-medium text-white">Cost Calculator</h3>
          </div>
        </div>
        <div className="card-body">
          {costData ? (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="p-4 bg-white/5 rounded-lg">
                <div className="text-sm text-white/50 mb-1">Hourly Cost</div>
                <div className="text-2xl font-bold text-accent-success">
                  ${costData.hourly_cost?.toFixed(2) || costData.hourly || 'N/A'}
                </div>
              </div>
              <div className="p-4 bg-white/5 rounded-lg">
                <div className="text-sm text-white/50 mb-1">Monthly Cost</div>
                <div className="text-2xl font-bold text-accent-warning">
                  ${costData.monthly_cost?.toFixed(0) || costData.monthly || 'N/A'}
                </div>
              </div>
              <div className="p-4 bg-white/5 rounded-lg">
                <div className="text-sm text-white/50 mb-1">Cost per 1M Tokens</div>
                <div className="text-2xl font-bold text-accent-primary">
                  ${costData.per_million_tokens?.toFixed(4) || 'N/A'}
                </div>
              </div>
              <div className="p-4 bg-white/5 rounded-lg">
                <div className="text-sm text-white/50 mb-1">Savings from Optimization</div>
                <div className="text-2xl font-bold text-accent-info">
                  {costData.savings_percent?.toFixed(0) || roiData?.savings || 0}%
                </div>
              </div>
            </div>
          ) : (
            <p className="text-white/50">Cost data not available</p>
          )}
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <DollarSign className="w-5 h-5 text-accent-primary" />
            <h3 className="font-medium text-white">Cost by GPU</h3>
          </div>
          <div className="flex items-center gap-2">
            <input
              value={costGpu}
              onChange={(e) => setCostGpu(e.target.value)}
              className="px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white"
              placeholder="H100"
            />
            <button onClick={refreshCostOnly} className="p-2 hover:bg-white/5 rounded-lg">
              {costLoading ? (
                <Loader2 className="w-4 h-4 animate-spin text-accent-primary" />
              ) : (
                <RefreshCw className="w-4 h-4 text-white/50" />
              )}
            </button>
          </div>
        </div>
        <div className="card-body">
          {costLoading ? (
            <div className="flex items-center gap-2 text-white/60">
              <Loader2 className="w-4 h-4 animate-spin" /> Updating cost profile...
            </div>
          ) : analysisCost ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm text-white/80">
              <div>
                <div className="text-white/40 text-xs">Hourly</div>
                <div className="text-xl font-bold text-accent-primary">
                  ${analysisCost.hourly?.toFixed?.(2) || analysisCost.hourly || '--'}
                </div>
              </div>
              <div>
                <div className="text-white/40 text-xs">Tokens/sec</div>
                <div className="text-xl font-bold text-accent-secondary">
                  {analysisCost.tokens_per_sec?.toFixed?.(0) || analysisCost.throughput || '--'}
                </div>
              </div>
              <div>
                <div className="text-white/40 text-xs">Efficiency</div>
                <div className="text-xl font-bold text-accent-success">
                  {analysisCost.efficiency || analysisCost.utilization || '--'}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-sm text-white/50">No cost data available for this GPU.</div>
          )}
        </div>
      </div>

      {/* What-If Analysis */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Calculator className="w-5 h-5 text-accent-info" />
            <h3 className="font-medium text-white">What-If Analysis</h3>
          </div>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div>
              <label className="block text-sm text-white/50 mb-2">Max VRAM (GB)</label>
              <input
                type="number"
                value={whatifVram}
                onChange={(e) => setWhatifVram(Number(e.target.value))}
                className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white"
              />
            </div>
            <div>
              <label className="block text-sm text-white/50 mb-2">Max Latency (ms)</label>
              <input
                type="number"
                value={whatifLatency}
                onChange={(e) => setWhatifLatency(Number(e.target.value))}
                className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white"
              />
            </div>
            <div className="flex items-end">
              <button
                onClick={runWhatIf}
                disabled={simulating}
                className="w-full px-4 py-2 bg-gradient-to-r from-accent-info to-accent-primary text-black rounded-lg font-medium disabled:opacity-50"
              >
                {simulating ? 'Simulating...' : 'Run Simulation'}
              </button>
            </div>
          </div>
          {whatifResult && (
            <div className="p-4 bg-accent-info/10 border border-accent-info/20 rounded-lg">
              <h4 className="font-medium text-accent-info mb-2">Simulation Results</h4>
              <pre className="text-sm text-white/70 overflow-x-auto">
                {JSON.stringify(whatifResult, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </div>

      {/* Scaling Analysis */}
      {scalingData && (
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-accent-warning" />
              <h3 className="font-medium text-white">Scaling Analysis</h3>
            </div>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {(scalingData.projections || scalingData.results || []).slice(0, 3).map((proj: any, i: number) => (
                <div key={i} className="p-4 bg-white/5 rounded-lg">
                  <div className="text-sm text-white/50 mb-1">{proj.name || `Scenario ${i + 1}`}</div>
                  <div className="text-xl font-bold text-accent-warning">
                    {proj.speedup?.toFixed(2) || proj.value}x
                  </div>
                  {proj.description && (
                    <p className="text-sm text-white/40 mt-1">{proj.description}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Power Analysis */}
      {powerData && (
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-accent-tertiary" />
              <h3 className="font-medium text-white">Power Efficiency</h3>
            </div>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="p-4 bg-white/5 rounded-lg">
                <div className="text-sm text-white/50 mb-1">Current Power</div>
                <div className="text-2xl font-bold text-accent-tertiary">
                  {powerData.current_watts || powerData.power}W
                </div>
              </div>
              <div className="p-4 bg-white/5 rounded-lg">
                <div className="text-sm text-white/50 mb-1">Efficiency</div>
                <div className="text-2xl font-bold text-accent-success">
                  {powerData.efficiency?.toFixed(0) || 'N/A'}%
                </div>
              </div>
              <div className="p-4 bg-white/5 rounded-lg">
                <div className="text-sm text-white/50 mb-1">TFLOPS/Watt</div>
                <div className="text-2xl font-bold text-accent-primary">
                  {powerData.tflops_per_watt?.toFixed(2) || 'N/A'}
                </div>
              </div>
              <div className="p-4 bg-white/5 rounded-lg">
                <div className="text-sm text-white/50 mb-1">CO2 per Hour</div>
                <div className="text-2xl font-bold text-accent-info">
                  {powerData.co2_per_hour?.toFixed(1) || 'N/A'}g
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <WarmupAuditCard />
      <CiCdIntegrationCard />
      <WhatIfSolverCard />
      <SystemDeepDiveCard />
      <BenchmarkScannerCard />
    </div>
  );
}
