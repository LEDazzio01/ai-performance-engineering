'use client';

import { useState, useEffect } from 'react';
import { Gamepad2, Loader2, AlertTriangle, RefreshCw, TrendingUp, Target } from 'lucide-react';
import { getAnalysisOptimizations, getIntelligenceTechniques } from '@/lib/api';

export function RLHFTab() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [optimizations, setOptimizations] = useState<any>(null);
  const [techniques, setTechniques] = useState<any[]>([]);

  async function loadData() {
    try {
      setLoading(true);
      setError(null);
      const [optsData, techData] = await Promise.all([
        getAnalysisOptimizations().catch(() => null),
        getIntelligenceTechniques().catch(() => []),
      ]);
      setOptimizations(optsData);
      const techniqueList = (techData as any)?.techniques || techData || [];
      setTechniques(techniqueList);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load RL/RLHF data');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="card">
        <div className="card-body flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 animate-spin text-accent-tertiary" />
          <span className="ml-3 text-white/50">Loading RL/RLHF data...</span>
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

  const rlhfMethods = [
    { name: 'PPO', desc: 'Proximal Policy Optimization', complexity: 'High', speedup: '2-3x' },
    { name: 'DPO', desc: 'Direct Preference Optimization', complexity: 'Low', speedup: '5-10x' },
    { name: 'RLHF', desc: 'RL from Human Feedback', complexity: 'Very High', speedup: '1.5-2x' },
    { name: 'RLAIF', desc: 'RL from AI Feedback', complexity: 'Medium', speedup: '3-5x' },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Gamepad2 className="w-5 h-5 text-accent-tertiary" />
            <h2 className="text-lg font-semibold text-white">RL/RLHF Optimization</h2>
          </div>
          <button onClick={loadData} className="p-2 hover:bg-white/5 rounded-lg">
            <RefreshCw className="w-4 h-4 text-white/50" />
          </button>
        </div>
      </div>

      {/* Methods comparison */}
      <div className="card">
        <div className="card-header">
          <h3 className="font-medium text-white">Training Methods</h3>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {rlhfMethods.map((method, i) => (
              <div key={i} className="p-4 bg-white/5 rounded-lg border border-white/10">
                <h4 className="text-lg font-bold text-accent-tertiary mb-1">{method.name}</h4>
                <p className="text-sm text-white/50 mb-3">{method.desc}</p>
                <div className="flex justify-between text-sm">
                  <span className="text-white/40">Complexity</span>
                  <span className="text-white">{method.complexity}</span>
                </div>
                <div className="flex justify-between text-sm mt-1">
                  <span className="text-white/40">Training Speedup</span>
                  <span className="text-accent-success font-bold">{method.speedup}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Optimization techniques */}
      {techniques.length > 0 && (
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <Target className="w-5 h-5 text-accent-success" />
              <h3 className="font-medium text-white">Optimization Techniques</h3>
            </div>
          </div>
          <div className="card-body space-y-3">
            {techniques.map((tech, i) => (
              <div
                key={i}
                className="p-4 bg-white/5 rounded-lg flex items-center justify-between"
              >
                <div>
                  <h4 className="font-medium text-white">{tech.name}</h4>
                  <p className="text-sm text-white/50">{tech.description}</p>
                </div>
                {tech.impact && (
                  <span className="text-accent-success font-bold">{tech.impact}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendations */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-accent-warning" />
            <h3 className="font-medium text-white">Recommendations</h3>
          </div>
        </div>
        <div className="card-body space-y-3">
          {[
            'Use DPO instead of PPO when possible for 5-10x faster training',
            'Enable gradient checkpointing to reduce memory by 60-80%',
            'Use LoRA/QLoRA for efficient fine-tuning with minimal memory',
            'Consider RLAIF for automated preference data generation',
          ].map((rec, i) => (
            <div
              key={i}
              className="p-4 bg-gradient-to-r from-accent-warning/10 to-transparent rounded-lg border-l-2 border-accent-warning"
            >
              <p className="text-white/80">{rec}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

