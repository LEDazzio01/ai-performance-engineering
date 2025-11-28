'use client';

import { useState, useEffect } from 'react';
import { Server, Loader2, AlertTriangle, RefreshCw, Network, Cpu } from 'lucide-react';
import { getParallelismTopology, getParallelismPresets, getParallelismProfiles } from '@/lib/api';
import { NcclTuningCard } from '@/components/NcclTuningCard';
import { ParallelismExtrasCard } from '@/components/ParallelismExtrasCard';

export function DistributedTab() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [topology, setTopology] = useState<any>(null);
  const [presets, setPresets] = useState<any[]>([]);
  const [profiles, setProfiles] = useState<any[]>([]);

  async function loadData() {
    try {
      setLoading(true);
      setError(null);
      const [topoData, presetsData, profilesData] = await Promise.all([
        getParallelismTopology().catch(() => null),
        getParallelismPresets().catch(() => []),
        getParallelismProfiles().catch(() => []),
      ]);
      setTopology(topoData);
      const presetList = (presetsData as any)?.presets || presetsData || [];
      const profileList = (profilesData as any)?.profiles || profilesData || [];
      setPresets(presetList);
      setProfiles(profileList);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load distributed training data');
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
          <Loader2 className="w-8 h-8 animate-spin text-accent-danger" />
          <span className="ml-3 text-white/50">Loading distributed training data...</span>
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
            <Server className="w-5 h-5 text-accent-danger" />
            <h2 className="text-lg font-semibold text-white">Distributed Training</h2>
          </div>
          <button onClick={loadData} className="p-2 hover:bg-white/5 rounded-lg">
            <RefreshCw className="w-4 h-4 text-white/50" />
          </button>
        </div>
      </div>

      {/* Topology */}
      {topology && (
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <Network className="w-5 h-5 text-accent-info" />
              <h3 className="font-medium text-white">Cluster Topology</h3>
            </div>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="p-4 bg-white/5 rounded-lg">
                <div className="text-sm text-white/50 mb-1">Nodes</div>
                <div className="text-2xl font-bold text-accent-primary">
                  {topology.nodes || topology.num_nodes || 1}
                </div>
              </div>
              <div className="p-4 bg-white/5 rounded-lg">
                <div className="text-sm text-white/50 mb-1">GPUs per Node</div>
                <div className="text-2xl font-bold text-accent-secondary">
                  {topology.gpus_per_node || topology.devices_per_node || 8}
                </div>
              </div>
              <div className="p-4 bg-white/5 rounded-lg">
                <div className="text-sm text-white/50 mb-1">Total GPUs</div>
                <div className="text-2xl font-bold text-accent-success">
                  {topology.total_gpus || (topology.nodes || 1) * (topology.gpus_per_node || 8)}
                </div>
              </div>
              <div className="p-4 bg-white/5 rounded-lg">
                <div className="text-sm text-white/50 mb-1">Interconnect</div>
                <div className="text-lg font-bold text-accent-warning">
                  {topology.interconnect || 'NVLink'}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Presets */}
      {presets.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">Training Configurations</h3>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {presets.map((preset, i) => (
                <div
                  key={i}
                  className="p-4 bg-white/5 rounded-lg border border-white/10 hover:border-accent-primary/30 cursor-pointer transition-all"
                >
                  <h4 className="font-medium text-white mb-2">{preset.name}</h4>
                  <p className="text-sm text-white/50 mb-3">{preset.description}</p>
                  <div className="flex flex-wrap gap-2">
                    {preset.strategies?.map((s: string, j: number) => (
                      <span key={j} className="px-2 py-1 bg-accent-primary/20 text-accent-primary text-xs rounded">
                        {s}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Strategies */}
      <div className="card">
        <div className="card-header">
          <h3 className="font-medium text-white">Parallelism Strategies</h3>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[
              { name: 'Data Parallel (DDP)', desc: 'Replicate model across GPUs, split data', color: 'accent-primary' },
              { name: 'FSDP', desc: 'Shard model parameters and gradients', color: 'accent-secondary' },
              { name: 'Tensor Parallel', desc: 'Split individual layers across GPUs', color: 'accent-tertiary' },
              { name: 'Pipeline Parallel', desc: 'Split model layers across stages', color: 'accent-warning' },
              { name: 'Sequence Parallel', desc: 'Split sequence dimension', color: 'accent-info' },
              { name: 'DeepSpeed ZeRO', desc: 'Memory-efficient distributed training', color: 'accent-success' },
            ].map((strategy, i) => (
              <div key={i} className="p-4 bg-white/5 rounded-lg">
                <h4 className={`font-medium text-${strategy.color} mb-1`}>{strategy.name}</h4>
                <p className="text-sm text-white/50">{strategy.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <NcclTuningCard />
      <ParallelismExtrasCard />
    </div>
  );
}
