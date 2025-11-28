'use client';

import { Gauge, RefreshCw, Zap, Clock, Server } from 'lucide-react';
import { getHfTrending, getModelsThatFit, getEfficiencyKernels } from '@/lib/api';
import { useApiQuery, getErrorMessage } from '@/lib/useApi';
import { EmptyState, ErrorState, LoadingState } from '@/components/DataState';

type InferenceDataset = {
  trending: any[];
  modelsThatFit: any[];
  efficiency: any;
};

export function InferenceTab() {
  const inferenceQuery = useApiQuery<InferenceDataset>('inference/data', async () => {
    const [trendingData, modelsData, effData] = await Promise.allSettled([
      getHfTrending(),
      getModelsThatFit(),
      getEfficiencyKernels(),
    ]);
    const trendingList =
      trendingData.status === 'fulfilled'
        ? (trendingData.value as any)?.models || trendingData.value || []
        : [];
    const modelsList =
      modelsData.status === 'fulfilled'
        ? (modelsData.value as any)?.models || modelsData.value || []
        : [];
    return {
      trending: trendingList,
      modelsThatFit: modelsList,
      efficiency: effData.status === 'fulfilled' ? effData.value : null,
    };
  });

  if (inferenceQuery.error) {
    return (
      <div className="card">
        <div className="card-body">
          <ErrorState
            message={getErrorMessage(inferenceQuery.error, 'Failed to load inference data')}
            onRetry={() => inferenceQuery.mutate()}
          />
        </div>
      </div>
    );
  }

  if (inferenceQuery.isLoading) {
    return (
      <div className="card">
        <div className="card-body">
          <LoadingState message="Loading inference data..." />
        </div>
      </div>
    );
  }

  const trending = inferenceQuery.data?.trending || [];
  const modelsThatFit = inferenceQuery.data?.modelsThatFit || [];
  const efficiency = inferenceQuery.data?.efficiency;

  const engines = [
    { name: 'vLLM', desc: 'PagedAttention, continuous batching', speedup: '24x', memory: '-50%' },
    { name: 'TensorRT-LLM', desc: 'NVIDIA optimized inference', speedup: '30x', memory: '-40%' },
    { name: 'TGI', desc: 'Text Generation Inference', speedup: '15x', memory: '-30%' },
    { name: 'SGLang', desc: 'RadixAttention, prefix caching', speedup: '20x', memory: '-45%' },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Gauge className="w-5 h-5 text-accent-info" />
            <h2 className="text-lg font-semibold text-white">Inference Optimization</h2>
          </div>
          <button
            onClick={() => inferenceQuery.mutate()}
            className="p-2 hover:bg-white/5 rounded-lg text-white/70 flex items-center gap-2"
            aria-label="Refresh inference data"
          >
            <RefreshCw className="w-4 h-4" />
            {inferenceQuery.isValidating && <span className="text-xs">Refreshing…</span>}
          </button>
        </div>
      </div>

      {/* Inference engines */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Server className="w-5 h-5 text-accent-primary" />
            <h3 className="font-medium text-white">Inference Engines</h3>
          </div>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {engines.map((engine, i) => (
              <div key={i} className="p-4 bg-white/5 rounded-lg border border-white/10">
                <h4 className="text-lg font-bold text-accent-info mb-1">{engine.name}</h4>
                <p className="text-sm text-white/50 mb-3">{engine.desc}</p>
                <div className="flex justify-between text-sm">
                  <span className="text-white/40">Throughput</span>
                  <span className="text-accent-success font-bold">{engine.speedup}</span>
                </div>
                <div className="flex justify-between text-sm mt-1">
                  <span className="text-white/40">Memory</span>
                  <span className="text-accent-primary font-bold">{engine.memory}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Optimization techniques */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-accent-warning" />
            <h3 className="font-medium text-white">Optimization Techniques</h3>
          </div>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[
              { name: 'Continuous Batching', impact: '3-5x throughput', desc: 'Dynamic batch scheduling' },
              { name: 'PagedAttention', impact: '24x throughput', desc: 'Virtual memory for KV cache' },
              { name: 'Flash Attention', impact: '2-4x speed', desc: 'Memory-efficient attention' },
              { name: 'Speculative Decoding', impact: '2-3x speed', desc: 'Draft model acceleration' },
              { name: 'INT8/FP8 Quantization', impact: '2x speed, 50% mem', desc: 'Reduced precision' },
              { name: 'KV Cache Compression', impact: '50% memory', desc: 'Compress cached keys/values' },
            ].map((tech, i) => (
              <div key={i} className="p-4 bg-white/5 rounded-lg flex items-center justify-between">
                <div>
                  <h4 className="font-medium text-white">{tech.name}</h4>
                  <p className="text-sm text-white/50">{tech.desc}</p>
                </div>
                <span className="text-accent-success font-bold text-sm">{tech.impact}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Models that fit */}
      <div className="card">
        <div className="card-header">
          <h3 className="font-medium text-white">Models That Fit Your GPU</h3>
        </div>
        <div className="card-body">
          {modelsThatFit.length === 0 ? (
            <EmptyState
              title="No models returned"
              description="Adjust filters or refresh to fetch fitting models."
              actionLabel="Refresh"
              onAction={() => inferenceQuery.mutate()}
            />
          ) : (
            <div className="space-y-2">
              {modelsThatFit.slice(0, 10).map((model, i) => (
                <div key={i} className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                  <span className="text-white">{model.name || model}</span>
                  {model.memory && (
                    <span className="text-sm text-white/50">{model.memory} VRAM</span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Trending models */}
      <div className="card">
        <div className="card-header">
          <h3 className="font-medium text-white">Trending on HuggingFace</h3>
        </div>
        <div className="card-body">
          {trending.length === 0 ? (
            <EmptyState
              title="No trending models"
              description="HuggingFace trending endpoint returned no data."
              actionLabel="Refresh"
              onAction={() => inferenceQuery.mutate()}
            />
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {trending.slice(0, 6).map((model: any, i: number) => (
                <div key={i} className="p-3 rounded-lg bg-white/5 border border-white/10">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-white font-semibold">{model.id || model.name || model.modelId}</div>
                      <div className="text-xs text-white/50">{model.library_name || model.framework || 'unknown'}</div>
                    </div>
                    {model.downloads && (
                      <div className="text-xs text-accent-primary text-right">
                        {model.downloads} downloads
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Efficiency */}
      {efficiency && (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">Kernel Efficiency</h3>
          </div>
          <div className="card-body grid grid-cols-1 md:grid-cols-3 gap-3">
            {(efficiency.kernels || efficiency.items || []).slice(0, 6).map((k: any, i: number) => (
              <div key={i} className="p-3 rounded-lg bg-white/5 border border-white/10">
                <div className="text-white font-semibold">{k.name || k.kernel || `Kernel ${i + 1}`}</div>
                <div className="text-sm text-white/50">{k.desc || k.category || 'General'}</div>
                {k.speedup && (
                  <div className="text-sm text-accent-success font-mono mt-1">{k.speedup}x</div>
                )}
              </div>
            ))}
            {(efficiency.kernels || efficiency.items || []).length === 0 && (
              <EmptyState
                title="No efficiency kernels"
                description="Run profiling to surface efficiency kernels."
                className="md:col-span-3"
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
}
