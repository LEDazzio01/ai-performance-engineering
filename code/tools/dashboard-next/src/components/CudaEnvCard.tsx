'use client';

import { useEffect, useState } from 'react';
import { Cpu, RefreshCw } from 'lucide-react';
import { getCudaEnvironment } from '@/lib/api';

export function CudaEnvCard() {
  const [env, setEnv] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  async function load() {
    try {
      setLoading(true);
      const data = await getCudaEnvironment();
      setEnv(data);
    } catch {
      setEnv(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  const items = [
    { label: 'CUDA Version', value: env?.cuda_version },
    { label: 'cuDNN Version', value: env?.cudnn_version },
    { label: 'PyTorch', value: env?.pytorch_version },
    { label: 'CUDA_VISIBLE_DEVICES', value: env?.cuda_visible_devices || 'All' },
    { label: 'TORCH_COMPILE_DEBUG', value: env?.torch_compile_debug || 'Not set' },
    { label: 'TF32 Enabled', value: env?.tf32_enabled ? 'Yes' : 'No' },
    { label: 'Flash Attention', value: env?.flash_attention ? 'Available' : 'Unavailable' },
    { label: 'Deterministic', value: env?.deterministic ? 'Enabled' : 'Disabled' },
  ];

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <Cpu className="w-5 h-5 text-accent-info" />
          <h3 className="font-medium text-white">CUDA Environment</h3>
        </div>
        <button onClick={load} className="p-2 hover:bg-white/5 rounded-lg">
          <RefreshCw className="w-4 h-4 text-white/50" />
        </button>
      </div>
      <div className="card-body">
        {loading ? (
          <div className="text-sm text-white/50">Inspecting environment...</div>
        ) : env ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {items.map((item) => (
              <div key={item.label} className="p-3 rounded-lg bg-white/5 border border-white/10">
                <div className="text-xs text-white/40 uppercase mb-1">{item.label}</div>
                <div className="text-white">{item.value || 'N/A'}</div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-sm text-white/50">CUDA info unavailable</div>
        )}
      </div>
    </div>
  );
}
