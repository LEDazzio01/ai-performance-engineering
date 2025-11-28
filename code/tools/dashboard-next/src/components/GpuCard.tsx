'use client';

import { GpuInfo } from '@/types';
import { formatBytes } from '@/lib/utils';
import { Cpu, Thermometer, Zap, HardDrive } from 'lucide-react';

interface GpuCardProps {
  gpu: GpuInfo;
}

export function GpuCard({ gpu }: GpuCardProps) {
  const memoryPercent = (gpu.memory_used / gpu.memory_total) * 100;
  const powerPercent = (gpu.power_draw / gpu.power_limit) * 100;

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-gradient-to-br from-accent-secondary to-accent-tertiary">
            <Cpu className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="font-semibold text-white">{gpu.name}</h3>
            <p className="text-xs text-white/50">
              CUDA {gpu.cuda_version} • CC {gpu.compute_capability}
            </p>
          </div>
        </div>
      </div>
      <div className="card-body space-y-4">
        {/* Memory */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2 text-sm text-white/70">
              <HardDrive className="w-4 h-4" />
              <span>Memory</span>
            </div>
            <span className="text-sm font-medium text-white">
              {formatBytes(gpu.memory_used * 1e9)} / {formatBytes(gpu.memory_total * 1e9)}
            </span>
          </div>
          <div className="h-2 bg-white/10 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-accent-primary to-accent-secondary rounded-full transition-all duration-500"
              style={{ width: `${memoryPercent}%` }}
            />
          </div>
        </div>

        {/* Utilization */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2 text-sm text-white/70">
              <Cpu className="w-4 h-4" />
              <span>Utilization</span>
            </div>
            <span className="text-sm font-medium text-white">{gpu.utilization}%</span>
          </div>
          <div className="h-2 bg-white/10 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-accent-success to-accent-primary rounded-full transition-all duration-500"
              style={{ width: `${gpu.utilization}%` }}
            />
          </div>
        </div>

        {/* Temperature & Power */}
        <div className="grid grid-cols-2 gap-4 pt-2 border-t border-white/5">
          <div className="flex items-center gap-2">
            <Thermometer className="w-4 h-4 text-accent-warning" />
            <div>
              <p className="text-lg font-bold text-white">{gpu.temperature}°C</p>
              <p className="text-xs text-white/50">Temperature</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Zap className="w-4 h-4 text-accent-info" />
            <div>
              <p className="text-lg font-bold text-white">{gpu.power_draw}W</p>
              <p className="text-xs text-white/50">Power</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}


