'use client';

import { useEffect, useState } from 'react';
import { BatteryCharging, Gauge, RefreshCw, Shield, Power } from 'lucide-react';
import { getGpuControl, setGpuPowerLimit, setGpuClockPin, setGpuPersistence, applyGpuPreset } from '@/lib/api';
import { useToast } from './Toast';
import { formatBytes } from '@/lib/utils';

interface GpuControlPanelProps {
  gpuInfo?: any;
}

export function GpuControlPanel({ gpuInfo }: GpuControlPanelProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [control, setControl] = useState<any>(null);
  const [powerLimit, setPowerLimitValue] = useState<number | null>(null);
  const [pinClocks, setPinClocks] = useState(false);
  const [persistence, setPersistence] = useState(false);
  const { showToast } = useToast();

  async function loadControl() {
    setError(null);
    try {
      const data = await getGpuControl();
      const ctrl: any = data || {};
      setControl(ctrl);
      if (ctrl?.power?.limit) {
        setPowerLimitValue(Math.round(ctrl.power.limit));
      }
      if (ctrl?.clocks_locked !== undefined) setPinClocks(ctrl.clocks_locked);
      if (ctrl?.persistence_mode !== undefined) setPersistence(ctrl.persistence_mode);
    } catch (e) {
      setError('GPU control not available (requires NVIDIA drivers)');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadControl();
  }, []);

  const handlePowerLimitChange = async () => {
    if (!powerLimit) return;
    try {
      await setGpuPowerLimit(powerLimit);
      showToast(`Power limit set to ${powerLimit}W`, 'success');
      loadControl();
    } catch {
      showToast('Failed to set power limit (may require sudo)', 'error');
    }
  };

  const handlePinClocks = async (pin: boolean) => {
    setPinClocks(pin);
    try {
      const res = await setGpuClockPin(pin);
      if ((res as any).success) {
        showToast(pin ? 'Clocks pinned to max' : 'Clock pinning disabled', 'success');
      } else {
        showToast('Clock pinning may require sudo', 'error');
        setPinClocks(!pin);
      }
    } catch {
      showToast('Clock pinning may require sudo', 'error');
      setPinClocks(!pin);
    }
  };

  const handlePersistence = async (enabled: boolean) => {
    setPersistence(enabled);
    try {
      const res = await setGpuPersistence(enabled);
      if (!(res as any).success) {
        setPersistence(!enabled);
        showToast('Persistence mode may require sudo', 'error');
      } else {
        showToast(`Persistence ${enabled ? 'enabled' : 'disabled'}`, 'success');
      }
    } catch {
      setPersistence(!enabled);
      showToast('Persistence mode may require sudo', 'error');
    }
  };

  const handlePreset = async (preset: 'max' | 'balanced' | 'quiet') => {
    try {
      const res = await applyGpuPreset(preset);
      if ((res as any).success) {
        showToast(`${preset} preset applied`, 'success');
      } else {
        showToast('Preset may require sudo; see terminal for commands', 'info');
      }
      loadControl();
    } catch {
      showToast('Preset failed (check permissions)', 'error');
    }
  };

  const powerDraw = gpuInfo?.power_draw ?? control?.power?.current;
  const powerLimitDisplay = control?.power?.limit || gpuInfo?.power_limit;
  const memTotal = gpuInfo?.memory_total ? formatBytes(gpuInfo.memory_total * 1e9) : null;
  const memUsed = gpuInfo?.memory_used ? formatBytes(gpuInfo.memory_used * 1e9) : null;
  const memUtil = gpuInfo?.memory_used && gpuInfo?.memory_total ? (gpuInfo.memory_used / gpuInfo.memory_total) * 100 : null;

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <BatteryCharging className="w-5 h-5 text-accent-warning" />
          <h3 className="font-medium text-white">GPU Power & Clocks</h3>
        </div>
        <button onClick={loadControl} className="p-2 hover:bg-white/5 rounded-lg">
          <RefreshCw className="w-4 h-4 text-white/50" />
        </button>
      </div>
      <div className="card-body space-y-4">
        {loading ? (
          <div className="text-white/50">Loading GPU control...</div>
        ) : error ? (
          <div className="text-sm text-accent-warning">{error}</div>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 bg-white/5 rounded-lg">
                <div className="text-sm text-white/50 mb-1">Power Draw</div>
                <div className="text-2xl font-bold text-accent-warning">{powerDraw ? `${powerDraw}W` : '—'}</div>
                <div className="text-xs text-white/40">Limit: {powerLimitDisplay ? `${powerLimitDisplay}W` : '—'}</div>
              </div>
              <div className="p-4 bg-white/5 rounded-lg">
                <div className="text-sm text-white/50 mb-1">Graphics Clock</div>
                <div className="text-2xl font-bold text-accent-primary">
                  {control?.clocks?.graphics ? `${control.clocks.graphics} MHz` : '—'}
                </div>
                <div className="text-xs text-white/40">Max {control?.clocks?.graphics_max || '—'} MHz</div>
              </div>
              <div className="p-4 bg-white/5 rounded-lg">
                <div className="text-sm text-white/50 mb-1">Memory</div>
                <div className="text-2xl font-bold text-accent-success">
                  {memUsed && memTotal ? `${memUsed} / ${memTotal}` : '—'}
                </div>
                {memUtil !== null && (
                  <div className="h-2 bg-white/10 rounded-full mt-2 overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-accent-success to-accent-primary" style={{ width: `${memUtil.toFixed(0)}%` }} />
                  </div>
                )}
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="text-sm text-white/60">Power limit (W)</div>
                <div className="text-sm text-white">{powerLimit ?? '—'}</div>
              </div>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min={150}
                  max={control?.power?.max_limit || 500}
                  value={powerLimit ?? control?.power?.limit ?? 200}
                  onChange={(e) => setPowerLimitValue(Number(e.target.value))}
                  className="flex-1 accent-accent-primary"
                />
                <button
                  onClick={handlePowerLimitChange}
                  className="px-4 py-2 bg-accent-primary/20 text-accent-primary rounded-lg font-medium"
                >
                  Apply
                </button>
              </div>
            </div>

            <div className="flex flex-wrap gap-3">
              <label className="flex items-center gap-2 px-3 py-2 bg-white/5 rounded-lg text-white/80 cursor-pointer">
                <input
                  type="checkbox"
                  checked={pinClocks}
                  onChange={(e) => handlePinClocks(e.target.checked)}
                  className="w-4 h-4 accent-accent-primary"
                />
                <span className="flex items-center gap-1">
                  <Gauge className="w-4 h-4" /> Pin max clocks
                </span>
              </label>

              <label className="flex items-center gap-2 px-3 py-2 bg-white/5 rounded-lg text-white/80 cursor-pointer">
                <input
                  type="checkbox"
                  checked={persistence}
                  onChange={(e) => handlePersistence(e.target.checked)}
                  className="w-4 h-4 accent-accent-primary"
                />
                <span className="flex items-center gap-1">
                  <Shield className="w-4 h-4" /> Persistence mode
                </span>
              </label>
            </div>

            <div className="flex flex-wrap gap-2">
              {[
                { id: 'max' as const, label: 'Max Perf', color: 'from-accent-danger/40 to-accent-warning/40' },
                { id: 'balanced' as const, label: 'Balanced', color: 'from-accent-primary/30 to-accent-secondary/30' },
                { id: 'quiet' as const, label: 'Quiet', color: 'from-white/10 to-white/5' },
              ].map((preset) => (
                <button
                  key={preset.id}
                  onClick={() => handlePreset(preset.id)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium bg-gradient-to-r ${preset.color} text-white`}
                >
                  {preset.label}
                </button>
              ))}
            </div>

            <p className="text-xs text-white/40 flex items-center gap-2">
              <Power className="w-4 h-4" />
              GPU control mirrors the original dashboard: operations may require sudo on some systems.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
