'use client';

import { useEffect, useState, useRef } from 'react';
import { Thermometer, Activity, Wind, Power } from 'lucide-react';
import { getGpuInfo } from '@/lib/api';

interface Reading {
  temperature: number;
  power: number;
  utilization: number;
  memoryUtil: number;
  fan: number;
}

export function GpuThermalMonitor() {
  const [reading, setReading] = useState<Reading | null>(null);
  const [running, setRunning] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const load = async () => {
    try {
      const data: any = await getGpuInfo();
      setReading({
        temperature: data.temperature || 0,
        power: data.power_draw || 0,
        utilization: data.utilization || 0,
        memoryUtil:
          data.memory_total && data.memory_used
            ? (data.memory_used / data.memory_total) * 100
            : 0,
        fan: data.fan_speed || 0,
      });
    } catch {
      // ignore errors; keep last reading
    }
  };

  const toggle = () => {
    if (running) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      setRunning(false);
    } else {
      load();
      intervalRef.current = setInterval(load, 1000);
      setRunning(true);
    }
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <Thermometer className="w-5 h-5 text-accent-warning" />
          <h3 className="font-medium text-white">Thermal & Power Monitor</h3>
        </div>
        <button
          onClick={toggle}
          className={`px-3 py-1.5 rounded-lg text-sm ${
            running
              ? 'bg-accent-danger/20 text-accent-danger border border-accent-danger/30'
              : 'bg-accent-success/20 text-accent-success border border-accent-success/30'
          }`}
        >
          {running ? 'Stop' : 'Start'}
        </button>
      </div>
      <div className="card-body space-y-3">
        {reading ? (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <Stat label="Temp" value={`${reading.temperature}°C`} icon={<Thermometer className="w-4 h-4" />} />
              <Stat label="Power" value={`${reading.power}W`} icon={<Power className="w-4 h-4" />} />
              <Stat label="GPU Util" value={`${reading.utilization}%`} icon={<Activity className="w-4 h-4" />} />
              <Stat label="Fan" value={`${reading.fan}%`} icon={<Wind className="w-4 h-4" />} />
            </div>
            <div>
              <div className="flex items-center justify-between text-xs text-white/60 mb-1">
                <span>Temperature</span>
                <span>{reading.temperature}°C</span>
              </div>
              <Bar value={Math.min(100, (reading.temperature / 90) * 100)} color={reading.temperature > 80 ? '#ff4757' : reading.temperature > 65 ? '#ffc43d' : '#00f5a0'} />
              <div className="flex items-center justify-between text-xs text-white/60 mt-3 mb-1">
                <span>Power</span>
                <span>{reading.power}W</span>
              </div>
              <Bar value={Math.min(100, reading.power / 500 * 100)} color="#ffc43d" />
              <div className="flex items-center justify-between text-xs text-white/60 mt-3 mb-1">
                <span>Perf / Watt (est.)</span>
                <span>{reading.power > 0 ? (reading.utilization / reading.power).toFixed(2) : '--'}</span>
              </div>
              <Bar value={Math.min(100, reading.utilization)} color="#00f5d4" />
            </div>
          </>
        ) : (
          <div className="text-sm text-white/50">Press start to sample GPU sensors.</div>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value, icon }: { label: string; value: string; icon: React.ReactNode }) {
  return (
    <div className="p-3 rounded-lg bg-white/5 border border-white/10 flex items-center gap-3">
      <div className="w-8 h-8 rounded-full bg-white/10 flex items-center justify-center text-white/70">
        {icon}
      </div>
      <div>
        <div className="text-xs text-white/50">{label}</div>
        <div className="text-white font-semibold">{value}</div>
      </div>
    </div>
  );
}

function Bar({ value, color }: { value: number; color: string }) {
  return (
    <div className="h-2 bg-white/10 rounded-full overflow-hidden">
      <div
        className="h-full rounded-full transition-all"
        style={{ width: `${value.toFixed(0)}%`, background: color }}
      />
    </div>
  );
}
