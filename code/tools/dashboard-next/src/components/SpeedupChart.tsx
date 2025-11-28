'use client';

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts';
import { Benchmark } from '@/types';
import { getSpeedupColor } from '@/lib/utils';

interface SpeedupChartProps {
  benchmarks: Benchmark[];
  height?: number;
  speedupCap?: number;
}

export function SpeedupChart({ benchmarks, height = 400, speedupCap }: SpeedupChartProps) {
  const data = benchmarks
    .filter((b) => b.status === 'succeeded' && b.speedup)
    .map((b) => ({
      name: b.name.length > 20 ? b.name.slice(0, 20) + '...' : b.name,
      fullName: b.name,
      speedup: b.speedup,
      rawSpeedup: b.raw_speedup ?? b.speedup,
      capped: b.speedup_capped,
      chapter: b.chapter,
    }))
    .sort((a, b) => b.speedup - a.speedup)
    .slice(0, 15);
  const hasCappedValues = data.some((d) => d.capped);

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <h3 className="text-lg font-semibold text-white">🚀 Top Speedups</h3>
          {hasCappedValues && speedupCap && (
            <span className="badge badge-warning">Capped at {speedupCap.toFixed(0)}x</span>
          )}
        </div>
        <span className="badge badge-info">{data.length} benchmarks</span>
      </div>
      <div className="card-body">
        <ResponsiveContainer width="100%" height={height}>
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 10, right: 30, left: 100, bottom: 10 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis
              type="number"
              tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
              axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
            />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 11 }}
              axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
              width={100}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(16, 16, 24, 0.95)',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '8px',
              }}
              formatter={(value: number, _name, props) => {
                const payload = (props as any)?.payload;
                const raw = payload?.rawSpeedup as number | undefined;
                const capped = payload?.capped && raw && raw !== value;
                const display = `${value.toFixed(2)}x`;
                return [capped ? `${display} (raw ${raw?.toFixed?.(2)}x)` : display, 'Speedup'];
              }}
              labelFormatter={(label) => data.find(d => d.name === label)?.fullName || label}
            />
            <ReferenceLine x={1} stroke="rgba(255,255,255,0.3)" strokeDasharray="3 3" />
            <Bar dataKey="speedup" radius={[0, 4, 4, 0]}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getSpeedupColor(entry.speedup)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

