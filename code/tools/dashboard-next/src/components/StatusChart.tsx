'use client';

import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from 'recharts';
import { getStatusColor } from '@/lib/utils';

interface StatusChartProps {
  summary: {
    total: number;
    succeeded: number;
    failed: number;
    skipped: number;
  };
}

export function StatusChart({ summary }: StatusChartProps) {
  const chartData = [
    { name: 'Succeeded', value: summary.succeeded, color: getStatusColor('succeeded') },
    { name: 'Failed', value: summary.failed, color: getStatusColor('failed') },
    { name: 'Skipped', value: summary.skipped, color: getStatusColor('skipped') },
  ].filter((d) => d.value > 0);

  return (
    <div className="card">
      <div className="card-header">
        <h3 className="text-lg font-semibold text-white">📊 Status Distribution</h3>
      </div>
      <div className="card-body">
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={90}
              paddingAngle={2}
              dataKey="value"
            >
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(16, 16, 24, 0.95)',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '8px',
              }}
              formatter={(value: number, name: string) => [
                `${value} (${summary.total > 0 ? ((value / summary.total) * 100).toFixed(1) : '0'}%)`,
                name,
              ]}
            />
            <Legend
              verticalAlign="bottom"
              formatter={(value) => <span className="text-white/70 text-sm">{value}</span>}
            />
          </PieChart>
        </ResponsiveContainer>

        {/* Summary stats below chart */}
          <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t border-white/5">
            <div className="text-center">
              <p className="text-2xl font-bold text-accent-success">{summary.succeeded}</p>
              <p className="text-xs text-white/50">Passed</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-accent-danger">{summary.failed}</p>
              <p className="text-xs text-white/50">Failed</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-accent-warning">{summary.skipped}</p>
              <p className="text-xs text-white/50">Skipped</p>
            </div>
          </div>
      </div>
    </div>
  );
}

