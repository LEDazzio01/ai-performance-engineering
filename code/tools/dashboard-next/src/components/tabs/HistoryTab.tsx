'use client';

import { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { History, TrendingUp, AlertTriangle, Trophy, Loader2, RefreshCw } from 'lucide-react';
import { getHistoryRuns, getHistoryTrends, getHistorySummary } from '@/lib/api';

export function HistoryTab() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [runs, setRuns] = useState<any>(null);
  const [trends, setTrends] = useState<any>(null);
  const [summary, setSummary] = useState<any>(null);
  const [timeRange, setTimeRange] = useState(30);

  async function loadData() {
    try {
      setLoading(true);
      setError(null);
      const [runsData, trendsData, summaryData] = await Promise.all([
        getHistoryRuns(),
        getHistoryTrends(),
        getHistorySummary().catch(() => null),
      ]);
      setRuns(runsData);
      setTrends(trendsData);
      setSummary(summaryData);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load history');
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
          <Loader2 className="w-8 h-8 animate-spin text-accent-primary" />
          <span className="ml-3 text-white/50">Loading history...</span>
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

  const runsList = runs?.runs || [];
  const trendData = trends?.by_date || trends?.history || [];
  const bestEver = trends?.best_ever?.speedup || Math.max(...trendData.map((d: any) => d.max_speedup || 0), 0);
  const regressions = trends?.regressions || [];
  const improvements = trends?.improvements || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <History className="w-5 h-5 text-accent-primary" />
            <h2 className="text-lg font-semibold text-white">Performance History</h2>
          </div>
          <div className="flex items-center gap-2">
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(Number(e.target.value))}
              className="px-3 py-1.5 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
            >
              <option value={7}>Last 7 days</option>
              <option value={14}>Last 14 days</option>
              <option value={30}>Last 30 days</option>
            </select>
            <button onClick={loadData} className="p-2 hover:bg-white/5 rounded-lg">
              <RefreshCw className="w-4 h-4 text-white/50" />
            </button>
          </div>
      </div>
    </div>

      {summary && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="card p-5">
            <div className="text-sm text-white/50 mb-1">Total Runs</div>
            <div className="text-2xl font-bold text-accent-primary">{summary.runs?.length || summary.total || 0}</div>
          </div>
          <div className="card p-5">
            <div className="text-sm text-white/50 mb-1">Avg Speedup</div>
            <div className="text-2xl font-bold text-accent-secondary">
              {summary.trends?.avg_speedup?.toFixed?.(2) || summary.avg_speedup || '—'}x
            </div>
          </div>
          <div className="card p-5">
            <div className="text-sm text-white/50 mb-1">Last Run</div>
            <div className="text-xl font-bold text-white">{summary.runs?.[0]?.date || summary.latest || 'N/A'}</div>
          </div>
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card p-5">
          <div className="flex items-center gap-2 text-sm text-white/50 mb-2">
            <TrendingUp className="w-4 h-4" />
            Current Avg Speedup
          </div>
          <div className="text-2xl font-bold text-accent-primary">
            {trendData.length > 0 ? trendData[trendData.length - 1]?.avg_speedup?.toFixed(2) || 'N/A' : 'N/A'}x
          </div>
        </div>
        <div className="card p-5">
          <div className="flex items-center gap-2 text-sm text-white/50 mb-2">
            <Trophy className="w-4 h-4" />
            Best Ever
          </div>
          <div className="text-2xl font-bold text-accent-warning">
            {bestEver?.toFixed?.(2) || bestEver || 'N/A'}x
          </div>
        </div>
        <div className="card p-5">
          <div className="flex items-center gap-2 text-sm text-white/50 mb-2">
            <History className="w-4 h-4" />
            Total Runs
          </div>
          <div className="text-2xl font-bold text-white">
            {runs?.total_runs || runsList.length}
          </div>
        </div>
        <div className="card p-5">
          <div className="flex items-center gap-2 text-sm text-white/50 mb-2">
            <AlertTriangle className="w-4 h-4" />
            Regressions
          </div>
          <div
            className={`text-2xl font-bold ${
              regressions.length > 0 ? 'text-accent-danger' : 'text-accent-success'
            }`}
          >
            {regressions.length}
          </div>
        </div>
      </div>

      {/* Trend chart */}
      {trendData.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">Performance Trend</h3>
          </div>
          <div className="card-body">
            <div className="h-[350px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trendData.slice(-timeRange)} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis
                    dataKey="date"
                    tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 11 }}
                    axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                    tickFormatter={(v) => v?.slice?.(5) || v}
                  />
                  <YAxis
                    tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
                    axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                    domain={['auto', 'auto']}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(16, 16, 24, 0.95)',
                      border: '1px solid rgba(255,255,255,0.1)',
                      borderRadius: '8px',
                    }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="avg_speedup"
                    name="Avg Speedup"
                    stroke="#00f5d4"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="max_speedup"
                    name="Max Speedup"
                    stroke="#9d4edd"
                    strokeWidth={2}
                    dot={false}
                    strokeDasharray="5 5"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* History table */}
      {runsList.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">Run History</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/5">
                  <th className="px-5 py-3 text-left text-xs font-medium text-white/50 uppercase">
                    Date
                  </th>
                  <th className="px-5 py-3 text-right text-xs font-medium text-white/50 uppercase">
                    Benchmarks
                  </th>
                  <th className="px-5 py-3 text-right text-xs font-medium text-white/50 uppercase">
                    Avg Speedup
                  </th>
                  <th className="px-5 py-3 text-right text-xs font-medium text-white/50 uppercase">
                    Max Speedup
                  </th>
                  <th className="px-5 py-3 text-center text-xs font-medium text-white/50 uppercase">
                    Status
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {runsList.slice(0, 10).map((row: any, i: number) => (
                  <tr key={i} className="hover:bg-white/[0.02]">
                    <td className="px-5 py-4 text-white">{row.date}</td>
                    <td className="px-5 py-4 text-right text-white/70">{row.benchmark_count || row.total}</td>
                    <td className="px-5 py-4 text-right font-bold text-accent-primary">
                      {row.avg_speedup?.toFixed?.(2) || row.avg_speedup}x
                    </td>
                    <td className="px-5 py-4 text-right font-bold text-accent-secondary">
                      {row.max_speedup?.toFixed?.(2) || row.max_speedup}x
                    </td>
                    <td className="px-5 py-4 text-center">
                      <span className="text-accent-success">✓ {row.successful || row.passed}/{row.benchmark_count || row.total}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
