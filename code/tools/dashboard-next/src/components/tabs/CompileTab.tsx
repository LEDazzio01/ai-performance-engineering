'use client';

import { useState, useEffect } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { Zap, Clock, TrendingUp, AlertCircle, CheckCircle, Loader2, RefreshCw } from 'lucide-react';
import { getProfilerCompile } from '@/lib/api';

export function CompileTab() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<any>(null);
  const [mode, setMode] = useState<'default' | 'reduce-overhead' | 'max-autotune'>('default');

  async function loadData() {
    try {
      setLoading(true);
      setError(null);
      const compileData = await getProfilerCompile();
      setData(compileData);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load compile data');
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
          <Loader2 className="w-8 h-8 animate-spin text-accent-warning" />
          <span className="ml-3 text-white/50">Loading compile data...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="card-body text-center py-16">
          <AlertCircle className="w-12 h-12 text-accent-danger mx-auto mb-4" />
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

  const functions = data?.functions || data?.results || [];
  const totalEager = functions.reduce((sum: number, d: any) => sum + (d.eager_ms || d.eager || 0), 0);
  const totalCompiled = functions.reduce((sum: number, d: any) => sum + (d.compiled_ms || d.compiled || 0), 0);
  const overallSpeedup = totalCompiled > 0 ? totalEager / totalCompiled : 0;
  const totalBreaks = functions.reduce((sum: number, d: any) => sum + (d.breaks || d.graph_breaks || 0), 0);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-accent-warning" />
            <h2 className="text-lg font-semibold text-white">torch.compile Analysis</h2>
          </div>
          <div className="flex items-center gap-2">
            {['default', 'reduce-overhead', 'max-autotune'].map((m) => (
              <button
                key={m}
                onClick={() => setMode(m as typeof mode)}
                className={`px-3 py-1.5 rounded-lg text-sm transition-all ${
                  mode === m
                    ? 'bg-accent-warning/20 text-accent-warning'
                    : 'text-white/50 hover:text-white'
                }`}
              >
                {m}
              </button>
            ))}
            <button onClick={loadData} className="p-2 hover:bg-white/5 rounded-lg">
              <RefreshCw className="w-4 h-4 text-white/50" />
            </button>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card p-5">
          <div className="flex items-center gap-2 text-sm text-white/50 mb-2">
            <TrendingUp className="w-4 h-4" />
            Overall Speedup
          </div>
          <div className="text-3xl font-bold text-accent-success">
            {overallSpeedup.toFixed(2)}x
          </div>
        </div>
        <div className="card p-5">
          <div className="flex items-center gap-2 text-sm text-white/50 mb-2">
            <Clock className="w-4 h-4" />
            Eager Time
          </div>
          <div className="text-2xl font-bold text-accent-tertiary">
            {totalEager.toFixed(1)}ms
          </div>
        </div>
        <div className="card p-5">
          <div className="flex items-center gap-2 text-sm text-white/50 mb-2">
            <Zap className="w-4 h-4" />
            Compiled Time
          </div>
          <div className="text-2xl font-bold text-accent-primary">
            {totalCompiled.toFixed(1)}ms
          </div>
        </div>
        <div className="card p-5">
          <div className="flex items-center gap-2 text-sm text-white/50 mb-2">
            {totalBreaks > 0 ? (
              <AlertCircle className="w-4 h-4 text-accent-warning" />
            ) : (
              <CheckCircle className="w-4 h-4 text-accent-success" />
            )}
            Graph Breaks
          </div>
          <div
            className={`text-2xl font-bold ${
              totalBreaks > 0 ? 'text-accent-warning' : 'text-accent-success'
            }`}
          >
            {totalBreaks}
          </div>
        </div>
      </div>

      {/* Comparison chart */}
      {functions.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">Eager vs Compiled Performance</h3>
          </div>
          <div className="card-body">
            <div className="h-[350px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={functions}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 11 }}
                    axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                  />
                  <YAxis
                    tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
                    axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                    unit="ms"
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(16, 16, 24, 0.95)',
                      border: '1px solid rgba(255,255,255,0.1)',
                      borderRadius: '8px',
                    }}
                    formatter={(value: number) => [`${value.toFixed(2)}ms`, '']}
                  />
                  <Legend />
                  <Bar
                    dataKey="eager_ms"
                    name="Eager"
                    fill="#f72585"
                    radius={[4, 4, 0, 0]}
                  />
                  <Bar
                    dataKey="compiled_ms"
                    name="Compiled"
                    fill="#00f5d4"
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Details table */}
      {functions.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">Function Details</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/5">
                  <th className="px-5 py-3 text-left text-xs font-medium text-white/50 uppercase">
                    Function
                  </th>
                  <th className="px-5 py-3 text-right text-xs font-medium text-white/50 uppercase">
                    Eager
                  </th>
                  <th className="px-5 py-3 text-right text-xs font-medium text-white/50 uppercase">
                    Compiled
                  </th>
                  <th className="px-5 py-3 text-right text-xs font-medium text-white/50 uppercase">
                    Speedup
                  </th>
                  <th className="px-5 py-3 text-center text-xs font-medium text-white/50 uppercase">
                    Graphs
                  </th>
                  <th className="px-5 py-3 text-center text-xs font-medium text-white/50 uppercase">
                    Breaks
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {functions.map((row: any, i: number) => {
                  const eager = row.eager_ms || row.eager || 0;
                  const compiled = row.compiled_ms || row.compiled || 0;
                  const speedup = compiled > 0 ? eager / compiled : 0;
                  return (
                    <tr key={i} className="hover:bg-white/[0.02]">
                      <td className="px-5 py-4 font-mono text-sm text-white">{row.name}</td>
                      <td className="px-5 py-4 text-right font-mono text-sm text-accent-tertiary">
                        {eager.toFixed(2)}ms
                      </td>
                      <td className="px-5 py-4 text-right font-mono text-sm text-accent-primary">
                        {compiled.toFixed(2)}ms
                      </td>
                      <td className="px-5 py-4 text-right">
                        <span className="font-bold text-accent-success">{speedup.toFixed(2)}x</span>
                      </td>
                      <td className="px-5 py-4 text-center text-white/70">{row.graphs || 1}</td>
                      <td className="px-5 py-4 text-center">
                        <span
                          className={
                            (row.breaks || 0) > 0 ? 'text-accent-warning font-bold' : 'text-white/40'
                          }
                        >
                          {row.breaks || 0}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
