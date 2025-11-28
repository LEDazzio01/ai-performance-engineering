'use client';

import { useMemo } from 'react';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Cell,
} from 'recharts';
import { PieChart as PieChartIcon, BarChart3, Target, TrendingUp, Loader2, AlertTriangle, Award } from 'lucide-react';
import { Benchmark } from '@/types';
import { OptimizationStackingCard } from '@/components/OptimizationStackingCard';
import { getAnalysisLeaderboards, getAnalysisPareto, getAnalysisBottlenecks } from '@/lib/api';
import { EmptyState, ErrorState, LoadingState } from '@/components/DataState';
import { getErrorMessage, useApiQuery } from '@/lib/useApi';

interface AnalysisTabProps {
  benchmarks: Benchmark[];
}

export function AnalysisTab({ benchmarks }: AnalysisTabProps) {
  const remoteQuery = useApiQuery('analysis/remote', async () => {
    const [lb, p, b] = await Promise.allSettled([
      getAnalysisLeaderboards(),
      getAnalysisPareto(),
      getAnalysisBottlenecks(),
    ]);

    return {
      leaderboards: lb.status === 'fulfilled' ? lb.value : null,
      pareto: p.status === 'fulfilled' ? p.value : null,
      bottlenecks: b.status === 'fulfilled' ? b.value : null,
    };
  });

  const leaderboards = remoteQuery.data?.leaderboards;
  const pareto = remoteQuery.data?.pareto;
  const bottlenecks = remoteQuery.data?.bottlenecks;

  const succeededWithSpeedup = benchmarks.filter(
    (b) => b.status === 'succeeded' && typeof b.speedup === 'number'
  );
  const overallAvgSpeedup =
    succeededWithSpeedup.length > 0
      ? succeededWithSpeedup.reduce((sum, b) => sum + (b.speedup || 0), 0) / succeededWithSpeedup.length
      : 0;

  // Calculate analysis data
  const chapters = Array.from(new Set(benchmarks.map((b) => b.chapter)));
  const chapterStats = chapters.map((chapter) => {
    const chapterBenchmarks = benchmarks.filter((b) => b.chapter === chapter);
    const succeeded = chapterBenchmarks.filter((b) => b.status === 'succeeded');
    const avgSpeedup =
      succeeded.length > 0
        ? succeeded.reduce((sum, b) => sum + (b.speedup || 0), 0) / succeeded.length
        : 0;
    return {
      chapter,
      total: chapterBenchmarks.length,
      succeeded: succeeded.length,
      avgSpeedup,
      successRate: (succeeded.length / chapterBenchmarks.length) * 100,
    };
  });

  // Radar chart data for performance dimensions
  const radarData = [
    { dimension: 'Speedup', value: 85 },
    { dimension: 'Memory Efficiency', value: 72 },
    { dimension: 'Compute Utilization', value: 78 },
    { dimension: 'Parallelism', value: 65 },
    { dimension: 'Cache Efficiency', value: 70 },
    { dimension: 'I/O Throughput', value: 88 },
  ];

  // Stacked analysis by type
  const typeStats = ['micro', 'kernel', 'e2e'].map((type) => {
    const typeBenchmarks = benchmarks.filter((b) => b.type === type);
    const succeeded = typeBenchmarks.filter((b) => b.status === 'succeeded');
    const failed = typeBenchmarks.filter((b) => b.status === 'failed');
    return {
      type: type.charAt(0).toUpperCase() + type.slice(1),
      succeeded: succeeded.length,
      failed: failed.length,
      avgSpeedup: succeeded.length > 0
        ? succeeded.reduce((sum, b) => sum + (b.speedup || 0), 0) / succeeded.length
        : 0,
    };
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <PieChartIcon className="w-5 h-5 text-accent-primary" />
            <h2 className="text-lg font-semibold text-white">Multi-Dimensional Analysis</h2>
          </div>
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card p-5">
          <div className="text-sm text-white/50 mb-2">Total Benchmarks</div>
          <div className="text-3xl font-bold text-accent-primary">{benchmarks.length}</div>
        </div>
        <div className="card p-5">
          <div className="text-sm text-white/50 mb-2">Chapters</div>
          <div className="text-3xl font-bold text-accent-secondary">{chapters.length}</div>
        </div>
        <div className="card p-5">
          <div className="text-sm text-white/50 mb-2">Overall Success</div>
          <div className="text-3xl font-bold text-accent-success">
            {benchmarks.length > 0
              ? ((benchmarks.filter((b) => b.status === 'succeeded').length / benchmarks.length) * 100).toFixed(0)
              : '0'}
            %
          </div>
        </div>
        <div className="card p-5">
          <div className="text-sm text-white/50 mb-2">Avg Speedup</div>
          <div className="text-3xl font-bold text-accent-warning">
            {overallAvgSpeedup.toFixed(2)}x
          </div>
        </div>
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Radar chart */}
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <Target className="w-5 h-5 text-accent-secondary" />
              <h3 className="font-medium text-white">Performance Dimensions</h3>
            </div>
          </div>
          <div className="card-body">
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData}>
                  <PolarGrid stroke="rgba(255,255,255,0.1)" />
                  <PolarAngleAxis
                    dataKey="dimension"
                    tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 11 }}
                  />
                  <PolarRadiusAxis
                    angle={30}
                    domain={[0, 100]}
                    tick={{ fill: 'rgba(255,255,255,0.3)', fontSize: 10 }}
                  />
                  <Radar
                    name="Performance"
                    dataKey="value"
                    stroke="#00f5d4"
                    fill="#00f5d4"
                    fillOpacity={0.3}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Type breakdown */}
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-accent-warning" />
              <h3 className="font-medium text-white">By Benchmark Type</h3>
            </div>
          </div>
          <div className="card-body">
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={typeStats}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis
                    dataKey="type"
                    tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
                    axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                  />
                  <YAxis
                    tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
                    axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(16, 16, 24, 0.95)',
                      border: '1px solid rgba(255,255,255,0.1)',
                      borderRadius: '8px',
                    }}
                  />
                  <Legend />
                  <Bar dataKey="succeeded" name="Succeeded" fill="#00f5a0" stackId="a" />
                  <Bar dataKey="failed" name="Failed" fill="#ff4757" stackId="a" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* Chapter breakdown */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-accent-success" />
            <h3 className="font-medium text-white">Chapter Performance</h3>
          </div>
        </div>
        <div className="card-body">
          <div className="space-y-4">
            {chapterStats.map((stat, i) => (
              <div key={i} className="flex items-center gap-4">
                <div className="w-20 text-sm text-white/70">{stat.chapter}</div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <div className="flex-1 h-3 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-accent-success to-accent-primary rounded-full"
                        style={{ width: `${stat.successRate}%` }}
                      />
                    </div>
                    <span className="text-sm text-white/50 w-12">{stat.successRate.toFixed(0)}%</span>
                  </div>
                </div>
                <div className="text-right w-24">
                  <div className="text-lg font-bold text-accent-primary">
                    {stat.avgSpeedup.toFixed(2)}x
                  </div>
                  <div className="text-xs text-white/40">
                    {stat.succeeded}/{stat.total}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <Award className="w-5 h-5 text-accent-primary" />
              <h3 className="font-medium text-white">Leaderboards</h3>
            </div>
            {remoteQuery.isValidating && <Loader2 className="w-4 h-4 animate-spin text-accent-primary" />}
          </div>
          <div className="card-body space-y-3">
            {remoteQuery.error ? (
              <ErrorState
                message={getErrorMessage(remoteQuery.error, 'Failed to load leaderboards')}
                onRetry={() => remoteQuery.mutate()}
              />
            ) : remoteQuery.isLoading ? (
              <LoadingState inline message="Loading leaderboards..." />
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {Object.entries(leaderboards || {}).slice(0, 4).map(([name, entries], idx) => {
                  const list = (entries as any[]) || [];
                  return (
                    <div key={idx} className="p-3 rounded-lg bg-white/5 border border-white/10 space-y-1">
                      <div className="text-xs text-white/50 uppercase">{name}</div>
                      {list.slice(0, 3).map((item: any, i: number) => (
                        <div key={i} className="flex items-center justify-between text-sm">
                          <span className="text-white/80 truncate">{item.name || item.benchmark || `Item ${i + 1}`}</span>
                          {item.speedup && (
                            <span className="font-mono text-accent-success">{item.speedup.toFixed(2)}x</span>
                          )}
                        </div>
                      ))}
                      {list.length === 0 && (
                        <div className="text-xs text-white/40">No data</div>
                      )}
                    </div>
                  );
                })}
                {!leaderboards && (
                  <div className="text-sm text-white/50 col-span-full">No leaderboard data returned.</div>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <Target className="w-5 h-5 text-accent-secondary" />
              <h3 className="font-medium text-white">Pareto Frontier</h3>
            </div>
          </div>
          <div className="card-body space-y-2">
            {remoteQuery.error ? (
              <ErrorState
                message={getErrorMessage(remoteQuery.error, 'Failed to load Pareto frontier')}
                onRetry={() => remoteQuery.mutate()}
              />
            ) : remoteQuery.isLoading ? (
              <LoadingState inline message="Fetching trade-offs..." />
            ) : (
              <>
                {(pareto?.frontier || pareto?.items || []).slice(0, 5).map((entry: any, i: number) => (
                  <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-white/5 border border-white/10">
                    <div>
                      <div className="text-white font-medium">{entry.name || entry.model || `Config ${i + 1}`}</div>
                      <div className="text-xs text-white/50">
                        {entry.latency && `Latency: ${entry.latency}ms`}{' '}
                        {entry.throughput && ` • TPS: ${entry.throughput}`}
                      </div>
                    </div>
                    {entry.score && (
                      <span className="text-accent-primary font-bold">{entry.score.toFixed?.(2) || entry.score}</span>
                    )}
                  </div>
                ))}
                {!pareto && (
                  <div className="text-sm text-white/50">No Pareto data available.</div>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-accent-warning" />
            <h3 className="font-medium text-white">Bottleneck Summary</h3>
          </div>
        </div>
        <div className="card-body space-y-2">
          {remoteQuery.error ? (
            <ErrorState
              message={getErrorMessage(remoteQuery.error, 'Failed to load bottlenecks')}
              onRetry={() => remoteQuery.mutate()}
            />
          ) : remoteQuery.isLoading ? (
            <LoadingState inline message="Loading bottleneck summary..." />
          ) : (bottlenecks?.items || bottlenecks?.summary || bottlenecks?.bottlenecks)?.length ? (
            (bottlenecks.items || bottlenecks.summary || bottlenecks.bottlenecks || []).slice(0, 5).map(
              (item: any, i: number) => (
                <div key={i} className="p-3 rounded-lg bg-white/5 border border-white/10 text-sm text-white/80">
                  {typeof item === 'string' ? item : item.description || item.name || `Bottleneck ${i + 1}`}
                </div>
              )
            )
          ) : (
            <EmptyState
              title="No bottlenecks reported"
              description="Run an analysis to surface top bottlenecks and recommendations."
            />
          )}
        </div>
      </div>

      <OptimizationStackingCard />
    </div>
  );
}
